# Make custom layers usable from Ultralytics YAML files and keep pretrained weights lined up after layers are inserted.
# Steward: plan C owner (plans/05-team-workflow.md). Used by Plan A (attention variants) and Plan C (attention, DySample, weighted fusion).
#
# Decisions behind this file (Phase 0 spike, checked against Ultralytics 8.4.175; record them in plans/decisions.md):
#  1. parse_model looks layer names up in the ultralytics.nn.tasks namespace, so registerCustomModules adds our classes there.
#  2. An unknown layer gets only its YAML arguments and keeps the input channel count, so layers that need the channel count
#     build themselves on the first forward pass (common/modules.py ChannelAwareModule).
#  3. Inserting a layer shifts the index of every later layer, so name-based weight loading would silently pair wrong layers.
#     insertLayersIntoYaml records a layer map and loadPretrainedWithLayerMap uses it, so "Transferred N/M" stays meaningful.
#  4. A multi-input custom layer such as a weighted Concat cannot be written in YAML (parse_model cannot size it), so
#     replaceConcatWithWeighted swaps it in after the model is built, and LayerMapTrainer repeats that on every rebuild and resume.
import copy
import json
import re
from pathlib import Path

from common.modules import CoordinateAttention, DySample, ECA, SimAM, WeightedConcat

# Layer names that can be used in a model YAML once registerCustomModules has run
customModules = {"ECA": ECA, "SimAM": SimAM, "CoordinateAttention": CoordinateAttention, "DySample": DySample}
# Placeholder that insertLayersIntoYaml replaces with the channel count of the layer an insertion follows
channelsPlaceholder = "{channels}"
# Attention blocks shared by Plans A and C: short name -> (YAML layer name, YAML arguments)
# CBAM is Ultralytics' own block and needs its channel count; the others come from common/modules.py and find it themselves
attentionBlocks = {
    "cbam": ("CBAM", [channelsPlaceholder, 7]),
    "eca": ("ECA", [0]),
    "simam": ("SimAM", [0.0001]),
    "ca": ("CoordinateAttention", [32]),
}
# Repository root, used to read the pinned Ultralytics version from requirements.txt
repositoryRoot = Path(__file__).resolve().parent.parent
# Folder for generated variant YAML files when a task does not choose its own
defaultConfigFolder = repositoryRoot / "configs_generated"


# The Ultralytics version pinned in requirements.txt, or None when the line cannot be found
def readPinnedUltralyticsVersion(requirementsPath=repositoryRoot / "requirements.txt"):
    match = re.search(r"^ultralytics==([\w.]+)", Path(requirementsPath).read_text(encoding="utf-8"), re.MULTILINE)
    return match.group(1) if match else None


# Warn when the installed Ultralytics differs from the pin, because custom code depends on its internals
def checkPinnedUltralyticsVersion():
    import ultralytics
    pinnedVersion = readPinnedUltralyticsVersion()
    if pinnedVersion and ultralytics.__version__ != pinnedVersion:
        print(f"WARNING: Ultralytics {ultralytics.__version__} is installed but requirements.txt pins {pinnedVersion}; custom layers and losses may break.")
    return pinnedVersion == ultralytics.__version__


# Add the custom layers to the ultralytics.nn.tasks namespace so YAML rows like [-1, 1, ECA, [3]] work; safe to call repeatedly
# Ultralytics' own CBAM is registered too: it exists in ultralytics.nn.modules but tasks.py does not import it (checked in 8.4.175)
def registerCustomModules():
    import ultralytics.nn.tasks as tasks
    from ultralytics.nn.modules.conv import CBAM
    checkPinnedUltralyticsVersion()
    for moduleName, moduleClass in {**customModules, "CBAM": CBAM}.items():
        setattr(tasks, moduleName, moduleClass)
    return sorted([*customModules, "CBAM"])


# Load a model YAML shipped with Ultralytics (for example "yolo26.yaml") as a dict
def loadBaseYaml(baseYamlName="yolo26.yaml"):
    import yaml
    from ultralytics.utils.checks import check_yaml
    return yaml.safe_load(Path(check_yaml(baseYamlName)).read_text(encoding="utf-8"))


# Put the scale letter into a base YAML name where Ultralytics expects it: yolo26.yaml -> yolo26n.yaml, yolo26-p2.yaml -> yolo26n-p2.yaml
# An optional suffix is appended before the file ending: ("yolo26-p2.yaml", "n", "eca") -> yolo26n-p2-eca.yaml
def scaledYamlName(baseYamlName, scale, suffix=""):
    match = re.match(r"^(yolo[a-z]*\d+)(.*)$", Path(baseYamlName).stem)
    if not match: raise ValueError(f"Cannot find the model family in '{baseYamlName}'")
    return f"{match.group(1)}{scale}{match.group(2)}{'-' + suffix if suffix else ''}.yaml"


# Output channel count of every layer of a scaled base model, measured with a dummy forward pass; layers without a tensor output are left out
# Needed for built-in layers such as CBAM, whose YAML argument is a channel count that Ultralytics does not scale with the model width
def measureLayerChannels(baseYamlName="yolo26.yaml", scale="n"):
    import torch
    from ultralytics.nn.tasks import DetectionModel
    network = DetectionModel(scaledYamlName(baseYamlName, scale), verbose=False).eval()
    channels = {}

    # Forward hook factory that records the channel count of one layer's tensor output
    def makeHook(layerIndex):
        # Store the channel count when the layer returns a feature map
        def hook(module, inputs, output):
            if isinstance(output, torch.Tensor) and output.ndim == 4: channels[layerIndex] = output.shape[1]
        return hook

    for layerIndex, layer in enumerate(network.model): layer.register_forward_hook(makeHook(layerIndex))
    with torch.no_grad():
        network(torch.zeros(1, 3, 128, 128))
    return channels


# Insert new layers into a model YAML and return (new YAML dict, layer map from base layer index to new layer index)
# insertions is a list of (afterLayerIndex, moduleName, args); the new layer reads from the layer before it, and every later
# reference to that base layer (a Concat or the Detect head) is rewired to the new layer, so the attended features are used everywhere.
# An argument equal to channelsPlaceholder becomes the channel count of the layer it follows (layerChannels maps layer index to channels).
# replacements maps a base layer index to (moduleName, args) for layers swapped in place without shifting any index (nn.Upsample to DySample)
def insertLayersIntoYaml(baseYaml, insertions=(), replacements=None, layerChannels=None):
    replacements = replacements or {}
    baseRows = baseYaml["backbone"] + baseYaml["head"]
    backboneLength = len(baseYaml["backbone"])
    insertionCounts = {}
    for afterLayer, _, _ in insertions:
        if not 0 <= afterLayer < len(baseRows): raise ValueError(f"Cannot insert after layer {afterLayer}; the model has {len(baseRows)} layers")
        insertionCounts[afterLayer] = insertionCounts.get(afterLayer, 0) + 1

    # New index of a base layer: its old index plus the number of layers inserted before it
    def shiftedIndex(baseIndex):
        return baseIndex + sum(count for afterLayer, count in insertionCounts.items() if afterLayer < baseIndex)

    # Index of the layer whose output stands for a base layer: the last layer inserted after it, or the layer itself
    def outputIndex(baseIndex):
        return shiftedIndex(baseIndex) + insertionCounts.get(baseIndex, 0)

    # Rewire one "from" entry; -1 keeps meaning "the previous layer", which is the inserted layer when one follows
    def rewire(fromEntry, rowIndex):
        if isinstance(fromEntry, list): return [rewire(entry, rowIndex) for entry in fromEntry]
        if fromEntry == -1: return -1
        absoluteIndex = fromEntry if fromEntry >= 0 else rowIndex + fromEntry
        return outputIndex(absoluteIndex)

    newRows, layerMap = [], {}
    for baseIndex, (fromEntry, repeats, moduleName, args) in enumerate(baseRows):
        if baseIndex in replacements: moduleName, args = replacements[baseIndex]
        newRows.append([rewire(fromEntry, baseIndex), repeats, moduleName, copy.deepcopy(args)])
        layerMap[baseIndex] = len(newRows) - 1
        for afterLayer, newModuleName, newArgs in insertions:
            if afterLayer != baseIndex: continue
            if channelsPlaceholder in newArgs and not layerChannels: raise ValueError(f"{newModuleName} needs layerChannels to fill in its channel count")
            filledArgs = [layerChannels[afterLayer] if argument == channelsPlaceholder else argument for argument in newArgs]
            newRows.append([-1, 1, newModuleName, filledArgs])
    newBackboneLength = backboneLength + sum(count for afterLayer, count in insertionCounts.items() if afterLayer < backboneLength)
    newYaml = {key: value for key, value in baseYaml.items() if key not in ("backbone", "head")}
    newYaml["backbone"], newYaml["head"] = newRows[:newBackboneLength], newRows[newBackboneLength:]
    return newYaml, layerMap


# Format one top-level YAML value; JSON syntax is valid YAML flow syntax, so dicts, lists, strings and booleans all use json.dumps
def formatYamlValue(value):
    return json.dumps(value)


# Write a model YAML to a file with one layer per line; the file name must contain the scale letter (for example yolo26n-eca.yaml)
def writeModelYaml(yamlDict, outputPath, headerComment=""):
    lines = [f"# {line}" for line in headerComment.splitlines()] if headerComment else []
    for key, value in yamlDict.items():
        if key in ("backbone", "head"): continue
        lines.append(f"{key}: {formatYamlValue(value)}")
    for sectionName in ("backbone", "head"):
        lines.append(f"{sectionName}:")
        lines.extend(f"  - {json.dumps(row)}" for row in yamlDict[sectionName])
    outputPath = Path(outputPath)
    outputPath.parent.mkdir(parents=True, exist_ok=True)
    outputPath.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return outputPath


# Build and write a variant YAML from a base YAML plus insertions and replacements; returns (yaml path, layer map)
# The file is named with the scale letter right after the family, for example yolo26n-eca.yaml or yolo26n-p2-eca.yaml, because
# Ultralytics reads the model scale from the file name
def writeVariantYaml(variantName, scale="n", baseYamlName="yolo26.yaml", insertions=(), replacements=None, outputFolder=defaultConfigFolder):
    baseYaml = loadBaseYaml(baseYamlName)
    needsChannels = any(channelsPlaceholder in list(args) for _, _, args in insertions)
    layerChannels = measureLayerChannels(baseYamlName, scale) if needsChannels else None
    variantYaml, layerMap = insertLayersIntoYaml(baseYaml, insertions, replacements, layerChannels)
    outputPath = Path(outputFolder) / scaledYamlName(baseYamlName, scale, variantName)
    description = f"Generated by common/registerModules.py from {baseYamlName}; insertions={list(insertions)}, replacements={replacements}"
    writeModelYaml(variantYaml, outputPath, description)
    return outputPath, layerMap


# Identity layer map for models that share their first layerCount layers with the base model (for example the P2 variant)
def prefixLayerMap(layerCount):
    return {index: index for index in range(layerCount)}


# Module that looks like a pretrained model whose layers sit at the variant's indices, so Ultralytics' own load() can transfer them
# Positions without a pretrained counterpart hold an Identity layer, which has no parameters and so transfers nothing
def buildIndexAlignedProxy(pretrainedModel, layerMap, variantLayerCount):
    import torch.nn as nn

    # Stand-in for the pretrained model: same weights, but every layer sits at the index it has in the variant
    class IndexAlignedPretrained(nn.Module):
        # Hold the re-indexed layers and copy the attributes that Ultralytics' load() reads
        def __init__(self):
            super().__init__()
            layers = [nn.Identity() for _ in range(variantLayerCount)]
            for baseIndex, variantIndex in layerMap.items(): layers[variantIndex] = pretrainedModel.model[baseIndex]
            self.model = nn.Sequential(*layers)
            self.names = getattr(pretrainedModel, "names", {})
            self.pt_path = getattr(pretrainedModel, "pt_path", None)

        # A proxy is never treated as fused, so load() takes the unfused path
        def is_fused(self):
            return False

    return IndexAlignedPretrained()


# Load pretrained weights into a variant network using a layer map; prints Ultralytics' "Transferred N/M items" line
# pretrainedSource is a .pt path or a loaded Ultralytics model; the layer map comes from insertLayersIntoYaml or prefixLayerMap
def loadPretrainedWithLayerMap(network, pretrainedSource, layerMap):
    from ultralytics.nn.tasks import load_checkpoint
    pretrainedModel = pretrainedSource if hasattr(pretrainedSource, "model") else load_checkpoint(str(pretrainedSource))[0]
    network.load(buildIndexAlignedProxy(pretrainedModel, layerMap, len(network.model)))
    return network


# Replace Concat layers (by their layer index in the built network) with WeightedConcat; the layers keep their index and wiring
def replaceConcatWithWeighted(network, layerIndices):
    from ultralytics.nn.modules import Concat
    for layerIndex in layerIndices:
        oldLayer = network.model[layerIndex]
        if type(oldLayer) is not Concat: raise TypeError(f"Layer {layerIndex} is {type(oldLayer).__name__}, not Concat")
        newLayer = WeightedConcat(inputCount=len(oldLayer.f), dimension=oldLayer.d)
        newLayer.i, newLayer.f, newLayer.type, newLayer.np = oldLayer.i, oldLayer.f, "common.modules.WeightedConcat", newLayer.fusionWeights.numel()
        network.model[layerIndex] = newLayer
    return network


# Concat layer indices of a model YAML, so a task can pass them to replaceConcatWithWeighted
def findConcatLayers(modelYaml):
    return [index for index, row in enumerate(modelYaml["backbone"] + modelYaml["head"]) if row[2] == "Concat"]


# Build a detection trainer class that (a) loads pretrained weights through a layer map, (b) swaps in weighted Concat layers on
# every model rebuild including resume, and (c) can be extended through the createModel hook. Pass the class to model.train(trainer=...).
# layerMap None means the stock name-based transfer; weightedConcatLayers is a tuple of layer indices of the variant model.
def makeLayerMapTrainer(layerMap=None, weightedConcatLayers=()):
    from ultralytics.models.yolo.detect import DetectionTrainer

    # Detection trainer whose model building knows about inserted layers and weighted Concat layers
    class LayerMapTrainer(DetectionTrainer):
        # Layer map and weighted-Concat positions are class attributes because Ultralytics builds the trainer itself
        variantLayerMap = layerMap
        variantWeightedConcatLayers = tuple(weightedConcatLayers)

        # Hook for subclasses that need another DetectionModel class (for example one with a custom loss)
        def createModel(self, cfg, verbose):
            return super().get_model(cfg=cfg, weights=None, verbose=verbose)

        # Build the variant, swap weighted Concat layers, then load either a same-architecture checkpoint or mapped pretrained weights
        def get_model(self, cfg=None, weights=None, verbose=True):
            model = self.createModel(cfg, verbose)
            replaceConcatWithWeighted(model, self.variantWeightedConcatLayers)
            if weights is None: return model
            sameArchitecture = len(weights.model) == len(model.model) and all(type(a) is type(b) for a, b in zip(weights.model, model.model))
            if self.variantLayerMap is None or sameArchitecture: model.load(weights)
            else: loadPretrainedWithLayerMap(model, weights, self.variantLayerMap)
            return model

    return LayerMapTrainer
