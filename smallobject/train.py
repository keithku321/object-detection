# Plan C training entry point: trains one variant for one seed on VisDrone, evaluates AP by object size, appends a results row, resumable.
#   python -m smallobject.train --variant baseline --seed 0 --epochs 50 --project /content/drive/MyDrive/yoloProjectResults
#   python -m smallobject.train --variant p2 --seed 0 --epochs 50 --project ...
#   python -m smallobject.train --variant p2+eca+dysample+wfusion+nwd --seed 0 --epochs 50 --project ...
# A variant name is a list of features joined by "+", so the ideas can be added one at a time (the ablation the plan requires):
#   baseline   plain pretrained YOLO26 (S0)
#   p2         stride-4 detection head from yolo26-p2.yaml; only the backbone and the first neck layers start pretrained (S1)
#   cbam | eca | simam | ca   one attention block after every neck output that feeds the Detect head (S2)
#   dysample   DySample instead of nearest-neighbour upsampling in the neck (S3)
#   wfusion    learnable weighted fusion instead of plain concatenation in the neck (S4)
#   nwd        NWD box loss (S5); alone it applies to the baseline architecture
# Always read the log line "Transferred N/M items from pretrained weights": the P2 head and every new layer start untrained.
import argparse
from pathlib import Path

from common.benchmark import countParametersAndGflops
from common.registerModules import attentionBlocks, findConcatLayers, loadBaseYaml, makeLayerMapTrainer, registerCustomModules, writeVariantYaml
from common.results import appendResultRow, buildResultRow, findResultRow
from common.training import copyFinalWeights, defaultBatchSize, endToEndNote, makeRunName, protocolImageSize, trainDetector
from smallobject.evalCoco import cocoMaximumDetections, evaluateSmallObjects
from smallobject.nwdLoss import defaultNwdConstant, defaultNwdWeight, makeNwdTrainer

# Name of this task in the results log and in weight file names
taskName = "smallobject"
# Folder for the generated model YAML files, one per architecture variant
configFolder = Path(__file__).resolve().parent / "configs"
# Dataset YAML shipped with Ultralytics; it downloads and converts VisDrone on first use
defaultDataYaml = "VisDrone.yaml"
# Number of leading layers that yolo26.yaml and yolo26-p2.yaml have in common (layers 0 to 16), so they can start pretrained
sharedLayersWithP2 = 17
# YAML arguments of DySample: upsampling factor and number of channel groups
dySampleArguments = [2, 4]
# Feature names accepted in a variant name
knownFeatures = ("baseline", "p2", *attentionBlocks, "dysample", "wfusion", "nwd")


# Split a variant name such as "p2+eca+nwd" into a set of features and check them
def parseVariant(variantName):
    features = set(variantName.split("+"))
    unknownFeatures = features - set(knownFeatures)
    if unknownFeatures: raise ValueError(f"Unknown features {sorted(unknownFeatures)} in '{variantName}'. Known: {knownFeatures}")
    if len(features & set(attentionBlocks)) > 1: raise ValueError("Use at most one attention block per variant")
    if "baseline" in features and len(features) > 1: raise ValueError("'baseline' cannot be combined with other features")
    return features


# Turn a variant name into what the training loop needs: model source, pretrained setting and trainer class
def resolveVariant(variantName, scale="n", nwdWeight=defaultNwdWeight, nwdConstant=defaultNwdConstant):
    features = parseVariant(variantName)
    pretrainedWeights = f"yolo26{scale}.pt"
    architectureFeatures = features - {"baseline", "nwd"}
    if not architectureFeatures:
        return {"modelSource": pretrainedWeights, "pretrained": True, "trainerClass": makeNwdTrainer(nwdWeight, nwdConstant) if "nwd" in features else None}
    baseYamlName = "yolo26-p2.yaml" if "p2" in features else "yolo26.yaml"
    baseYaml = loadBaseYaml(baseYamlName)
    baseRows = baseYaml["backbone"] + baseYaml["head"]
    attentionName = next((feature for feature in features if feature in attentionBlocks), None)
    insertions = [(layerIndex, *attentionBlocks[attentionName]) for layerIndex in baseRows[-1][0]] if attentionName else []
    replacements = {index: ("DySample", dySampleArguments) for index, row in enumerate(baseRows) if row[2] == "nn.Upsample"} if "dysample" in features else None
    fileLabel = "-".join(sorted(architectureFeatures - {"p2"})) or "plain"
    yamlPath, insertionMap = writeVariantYaml(fileLabel, scale, baseYamlName, insertions, replacements, configFolder)
    pretrainedLayerCount = sharedLayersWithP2 if "p2" in features else len(baseRows)
    layerMap = {baseIndex: insertionMap[baseIndex] for baseIndex in range(pretrainedLayerCount)}
    weightedConcatLayers = ()
    if "wfusion" in features:
        import yaml
        weightedConcatLayers = tuple(findConcatLayers(yaml.safe_load(Path(yamlPath).read_text(encoding="utf-8"))))
    trainerClass = makeNwdTrainer(nwdWeight, nwdConstant, layerMap, weightedConcatLayers) if "nwd" in features else makeLayerMapTrainer(layerMap, weightedConcatLayers)
    return {"modelSource": yamlPath, "pretrained": pretrainedWeights, "trainerClass": trainerClass}


# Train one variant for one seed, evaluate AP by object size and log a results row; returns the path of the copied weights
def runVariant(variantName, seed, epochs, project, dataYaml=defaultDataYaml, scale="n", imageSize=protocolImageSize, batch=defaultBatchSize,
               device=None, workers=None, evalSplit="val", nwdWeight=defaultNwdWeight, nwdConstant=defaultNwdConstant,
               maximumDetections=cocoMaximumDetections, extraTrainArguments=None):
    registerCustomModules()
    variant = resolveVariant(variantName, scale, nwdWeight, nwdConstant)
    reportedName = variantName if imageSize == protocolImageSize else f"{variantName}@{imageSize}"
    runName = makeRunName(reportedName.replace("+", "-"), seed)
    bestWeights = trainDetector(variant["modelSource"], dataYaml, seed, epochs, Path(project) / taskName, runName, imageSize, batch, device, workers,
                                variant["trainerClass"], variant["pretrained"], trainArguments=extraTrainArguments)
    if findResultRow(taskName, f"yolo26{scale}", reportedName, seed, f"split={evalSplit};"):
        print(f"{reportedName} seed {seed} is already logged for split={evalSplit}; skipping evaluation")
    else:
        metrics = evaluateSmallObjects(bestWeights, dataYaml, evalSplit, imageSize, batch, device, maximumDetections)
        metrics.update(countParametersAndGflops(bestWeights, imageSize))
        notes = f"split={evalSplit}; {endToEndNote}; epochs={epochs}; imgsz={imageSize}; maxDets={maximumDetections}"
        if "nwd" in variantName: notes += f"; nwdWeight={nwdWeight}; nwdConstant={nwdConstant:.2f}"
        appendResultRow(buildResultRow(taskName, f"yolo26{scale}", reportedName, seed, metrics, notes))
        print({key: round(value, 4) for key, value in metrics.items() if key in ("map50", "map50_95", "apSmall", "apMedium", "apLarge", "parameters", "gflops")})
    return copyFinalWeights(bestWeights, Path(project) / "weights", taskName, reportedName.replace("+", "-"), seed)


# Command line entry point
def main():
    parser = argparse.ArgumentParser(description="Train one small-object detection variant")
    parser.add_argument("--variant", default="baseline", help="features joined by '+', for example p2+eca+nwd")
    parser.add_argument("--scale", default="n", choices=["n", "s", "m", "l", "x"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--data", default=defaultDataYaml)
    parser.add_argument("--project", help="folder for runs; use a folder on Google Drive in Colab")
    parser.add_argument("--imgsz", type=int, default=protocolImageSize)
    parser.add_argument("--batch", type=int, default=defaultBatchSize)
    parser.add_argument("--device", default=None)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--eval-split", default="val", choices=["val", "test"])
    parser.add_argument("--nwd-weight", type=float, default=defaultNwdWeight)
    parser.add_argument("--nwd-constant", type=float, default=defaultNwdConstant)
    parser.add_argument("--write-config", action="store_true", help="write the variant's model YAML and exit")
    arguments = parser.parse_args()
    if arguments.write_config:
        registerCustomModules()
        print(resolveVariant(arguments.variant, arguments.scale)["modelSource"])
        return
    if not arguments.project: parser.error("--project is required for training")
    runVariant(arguments.variant, arguments.seed, arguments.epochs, arguments.project, arguments.data, arguments.scale, arguments.imgsz, arguments.batch,
               arguments.device, arguments.workers, arguments.eval_split, arguments.nwd_weight, arguments.nwd_constant)


if __name__ == "__main__":
    main()
