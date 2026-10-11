# Plan A training entry point: trains one variant for one seed, appends a row to results/underwater.csv, and is resumable.
#   python -m underwater.train --variant baseline --seed 0 --epochs 50 --data /content/datasets/brackish/dataset.yaml --project /content/drive/MyDrive/yoloProjectResults/underwater
#   python -m underwater.train --variant eca-a --scale n --seed 0 --epochs 50 --data ... --project ...
#   python -m underwater.train --write-configs        (regenerate underwater/configs/*.yaml and exit)
# Variant names: baseline, yolo11, yolov8 (plain pretrained references), and <block>-<placement> such as cbam-a, eca-b, simam-c, ca-a.
# Blocks: cbam (built in), eca, simam, ca (common/modules.py). Placements (layer numbers of the base yolo26 YAML):
#   a = after backbone stages 4, 6, 8   b = after neck outputs 16, 19, 22   c = both
import argparse
from pathlib import Path

from common.registerModules import attentionBlocks, makeLayerMapTrainer, registerCustomModules, writeVariantYaml
from common.training import copyFinalWeights, defaultBatchSize, evaluateAndLog, makeRunName, protocolImageSize, trainDetector

# Name of this task in the results log and in weight file names
taskName = "underwater"
# Folder for the generated model YAML files, one per attention variant
configFolder = Path(__file__).resolve().parent / "configs"
# Placements: variant suffix -> base layer indices after which the attention block is inserted
placements = {"a": (4, 6, 8), "b": (16, 19, 22), "c": (4, 6, 8, 16, 19, 22)}
# Plain pretrained references: variant name -> (model family used in the results row, pretrained weights name with the scale letter)
plainVariants = {"baseline": ("yolo26", "yolo26{scale}.pt"), "yolo11": ("yolo11", "yolo11{scale}.pt"), "yolov8": ("yolov8", "yolov8{scale}.pt")}


# Write the YAML of one attention variant and return (YAML path, layer map)
def buildAttentionVariant(variantName, scale):
    blockName, placementName = variantName.split("-")
    moduleName, moduleArguments = attentionBlocks[blockName]
    insertions = [(layerIndex, moduleName, moduleArguments) for layerIndex in placements[placementName]]
    return writeVariantYaml(variantName, scale, "yolo26.yaml", insertions=insertions, outputFolder=configFolder)


# Every attention variant name, for example "eca-a"
def listAttentionVariants():
    return [f"{blockName}-{placementName}" for blockName in attentionBlocks for placementName in placements]


# Turn a variant name into what the training loop needs: model name, model source, pretrained setting and trainer class
def resolveVariant(variantName, scale):
    if variantName in plainVariants:
        family, weightsName = plainVariants[variantName]
        return {"modelName": f"{family}{scale}", "modelSource": weightsName.format(scale=scale), "pretrained": True, "trainerClass": None}
    if variantName in listAttentionVariants():
        yamlPath, layerMap = buildAttentionVariant(variantName, scale)
        return {"modelName": f"yolo26{scale}", "modelSource": yamlPath, "pretrained": f"yolo26{scale}.pt", "trainerClass": makeLayerMapTrainer(layerMap)}
    raise ValueError(f"Unknown variant '{variantName}'. Known: {sorted(plainVariants) + listAttentionVariants()}")


# Train one variant for one seed, log a results row and copy the final weights; returns the path of the copied weights
# label overrides the variant name in the run folder and results row (for example "enhance-baseline" when data is the enhanced copy)
def runVariant(variantName, seed, epochs, dataYaml, project, scale="n", imageSize=protocolImageSize, batch=defaultBatchSize,
               device=None, workers=None, evalSplit="val", label=None, weightsFolder=None, extraTrainArguments=None):
    registerCustomModules()
    variant = resolveVariant(variantName, scale)
    reportedName = label or variantName
    runName = makeRunName(reportedName, seed)
    bestWeights = trainDetector(variant["modelSource"], dataYaml, seed, epochs, Path(project) / taskName, runName, imageSize, batch,
                                device, workers, variant["trainerClass"], variant["pretrained"], trainArguments=extraTrainArguments)
    evaluateAndLog(taskName, variant["modelName"], reportedName, seed, bestWeights, dataYaml, evalSplit, imageSize, batch, device, notes=f"epochs={epochs}")
    return copyFinalWeights(bestWeights, weightsFolder or Path(project) / "weights", taskName, reportedName, seed)


# Command line entry point
def main():
    parser = argparse.ArgumentParser(description="Train one underwater detection variant")
    parser.add_argument("--variant", default="baseline")
    parser.add_argument("--scale", default="n", choices=["n", "s", "m", "l", "x"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--data", help="dataset.yaml of the training data")
    parser.add_argument("--project", help="folder for runs; use a folder on Google Drive in Colab")
    parser.add_argument("--imgsz", type=int, default=protocolImageSize)
    parser.add_argument("--batch", type=int, default=defaultBatchSize)
    parser.add_argument("--device", default=None)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--eval-split", default="val", choices=["val", "test"])
    parser.add_argument("--label", default=None)
    parser.add_argument("--write-configs", action="store_true", help="write all attention variant YAML files and exit")
    arguments = parser.parse_args()
    if arguments.write_configs:
        registerCustomModules()
        for variantName in listAttentionVariants(): print(buildAttentionVariant(variantName, arguments.scale)[0])
        return
    if not arguments.data or not arguments.project: parser.error("--data and --project are required for training")
    runVariant(arguments.variant, arguments.seed, arguments.epochs, arguments.data, arguments.project, arguments.scale, arguments.imgsz,
               arguments.batch, arguments.device, arguments.workers, arguments.eval_split, arguments.label)


if __name__ == "__main__":
    main()
