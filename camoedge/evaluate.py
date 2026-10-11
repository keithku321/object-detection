# Evaluation for CamoEdge: per-test-set metrics, a per-super-class table for COD10K, and rows in results/camoedge.csv.
#   python -m camoedge.evaluate --checkpoint last.pt --dataset /content/datasets/camo
# Predictions are resized to each image's original mask size before scoring, as the standard protocol does.
import argparse
import csv
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from camoedge.dataset import buildTestLoader, collectTestPairs, defaultImageSize, superClassOf
from camoedge.metrics import CamoMetrics
from camoedge.model import CamoEdgeNet, countParameters
from common.environment import getDefaultDevice, getDefaultWorkerCount
from common.results import appendResultRow, buildResultRow, findResultRow

# Name of this task in the results log
taskName = "camoedge"


# Rebuild the network stored in a training checkpoint and load its weights; returns (model in eval mode, checkpoint settings)
def loadCheckpointModel(checkpointPath, device=None):
    device = device or getDefaultDevice()
    checkpoint = torch.load(checkpointPath, map_location="cpu", weights_only=False)
    settings = checkpoint["settings"]
    model = CamoEdgeNet(encoderName=settings["encoderName"], pretrained=False, **settings["modelSettings"])
    model.load_state_dict(checkpoint["model"])
    return model.to(device).eval(), settings


# Probability map of the final mask output for one normalised image batch, resized to the given (height, width); returns a numpy array
@torch.inference_mode()
def predictProbability(model, imageTensor, outputSize):
    logits = model(imageTensor)["masks"][0]
    return torch.sigmoid(F.interpolate(logits.float(), size=outputSize, mode="bilinear", align_corners=False))[0, 0].cpu().numpy()


# Score a model on a list of (image, mask) pairs; returns the overall metrics and, when groupBySuperClass is set, metrics per super-class
def evaluatePairs(model, pairs, imageSize=defaultImageSize, device=None, groupBySuperClass=False, workers=None, maximumCount=None):
    device = device or getDefaultDevice()
    overall, groups = CamoMetrics(), {}
    loader = buildTestLoader(pairs, imageSize, getDefaultWorkerCount() if workers is None else workers, maximumCount)
    for batch in loader:
        trueMask = np.asarray(Image.open(batch["maskPath"][0]).convert("L")) > 127
        probability = predictProbability(model, batch["image"].to(device), trueMask.shape)
        overall.update(probability, trueMask)
        if groupBySuperClass: groups.setdefault(superClassOf(batch["name"][0]), CamoMetrics()).update(probability, trueMask)
    return overall.compute(), {groupName: metrics.compute() for groupName, metrics in sorted(groups.items())}


# Evaluate a checkpoint on every available test set
# Returns ({test set: metrics}, {super-class: metrics} for COD10K, the checkpoint settings, the parameter count)
def evaluateCheckpoint(checkpointPath, datasetRoot, device=None, maximumCount=None):
    model, settings = loadCheckpointModel(checkpointPath, device)
    metricsByTestSet, perCategory = {}, {}
    for testName, pairs in collectTestPairs(datasetRoot).items():
        metrics, groups = evaluatePairs(model, pairs, settings["imageSize"], device, groupBySuperClass=testName == "COD10K", maximumCount=maximumCount)
        metricsByTestSet[testName] = metrics
        if groups: perCategory = groups
        print(testName, {key: round(value, 4) for key, value in metrics.items()})
    return metricsByTestSet, perCategory, settings, countParameters(model)


# Write the per-super-class table next to the checkpoint, because it has too many rows for the shared results file
def writeCategoryTable(perCategory, outputPath):
    if not perCategory: return None
    columns = ["superClass"] + list(next(iter(perCategory.values())))
    with Path(outputPath).open("w", newline="", encoding="utf-8") as csvFile:
        writer = csv.DictWriter(csvFile, fieldnames=columns)
        writer.writeheader()
        for superClass, metrics in perCategory.items(): writer.writerow({"superClass": superClass, **{key: round(value, 6) for key, value in metrics.items()}})
    return Path(outputPath)


# Evaluate a checkpoint and append one results row per test set; the variant column is "<experiment>@<test set>"
def evaluateAndLog(checkpointPath, datasetRoot, device=None, maximumCount=None, notes=""):
    metricsByTestSet, perCategory, settings, parameterCount = evaluateCheckpoint(checkpointPath, datasetRoot, device, maximumCount)
    for testName, metrics in metricsByTestSet.items():
        metrics = {**metrics, "parameters": parameterCount}
        rowNotes = f"epochs={settings['epochs']}; imgsz={settings['imageSize']}; checkpoint=last; {notes}".strip("; ")
        variantLabel = f"{settings['variantName']}@{testName}"
        if findResultRow(taskName, settings["encoderName"], variantLabel, settings["seed"], f"epochs={settings['epochs']};"): continue
        appendResultRow(buildResultRow(taskName, settings["encoderName"], variantLabel, settings["seed"], metrics, rowNotes))
    writeCategoryTable(perCategory, Path(checkpointPath).parent / "perCategory.csv")
    return metricsByTestSet


# Command line entry point
def main():
    parser = argparse.ArgumentParser(description="Evaluate a CamoEdge checkpoint on the test sets")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset", required=True, help="dataset root that contains TestDataset/")
    parser.add_argument("--device", default=None)
    parser.add_argument("--no-log", action="store_true", help="print the metrics without appending results rows")
    arguments = parser.parse_args()
    if arguments.no_log: evaluateCheckpoint(arguments.checkpoint, arguments.dataset, arguments.device)
    else: evaluateAndLog(arguments.checkpoint, arguments.dataset, arguments.device)


if __name__ == "__main__":
    main()
