# One training, resume, evaluate and log loop for the Ultralytics-based plans (A, B and C), so every variant is trained
# under the same protocol (plans/00-main-plan.md section 6). A task passes its own model source, trainer class and arguments.
# Proposed steward: plan B owner (same area as common/results.py). Changes need review like any shared file.
import shutil
from pathlib import Path

from common.benchmark import countParametersAndGflops
from common.environment import getDefaultDevice, getDefaultWorkerCount
from common.plotResults import extractPerClassScores, extractRowMetrics, plotDetectionRun, trySavePlot
from common.results import appendResultRow, buildResultRow, extractDetectionMetrics, findResultRow

# Image size used for training and validation in every comparison
protocolImageSize = 640
# Batch size that fits an 8 GB laptop GPU and a free Colab T4 for nano and small models at 640 px
defaultBatchSize = 16
# Seeds for the final multi-seed runs; exploration may use only the first one
finalSeeds = (0, 1, 2)
# Which YOLO26 head produces the reported accuracy. In Ultralytics 8.4.175, val, predict and the validation inside training use the
# one-to-many head with external NMS unless nms=False is passed (the default nms=None means external NMS). The speed protocol times
# the NMS-free one-to-one head, so accuracy is measured on that same head: every train, val and predict call passes these arguments.
# Models without a one-to-one head (YOLO11, YOLOv8 reference rows) ignore the setting and keep using NMS.
# This is a protocol decision for the whole team: change it only by agreement and record it in plans/decisions.md.
endToEndArguments = {"nms": False}
# Text added to the notes of every results row so the head used for a number is never ambiguous
endToEndNote = "head=one2one(nms=False)"


# Name of the run folder for one variant and seed, for example "cbam-seed1"
def makeRunName(variant, seed):
    return f"{variant}-seed{seed}"


# Path of the last checkpoint if training stopped before finishing, otherwise None
# A finished run has its optimiser state stripped from last.pt (stored under Ultralytics' key "optimizer"), so a missing one means there is nothing to resume
def findResumableCheckpoint(runFolder):
    lastPath = Path(runFolder) / "weights" / "last.pt"
    if not lastPath.exists(): return None
    import torch
    checkpoint = torch.load(lastPath, map_location="cpu", weights_only=False)
    return lastPath if checkpoint.get("optimizer") is not None else None


# Train one detector. The run folder is fixed by project and runName, so a restarted session continues from last.pt on Drive.
# modelSource is a .pt path (pretrained start) or a model YAML; pretrained is True, False or a .pt path used to initialise a YAML model.
# trainerClass is an optional Ultralytics trainer subclass (layer map trainer, distillation trainer, NWD trainer).
# Returns the path of best.pt.
def trainDetector(modelSource, dataYaml, seed, epochs, project, runName, imageSize=protocolImageSize, batch=defaultBatchSize,
                  device=None, workers=None, trainerClass=None, pretrained=True, resume=True, trainArguments=None):
    from ultralytics import YOLO
    runFolder = Path(project).resolve() / runName
    trainerArguments = {"trainer": trainerClass} if trainerClass else {}
    resumableCheckpoint = findResumableCheckpoint(runFolder) if resume else None
    if resumableCheckpoint:
        print(f"Resuming {runName} from {resumableCheckpoint}")
        YOLO(str(resumableCheckpoint)).train(resume=True, data=str(dataYaml), **trainerArguments)
    elif (runFolder / "weights" / "best.pt").exists() and resume:
        print(f"{runName} already finished; skipping training")
    else:
        arguments = dict(
            data=str(dataYaml), epochs=epochs, imgsz=imageSize, batch=batch, seed=seed, deterministic=True,
            project=str(Path(project).resolve()), name=runName, exist_ok=True, pretrained=pretrained,
            device=device or getDefaultDevice(), workers=getDefaultWorkerCount() if workers is None else workers,
            **endToEndArguments,
        )
        arguments.update(trainArguments or {})
        YOLO(str(modelSource)).train(**trainerArguments, **arguments)
    return runFolder / "weights" / "best.pt"


# Validate a trained model on a split and return the Ultralytics metrics object
# The dataloader uses the same worker count as training, which is kept small on Windows where many workers hang or run slowly
def evaluateDetector(weightsPath, dataYaml, split="val", imageSize=protocolImageSize, batch=defaultBatchSize, device=None, extraArguments=None):
    from ultralytics import YOLO
    arguments = dict(data=str(dataYaml), split=split, imgsz=imageSize, batch=batch, device=device or getDefaultDevice(), plots=False, verbose=False,
                     workers=getDefaultWorkerCount(), **endToEndArguments)
    arguments.update(extraArguments or {})
    return YOLO(str(weightsPath)).val(**arguments)


# Save the training-results plot of a finished run into plots/ inside its run folder (weights sit in <run folder>/weights/)
# finalMetrics are the numbers of the final evaluation and perClass the per-class scores, when available
def savePlotOfRun(task, weightsPath, finalMetrics=None, perClass=None, notes=""):
    runFolder = Path(weightsPath).resolve().parent.parent
    return trySavePlot(plotDetectionRun, runFolder, task, runFolder.name, finalMetrics, perClass, notes)


# Evaluate a trained model, add parameter and GFLOP counts, and append one row to results/<task>.csv; returns the row
# A run that already has a row for the same split is not evaluated or logged again, so cells can be re-run after a Colab restart;
# pass relog=True to force a new row. savePlot=True also saves the results plot of the run (see common/plotResults.py).
def evaluateAndLog(task, modelName, variant, seed, weightsPath, dataYaml, split="val", imageSize=protocolImageSize,
                   batch=defaultBatchSize, device=None, extraMetrics=None, extraArguments=None, notes="", resultsFolder=None, relog=False, savePlot=True):
    existingRow = None if relog else (findResultRow(task, modelName, variant, seed, f"split={split};") if resultsFolder is None
                                      else findResultRow(task, modelName, variant, seed, f"split={split};", resultsFolder))
    if existingRow:
        print(f"{variant} seed {seed} is already logged for split={split}; skipping evaluation")
        if savePlot: savePlotOfRun(task, weightsPath, extractRowMetrics(existingRow), notes=existingRow.get("notes", ""))
        return existingRow
    validationMetrics = evaluateDetector(weightsPath, dataYaml, split, imageSize, batch, device, extraArguments)
    metrics = extractDetectionMetrics(validationMetrics)
    metrics.update(countParametersAndGflops(weightsPath, imageSize))
    if extraMetrics: metrics.update(extraMetrics)
    headNote = endToEndNote if (extraArguments or {}).get("nms", endToEndArguments["nms"]) is False else "head=one2many(NMS)"
    row = buildResultRow(task, modelName, variant, seed, metrics, notes=f"split={split}; {headNote}; {notes}".strip("; "))
    if resultsFolder is None: appendResultRow(row)
    else: appendResultRow(row, resultsFolder)
    print({key: row[key] for key in ("task", "model", "variant", "seed", "map50", "map50_95", "parameters", "gflops")})
    if savePlot: savePlotOfRun(task, weightsPath, extractRowMetrics(row), extractPerClassScores(validationMetrics), row["notes"])
    return row


# Copy best.pt to <weightsFolder>/<task>-<variant>-seed<k>.pt, the naming rule for weights that the demo or a release uses
def copyFinalWeights(bestWeightsPath, weightsFolder, task, variant, seed):
    weightsFolder = Path(weightsFolder)
    weightsFolder.mkdir(parents=True, exist_ok=True)
    targetPath = weightsFolder / f"{task}-{variant}-seed{seed}.pt"
    shutil.copy2(bestWeightsPath, targetPath)
    return targetPath
