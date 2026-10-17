# Plan B training entry point: one fold and one seed per call, resumable, with a results row per run and per weather type.
#   python -m weather.train --mode student --fold 0 --seed 0 --epochs 100 --dataset /content/datasets/dawn --project /content/drive/MyDrive/yoloProjectResults
#   python -m weather.train --mode teacher --fold 0 --seed 0 ...
#   python -m weather.train --mode full --fold 0 --seed 0 ...          (needs the teacher of that fold)
# Modes: student (W0, plain yolo26n), teacher (W1, plain yolo26m), and the distillation presets of weather/distillTrainer.py:
#   lambda0 (sanity check), feat (W3), cls and box (W4), full (W5), full-fg and full-both-heads (W6).
#
# Two rules that keep the 5-fold numbers honest:
#  - The teacher of fold k is trained on the training folds of fold k only, never on the held-out fold.
#  - Reported numbers come from last.pt (the final epoch), not best.pt: best.pt is chosen on the held-out fold, which would be
#    model selection on the test data.
import argparse
from pathlib import Path

from common.benchmark import countParametersAndGflops
from common.plotResults import extractRowMetrics, plotGroupedMetrics, trySavePlot
from common.results import findResultRow
from common.training import copyFinalWeights, defaultBatchSize, evaluateAndLog, protocolImageSize, savePlotOfRun, trainDetector
from weather.distillTrainer import distillPresets, trainDistilledStudent
from weather.evalByWeather import logByWeather
from weather.prepareDawn import buildFoldYamls, foldCount, getDawnSettings

# Name of this task in the results log and in weight file names
taskName = "weather"
# Model scale letters of the student and the teacher
studentScale, teacherScale = "n", "m"
# Seed of the teacher that every student of a fold learns from
teacherSeed = 0


# Run folder name of one mode, fold and seed, for example "distill-full-fold2-seed1"
def makeFoldRunName(variant, foldIndex, seed):
    return f"{variant}-fold{foldIndex}-seed{seed}"


# Path of the trained teacher of a fold (its final checkpoint)
def getTeacherWeights(project, foldIndex):
    return Path(project) / taskName / makeFoldRunName("teacher", foldIndex, teacherSeed) / "weights" / "last.pt"


# True when a distilled student has exactly as many parameters as the plain student, which proves nothing of the teacher was kept
def checkStudentSize(distilledWeights, plainStudentWeights):
    distilledCount = countParametersAndGflops(distilledWeights)["parameters"]
    plainCount = countParametersAndGflops(plainStudentWeights)["parameters"]
    print(f"Parameters: distilled {distilledCount:,}, plain student {plainCount:,} -> {'OK' if distilledCount == plainCount else 'MISMATCH'}")
    return distilledCount == plainCount


# Train and evaluate one mode on one fold with one seed; returns the path of the reported weights
# overrides changes single distillation settings (lambdaFeature, lambdaClass, lambdaBox, temperature, ...) for sweeps;
# label then names the variant in the results row so sweep points stay apart
def runFold(mode, foldIndex, seed, epochs, datasetRoot, project, imageSize=protocolImageSize, batch=defaultBatchSize, device=None,
            workers=None, overrides=None, label=None):
    _, classNames = getDawnSettings()
    dataYaml = buildFoldYamls(datasetRoot, classNames)[foldIndex]
    projectFolder = Path(project) / taskName
    if mode in ("student", "teacher"):
        scale = studentScale if mode == "student" else teacherScale
        variant, modelName = label or mode, f"yolo26{scale}"
        runName = makeFoldRunName(variant, foldIndex, seed)
        bestWeights = trainDetector(f"yolo26{scale}.pt", dataYaml, seed, epochs, projectFolder, runName, imageSize, batch, device, workers)
    elif mode in distillPresets:
        teacherWeights = getTeacherWeights(project, foldIndex)
        if not teacherWeights.exists(): raise FileNotFoundError(f"Train the teacher of fold {foldIndex} first (--mode teacher); expected {teacherWeights}")
        variant, modelName = label or f"distill-{mode}", f"yolo26{studentScale}"
        runName = makeFoldRunName(variant, foldIndex, seed)
        bestWeights = trainDistilledStudent(f"yolo26{studentScale}.pt", teacherWeights, dataYaml, seed, epochs, projectFolder, runName, mode, overrides,
                                            imageSize, batch, device, workers)
    else:
        raise ValueError(f"Unknown mode '{mode}'. Known: student, teacher, {sorted(distillPresets)}")
    reportedWeights = bestWeights.parent / "last.pt"
    notes = f"fold={foldIndex}; epochs={epochs}; checkpoint=last" + (f"; overrides={overrides}" if overrides else "")
    loggedRow = findResultRow(taskName, modelName, variant, seed, f"fold={foldIndex};")
    if loggedRow:
        print(f"{variant} fold {foldIndex} seed {seed} is already logged; skipping evaluation")
        savePlotOfRun(taskName, reportedWeights, extractRowMetrics(loggedRow), notes=loggedRow.get("notes", ""))
    else: evaluateAndLog(taskName, modelName, variant, seed, reportedWeights, dataYaml, "val", imageSize, batch, device, notes=notes, relog=True)
    metricsByWeather = logByWeather(modelName, variant, seed, reportedWeights, datasetRoot, foldIndex, imageSize, device, notes=f"epochs={epochs}; checkpoint=last")
    trySavePlot(plotGroupedMetrics, taskName, runName, "by-weather", metricsByWeather, groupTitle="weather", notes=f"fold {foldIndex}")
    return copyFinalWeights(reportedWeights, Path(project) / "weights", taskName, f"{variant}-fold{foldIndex}", seed)


# Run one mode over several folds and seeds (the final 5-fold x 3-seed tables); finished runs are skipped, so it can be restarted
def runAllFolds(mode, seeds, epochs, datasetRoot, project, folds=range(foldCount), **runArguments):
    return [runFold(mode, foldIndex, seed, epochs, datasetRoot, project, **runArguments) for foldIndex in folds for seed in seeds]


# Command line entry point
def main():
    parser = argparse.ArgumentParser(description="Train one WeatherDistill mode on one DAWN fold")
    parser.add_argument("--mode", required=True, help=f"student, teacher or one of {sorted(distillPresets)}")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--dataset", required=True, help="root of the prepared DAWN dataset (weather/prepareDawn.py)")
    parser.add_argument("--project", required=True, help="folder for runs; use a folder on Google Drive in Colab")
    parser.add_argument("--imgsz", type=int, default=protocolImageSize)
    parser.add_argument("--batch", type=int, default=defaultBatchSize)
    parser.add_argument("--device", default=None)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--lambda-feat", type=float, default=None)
    parser.add_argument("--lambda-cls", type=float, default=None)
    parser.add_argument("--lambda-box", type=float, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--label", default=None, help="variant name for the results row, needed for sweep points")
    arguments = parser.parse_args()
    overrideValues = {"lambdaFeature": arguments.lambda_feat, "lambdaClass": arguments.lambda_cls, "lambdaBox": arguments.lambda_box, "temperature": arguments.temperature}
    overrides = {key: value for key, value in overrideValues.items() if value is not None}
    runFold(arguments.mode, arguments.fold, arguments.seed, arguments.epochs, arguments.dataset, arguments.project, arguments.imgsz,
            arguments.batch, arguments.device, arguments.workers, overrides or None, arguments.label)


if __name__ == "__main__":
    main()
