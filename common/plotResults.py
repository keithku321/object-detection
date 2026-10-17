### Result plots: one detailed Matplotlib figure per finished run, saved as a PNG in results/plots/<task>/ (tracked by git, so it goes into a PR).
### Flow: a training script finishes, then evaluates; the evaluation numbers (and per-class scores) are handed to plotDetectionRun,
### which also reads the per-epoch results.csv of the run. Plans A, B and C go through common/training.py (evaluateAndLog or savePlotOfRun),
### Plan B adds a by-weather plot, and Plan D goes through camoedge/evaluate.py (plotCheckpoint).
### The file name carries the task, the run, the kind of plot and the local date and time, for example
###   underwater_baseline-seed0_training-results_2026-10-17_23-40-12.png
### The training-results plot of a detection run is followed by Ultralytics' normalised confusion matrix, copied next to it under a matching name.
### Figures are built with matplotlib.figure.Figure, not pyplot, so no backend is switched and a notebook keeps its own inline plots.
###   python -m common.plotResults --task underwater --run runs/underwater/baseline-seed0       (redraw the plot of a finished detection run)
###   python -m common.plotResults --task camoedge --checkpoint runs/camoedge/E1-seed0/last.pt  (redraw the plot of a CamoEdge run)
### Steward: plan B owner (same area as common/results.py). Changes need review like any shared file.
import argparse
import csv
import shutil
import textwrap
from datetime import datetime
from pathlib import Path

from common.results import resultsFolder

# Folder for the final plots, one sub-folder per task; it is inside results/ and is not ignored by git, so plots can be committed
defaultPlotsFolder = resultsFolder / "plots"
# Name of the normalised confusion matrix that Ultralytics saves in a run folder
confusionMatrixFileName = "confusion_matrix_normalized.png"
# Resolution of saved plots in dots per inch
plotDpi = 150
# Size of the figure in inches (width, height)
figureSize = (16, 11)
# Font size of titles, axis labels and small texts
titleFontSize, labelFontSize, smallFontSize = 11, 9, 8
# Colour-blind-safe colours (Okabe and Ito); train curves, validation curves and bar series take them in this order
palette = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#7F7F7F"]
# A loss that spans more than this ratio between its largest and smallest value is drawn on a logarithmic axis
logScaleRatio = 50
# Longest line, in characters, of the text panel
summaryLineWidth = 44
# Result row columns that are not metrics and so are left out of the summary text
nonMetricColumns = {"task", "model", "variant", "seed", "gpu", "commit", "ultralytics", "torch", "timestamp", "notes"}


# Build the file name of a plot: <task>_<run>_<plot kind>_<date>_<time>.png, safe on Windows (no colons or spaces)
def makePlotFileName(task, runName, plotKind, moment=None):
    moment = moment or datetime.now()
    rawName = f"{task}_{runName}_{plotKind}_{moment.strftime('%Y-%m-%d_%H-%M-%S')}"
    return "".join(character if character.isalnum() or character in "-_.@+" else "-" for character in rawName) + ".png"


# Folder where the plots of a task are saved (created when needed): outputFolder when given, otherwise results/plots/<task>/
def getPlotsFolder(task, outputFolder=None):
    folder = Path(outputFolder) if outputFolder else defaultPlotsFolder / task
    folder.mkdir(parents=True, exist_ok=True)
    return folder


# Read the per-epoch table that Ultralytics writes to <run folder>/results.csv as {column: [values]}; column names lose their padding spaces
def readEpochHistory(runFolder):
    csvPath = Path(runFolder) / "results.csv"
    if not csvPath.exists(): return {}
    with csvPath.open(newline="", encoding="utf-8") as csvFile:
        rows = [{name.strip(): value for name, value in row.items()} for row in csv.DictReader(csvFile)]
    history = {}
    for row in rows:
        for name, value in row.items():
            try: history.setdefault(name, []).append(float(value))
            except (TypeError, ValueError): pass
    return history


# Numeric metric columns of a results row (a dict from common/results.py, values may still be text) as {name: float}
def extractRowMetrics(row):
    metrics = {}
    for name, value in row.items():
        if name in nonMetricColumns or value in ("", None): continue
        try: metrics[name] = float(value)
        except (TypeError, ValueError): pass
    return metrics


# Per-class scores from an Ultralytics validation result as {class name: {"mAP50": value, "mAP50-95": value, "instances": count}}, or None
def extractPerClassScores(validationMetrics):
    try:
        boxMetrics, names = validationMetrics.box, validationMetrics.names
        counts = getattr(validationMetrics, "nt_per_class", None)
        scores = {}
        for classIndex, ap50, ap in zip(boxMetrics.ap_class_index, boxMetrics.ap50, boxMetrics.ap):
            scores[names[int(classIndex)]] = {"mAP50": float(ap50), "mAP50-95": float(ap), "instances": int(counts[int(classIndex)]) if counts is not None else None}
        return scores or None
    except (AttributeError, TypeError, KeyError, IndexError):
        return None


# Text for a number in the summary: thousands separators for large whole numbers, four decimals otherwise
def formatValue(value):
    if isinstance(value, (int, float)) and float(value).is_integer() and abs(value) >= 1000: return f"{int(value):,}"
    if isinstance(value, float): return f"{value:.4g}" if abs(value) >= 100 else f"{value:.4f}"
    return str(value)


# Axis styling shared by every panel: bold title, axis labels, light grid and whole-number epoch ticks when the x axis is the epoch
def styleAxis(axis, title, yLabel, xLabel="Epoch", epochAxis=True):
    axis.set_title(title, fontsize=titleFontSize, fontweight="bold")
    axis.set_xlabel(xLabel, fontsize=labelFontSize)
    axis.set_ylabel(yLabel, fontsize=labelFontSize)
    axis.grid(True, alpha=0.3)
    axis.tick_params(labelsize=smallFontSize)
    if epochAxis:
        from matplotlib.ticker import MaxNLocator
        axis.xaxis.set_major_locator(MaxNLocator(integer=True))


# Text shown in a panel that has nothing to draw, so an empty panel explains itself
def drawEmptyPanel(axis, title, message):
    axis.set_title(title, fontsize=titleFontSize, fontweight="bold")
    axis.text(0.5, 0.5, message, ha="center", va="center", fontsize=labelFontSize, color="#555555", transform=axis.transAxes, wrap=True)
    axis.set_xticks([])
    axis.set_yticks([])


# Draw line series against the epoch; series is a list of (values, label, colour, line style). Markers are always drawn so a one-epoch run is visible.
# A series that is missing or empty is skipped; with nothing to draw the panel says so. Returns True when something was drawn.
def drawSeries(axis, epochs, series, title, yLabel, bestLabel=None, higherIsBetter=True):
    drawn = [entry for entry in series if entry[0]]
    if not drawn or not epochs:
        drawEmptyPanel(axis, title, "no data for this plot")
        return False
    allValues = [value for values, _, _, _ in drawn for value in values]
    for values, label, colour, lineStyle in drawn:
        # Dashed (validation) series use hollow square markers so they stay apart from the solid (training) series when both sit on one point
        isDashed = lineStyle == "--"
        axis.plot(epochs[:len(values)], values, label=label, color=colour, linestyle=lineStyle, linewidth=1.8, marker="s" if isDashed else "o",
                  markersize=7 if isDashed else 5, markerfacecolor="white" if isDashed else colour, markeredgewidth=1.6)
        if bestLabel == label:
            bestIndex = max(range(len(values)), key=lambda index: values[index] if higherIsBetter else -values[index])
            axis.scatter([epochs[bestIndex]], [values[bestIndex]], marker="*", s=170, color=colour, edgecolor="black", zorder=5)
            axis.annotate(f"best {values[bestIndex]:.3f} (epoch {int(epochs[bestIndex])})", (epochs[bestIndex], values[bestIndex]), textcoords="offset points",
                          xytext=(0, 10), ha="center", fontsize=smallFontSize, fontweight="bold")
    useLogScale = min(allValues) > 0 and max(allValues) / min(allValues) > logScaleRatio
    if useLogScale: axis.set_yscale("log")
    styleAxis(axis, title, yLabel + (" (log scale)" if useLogScale else ""))
    if len(epochs) == 1: axis.set_xlim(epochs[0] - 0.5, epochs[0] + 0.5)
    if len(epochs) <= 10: axis.set_xticks([int(epoch) for epoch in epochs])
    axis.margins(y=0.15)
    axis.legend(fontsize=smallFontSize, framealpha=0.9)
    return True


# Draw grouped bars: groups is {series label: [one value per category]}; every bar carries its value. upperLimit fixes the top of the axis (1.0 for
# scores between 0 and 1) and None lets Matplotlib choose.
def drawBars(axis, categories, groups, title, yLabel, upperLimit=1.0, xLabel=""):
    if not categories or not groups:
        drawEmptyPanel(axis, title, "no data for this plot")
        return False
    width = 0.8 / len(groups)
    for seriesIndex, (label, values) in enumerate(groups.items()):
        positions = [categoryIndex - 0.4 + width * (seriesIndex + 0.5) for categoryIndex in range(len(categories))]
        bars = axis.bar(positions, values, width, label=label, color=palette[seriesIndex % len(palette)], edgecolor="white")
        axis.bar_label(bars, labels=[f"{value:.3f}" if len(categories) * len(groups) <= 24 else f"{value:.2f}" for value in values], fontsize=smallFontSize - 1, padding=2)
    axis.set_xticks(range(len(categories)))
    crowded = len(categories) > 8 or max(len(str(category).split("\n")[0]) for category in categories) > 10
    axis.set_xticklabels(categories, rotation=30 if crowded else 0, ha="right" if crowded else "center")
    styleAxis(axis, title, yLabel, xLabel, epochAxis=False)
    axis.grid(True, axis="y", alpha=0.3)
    axis.grid(False, axis="x")
    if upperLimit: axis.set_ylim(0, upperLimit * 1.12)
    else: axis.margins(y=0.15)
    if len(groups) > 1: axis.legend(fontsize=smallFontSize, framealpha=0.9)
    return True


# Fill a panel with plain monospaced text (a run summary); long lines are wrapped
def drawSummary(axis, title, lines):
    wrapped = []
    for line in lines: wrapped.extend(textwrap.wrap(line, summaryLineWidth, subsequent_indent="    ") or [""])
    axis.set_title(title, fontsize=titleFontSize, fontweight="bold")
    axis.text(0.0, 1.0, "\n".join(wrapped), ha="left", va="top", fontsize=smallFontSize + 0.5, family="monospace", transform=axis.transAxes)
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values(): spine.set_color("#BBBBBB")


# Save a finished figure into the plots folder of its task and print where it went; returns the file path
def saveFigure(figure, task, runName, plotKind, outputFolder=None, moment=None):
    moment = moment or datetime.now()
    figure.text(0.995, 0.003, f"created {moment.strftime('%Y-%m-%d %H:%M:%S')}", ha="right", va="bottom", fontsize=smallFontSize - 1, color="#777777")
    plotPath = getPlotsFolder(task, outputFolder) / makePlotFileName(task, runName, plotKind, moment)
    figure.savefig(plotPath, dpi=plotDpi)
    print(f"Saved plot: {plotPath}")
    return plotPath


# Copy the normalised confusion matrix of a run folder next to a saved plot, named like the plot; returns the new path, or None when the run has none
def copyConfusionMatrix(runFolder, plotPath):
    sourcePath = Path(runFolder) / confusionMatrixFileName
    if not sourcePath.exists(): return None
    targetPath = Path(plotPath).with_name(Path(plotPath).name.replace("training-results", "confusion-matrix-normalised"))
    shutil.copy2(sourcePath, targetPath)
    print(f"Saved plot: {targetPath}")
    return targetPath


# Training-results plot of an Ultralytics detection run, read from <run folder>/results.csv. Returns the PNG path, or None when there is no results.csv.
# finalMetrics: numeric metrics of the final evaluation (a results row), perClass: output of extractPerClassScores, notes: free text for the summary.
def plotDetectionRun(runFolder, task, runName, finalMetrics=None, perClass=None, notes="", outputFolder=None, moment=None):
    from matplotlib.figure import Figure
    moment = moment or datetime.now()
    history = readEpochHistory(runFolder)
    if not history.get("epoch"):
        print(f"No results.csv with epochs in {runFolder}; plot skipped")
        return None
    epochs = history["epoch"]
    figure = Figure(figsize=figureSize, layout="constrained")
    grid = figure.add_gridspec(3, 3)
    axes = [figure.add_subplot(grid[row, column]) for row in range(2) for column in range(3)]
    colourOf = lambda index: palette[index % len(palette)]
    # Row 1: losses, training as solid lines and validation as dashed lines of the same colour
    drawSeries(axes[0], epochs, [(history.get("train/box_loss"), "train", colourOf(0), "-"), (history.get("val/box_loss"), "validation", colourOf(0), "--")],
               "Box loss", "loss")
    drawSeries(axes[1], epochs, [(history.get("train/cls_loss"), "train", colourOf(1), "-"), (history.get("val/cls_loss"), "validation", colourOf(1), "--")],
               "Class loss", "loss")
    otherLossNames = sorted({name.split("/", 1)[1] for name in history if name.startswith(("train/", "val/")) and name.endswith("loss")} - {"box_loss", "cls_loss"})
    otherSeries = []
    for index, lossName in enumerate(otherLossNames):
        otherSeries.append((history.get(f"train/{lossName}"), f"train {lossName}", colourOf(index + 2), "-"))
        otherSeries.append((history.get(f"val/{lossName}"), f"validation {lossName}", colourOf(index + 2), "--"))
    drawSeries(axes[2], epochs, otherSeries, "Other losses (distillation, L1, DFL)", "loss")
    # Row 2: validation scores after every epoch, and the learning rate
    drawSeries(axes[3], epochs, [(history.get("metrics/mAP50(B)"), "mAP50", colourOf(0), "-"), (history.get("metrics/mAP50-95(B)"), "mAP50-95", colourOf(1), "-")],
               "Validation mAP (higher is better)", "mAP", bestLabel="mAP50-95")
    drawSeries(axes[4], epochs, [(history.get("metrics/precision(B)"), "precision", colourOf(2), "-"), (history.get("metrics/recall(B)"), "recall", colourOf(3), "-")],
               "Validation precision and recall", "score")
    learningRateGroups = [history.get(f"lr/pg{index}") for index in range(3)]
    if all(group == learningRateGroups[0] for group in learningRateGroups): learningRateSeries = [(learningRateGroups[0], "all parameter groups", colourOf(0), "-")]
    else: learningRateSeries = [(group, f"parameter group {index}", colourOf(index), "-") for index, group in enumerate(learningRateGroups)]
    drawSeries(axes[5], epochs, learningRateSeries, "Learning rate", "learning rate")
    # Row 3: per-class scores (or accuracy by object size), and the text summary
    barAxis, summaryAxis = figure.add_subplot(grid[2, :2]), figure.add_subplot(grid[2, 2])
    if perClass:
        categories = [f"{name}\n(n={scores['instances']})" if scores.get("instances") is not None else name for name, scores in perClass.items()]
        drawBars(barAxis, categories, {"mAP50": [scores["mAP50"] for scores in perClass.values()], "mAP50-95": [scores["mAP50-95"] for scores in perClass.values()]},
                 "Final evaluation per class (n = instances)", "mAP")
    elif finalMetrics and all(key in finalMetrics for key in ("apSmall", "apMedium", "apLarge")):
        drawBars(barAxis, ["small", "medium", "large"], {"COCO AP": [finalMetrics["apSmall"], finalMetrics["apMedium"], finalMetrics["apLarge"]]},
                 "Final evaluation: COCO AP by object size", "AP")
    else:
        lastScores = {label: history[column][-1] for label, column in (("mAP50", "metrics/mAP50(B)"), ("mAP50-95", "metrics/mAP50-95(B)"),
                      ("precision", "metrics/precision(B)"), ("recall", "metrics/recall(B)")) if history.get(column)}
        drawBars(barAxis, list(lastScores), {"last epoch": list(lastScores.values())}, "Scores after the last epoch", "score")
    summaryLines = [f"task        {task}", f"run         {runName}", f"epochs      {len(epochs)}"]
    if history.get("time"): summaryLines.append(f"train time  {history['time'][-1] / 3600:.2f} h")
    if history.get("metrics/mAP50-95(B)"):
        bestIndex = max(range(len(epochs)), key=lambda index: history["metrics/mAP50-95(B)"][index])
        summaryLines.append(f"best epoch  {int(epochs[bestIndex])} (mAP50-95 {history['metrics/mAP50-95(B)'][bestIndex]:.4f})")
    if finalMetrics:
        summaryLines += ["", "final evaluation"] + [f"  {name:<12}{formatValue(value)}" for name, value in finalMetrics.items()]
    if notes: summaryLines += ["", f"notes: {notes}"]
    drawSummary(summaryAxis, "Run summary", summaryLines)
    figure.suptitle(f"{task} | {runName} | training results ({len(epochs)} epoch{'s' if len(epochs) != 1 else ''})", fontsize=14, fontweight="bold")
    plotPath = saveFigure(figure, task, runName, "training-results", outputFolder, moment)
    copyConfusionMatrix(runFolder, plotPath)
    return plotPath


# Bar plot of one model scored on several groups, for example the weather types of Plan B; metricsByGroup is {group: {metric: value, "images": count}}
def plotGroupedMetrics(task, runName, plotKind, metricsByGroup, metricNames=("map50", "map50_95", "precision", "recall"), groupTitle="group",
                       notes="", outputFolder=None, moment=None):
    from matplotlib.figure import Figure
    if not metricsByGroup:
        print(f"No {groupTitle} results to plot; plot skipped")
        return None
    moment = moment or datetime.now()
    figure = Figure(figsize=(12, 7), layout="constrained")
    axis = figure.add_subplot(1, 1, 1)
    categories = [f"{group}\n(n={int(metrics['images'])})" if "images" in metrics else group for group, metrics in metricsByGroup.items()]
    groups = {name: [metrics.get(name, 0.0) for metrics in metricsByGroup.values()] for name in metricNames}
    drawBars(axis, categories, groups, f"Final evaluation by {groupTitle} (n = images)", "score", xLabel=groupTitle)
    figure.suptitle(f"{task} | {runName} | by {groupTitle}" + (f" | {notes}" if notes else ""), fontsize=13, fontweight="bold")
    return saveFigure(figure, task, runName, plotKind, outputFolder, moment)


# Training plot of a CamoEdge run from the per-epoch history stored in its checkpoint. history: list of {"epoch", "loss", "maskLoss", "edgeLoss", "seconds"},
# settings: the checkpoint settings, learningRates: one value per epoch. metricsByTestSet ({test set: metrics}) and perCategory ({super-class: metrics})
# come from camoedge/evaluate.py and are optional. Returns the PNG path.
def plotCamoEdgeRun(runFolder, history, settings, learningRates=None, metricsByTestSet=None, perCategory=None, notes="", outputFolder=None, moment=None):
    from matplotlib.figure import Figure
    moment = moment or datetime.now()
    if not history:
        print(f"No finished epoch in {runFolder}; plot skipped")
        return None
    epochs = [record["epoch"] for record in history]
    column = lambda name: [record[name] for record in history if name in record]
    figure = Figure(figsize=figureSize, layout="constrained")
    grid = figure.add_gridspec(3, 3)
    axes = [figure.add_subplot(grid[row, columnIndex]) for row in range(2) for columnIndex in range(3)]
    drawSeries(axes[0], epochs, [(column("loss"), "total loss", palette[0], "-")], "Total training loss", "loss")
    drawSeries(axes[1], epochs, [(column("maskLoss"), "mask loss", palette[1], "-"), (column("edgeLoss"), "edge loss", palette[2], "-")], "Mask and edge loss", "loss")
    drawSeries(axes[2], epochs, [(column("seconds"), "seconds per epoch", palette[3], "-")], "Time per epoch", "seconds")
    drawSeries(axes[3], epochs, [(learningRates, "learning rate", palette[4], "-")], "Learning rate (polynomial decay)", "learning rate")
    scoreNames = [("sAlpha", "S-measure"), ("eMean", "mean E-measure"), ("fWeighted", "weighted F"), ("boundaryF", "boundary F")]
    testSets = list(metricsByTestSet or {})
    if testSets:
        drawBars(axes[4], [f"{name}\n(n={int(metricsByTestSet[name].get('images', 0))})" for name in testSets],
                 {label: [metricsByTestSet[name][key] for name in testSets] for key, label in scoreNames}, "Test scores (higher is better; n = images)", "score")
        drawBars(axes[5], testSets, {"MAE": [metricsByTestSet[name]["mae"] for name in testSets]}, "Mean absolute error (lower is better)", "MAE", upperLimit=None)
    else:
        drawEmptyPanel(axes[4], "Test scores", "run camoedge.evaluate to fill this panel")
        drawEmptyPanel(axes[5], "Mean absolute error", "run camoedge.evaluate to fill this panel")
    barAxis, summaryAxis = figure.add_subplot(grid[2, :2]), figure.add_subplot(grid[2, 2])
    if perCategory:
        drawBars(barAxis, list(perCategory), {label: [perCategory[name][key] for name in perCategory] for key, label in scoreNames[:1] + scoreNames[3:]},
                 "COD10K per super-class", "score")
    else:
        drawEmptyPanel(barAxis, "COD10K per super-class", "available after evaluation on COD10K")
    runName = settings.get("variantName", "run") + f"-seed{settings.get('seed', 0)}"
    summaryLines = ["task        camoedge", f"run        {runName}", f"encoder     {settings.get('encoderName', '?')}",
                    f"epochs      {len(history)} of {settings.get('epochs', '?')}", f"batch size  {settings.get('batchSize', '?')}", f"image size  {settings.get('imageSize', '?')}",
                    f"start lr    {settings.get('learningRate', '?')}", f"train time  {sum(column('seconds')) / 3600:.2f} h", f"final loss  {history[-1]['loss']:.4f}"]
    if notes: summaryLines += ["", f"notes: {notes}"]
    drawSummary(summaryAxis, "Run summary", summaryLines)
    figure.suptitle(f"camoedge | {runName} | training results ({len(history)} epoch{'s' if len(history) != 1 else ''})", fontsize=14, fontweight="bold")
    return saveFigure(figure, "camoedge", runName, "training-results", outputFolder, moment)


# Call a plotting function and never let a plotting problem stop a run that took hours: the problem is printed and None is returned
def trySavePlot(plotFunction, *arguments, **keywordArguments):
    try:
        return plotFunction(*arguments, **keywordArguments)
    except Exception as error:
        print(f"WARNING: could not save the plot ({type(error).__name__}: {error}); training results are not affected")
        return None


# Command line entry point: redraw the plot of a finished run
def main():
    parser = argparse.ArgumentParser(description="Redraw the results plot of a finished run")
    parser.add_argument("--task", required=True, choices=["underwater", "weather", "smallobject", "camoedge"])
    parser.add_argument("--run", help="run folder that contains results.csv (Ultralytics tasks)")
    parser.add_argument("--checkpoint", help="last.pt of a CamoEdge run")
    parser.add_argument("--output", help="folder for the PNG; default is results/plots/<task>/")
    arguments = parser.parse_args()
    if arguments.task == "camoedge":
        if not arguments.checkpoint: parser.error("--checkpoint is required for camoedge")
        from camoedge.evaluate import plotCheckpoint
        plotCheckpoint(arguments.checkpoint, outputFolder=arguments.output)
    else:
        if not arguments.run: parser.error("--run is required for Ultralytics tasks")
        plotDetectionRun(arguments.run, arguments.task, Path(arguments.run).resolve().name, outputFolder=arguments.output)


if __name__ == "__main__":
    main()
