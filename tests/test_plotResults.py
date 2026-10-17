# Tests for the result plots (common/plotResults.py); they need Matplotlib only and use made-up run folders
import re
from datetime import datetime
from types import SimpleNamespace

import pytest

pytest.importorskip("matplotlib")

import common.plotResults as plotResults  # noqa: E402
from common.plotResults import (extractPerClassScores, extractRowMetrics, makePlotFileName, plotCamoEdgeRun, plotDetectionRun,  # noqa: E402
                                plotGroupedMetrics, readEpochHistory, trySavePlot)

# Every test saves its plots into a temporary folder instead of the real results/plots folder
@pytest.fixture(autouse=True)
def redirectPlots(tmp_path, monkeypatch):
    monkeypatch.setattr(plotResults, "defaultPlotsFolder", tmp_path / "saved-plots")


# Header of the results.csv that Ultralytics 8.4.175 writes for YOLO26 (Ultralytics pads some names with spaces)
ultralyticsHeader = ("epoch,time,train/box_loss,train/cls_loss,train/l1_loss,metrics/precision(B),metrics/recall(B),metrics/mAP50(B),metrics/mAP50-95(B),"
                     "val/box_loss,val/cls_loss,val/l1_loss,lr/pg0,lr/pg1,lr/pg2")


# Write a results.csv with the given number of epochs and return the run folder
def makeRunFolder(tmp_path, epochCount):
    runFolder = tmp_path / "baseline-seed0"
    runFolder.mkdir()
    lines = [ultralyticsHeader]
    for epoch in range(1, epochCount + 1):
        lines.append(f"{epoch},{epoch * 60.0},{2.0 / epoch},{30.0 / epoch ** 2},0.01,{0.3 + 0.05 * epoch},{0.3 + 0.04 * epoch},{0.4 + 0.05 * epoch},{0.2 + 0.03 * epoch},"
                     f"{1.9 / epoch},{20.0 / epoch ** 2},0.01,0.001,0.001,0.001")
    (runFolder / "results.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return runFolder


# The file name carries task, run, plot kind and date and time, and has no characters that Windows rejects
def testPlotFileName():
    name = makePlotFileName("underwater", "baseline-seed0", "training-results", datetime(2026, 10, 17, 23, 40, 12))
    assert name == "underwater_baseline-seed0_training-results_2026-10-17_23-40-12.png"
    assert not re.search(r'[<>:"/\\|?* ]', makePlotFileName("weather", "distill full:fold0", "by-weather"))


# Column names are stripped and values become floats
def testReadEpochHistory(tmp_path):
    runFolder = makeRunFolder(tmp_path, 3)
    history = readEpochHistory(runFolder)
    assert history["epoch"] == [1.0, 2.0, 3.0] and len(history["metrics/mAP50-95(B)"]) == 3
    assert readEpochHistory(tmp_path / "missing") == {}


# A detection run is drawn for one epoch and for many, with and without final metrics; the PNG lands in <plots folder>/<task>/ and the run's
# normalised confusion matrix is copied next to it under a matching name
@pytest.mark.parametrize("epochCount", [1, 5])
def testPlotDetectionRun(tmp_path, epochCount):
    runFolder = makeRunFolder(tmp_path, epochCount)
    (runFolder / "confusion_matrix_normalized.png").write_bytes(b"png")
    perClass = {"crab": {"mAP50": 0.1, "mAP50-95": 0.05, "instances": 1226}, "fish": {"mAP50": 0.6, "mAP50-95": 0.3, "instances": 325}}
    plotPath = plotDetectionRun(runFolder, "underwater", runFolder.name, {"map50": 0.47, "map50_95": 0.24, "parameters": 2390950.0}, perClass, "split=val")
    assert plotPath.parent == tmp_path / "saved-plots" / "underwater" and plotPath.stat().st_size > 10_000
    assert not (runFolder / "plots").exists()
    assert (plotPath.parent / plotPath.name.replace("training-results", "confusion-matrix-normalised")).read_bytes() == b"png"
    assert plotDetectionRun(runFolder, "underwater", runFolder.name, {"apSmall": 0.1, "apMedium": 0.3, "apLarge": 0.5}) is not None
    assert plotDetectionRun(runFolder, "underwater", runFolder.name) is not None


# The default plots folder is results/plots in the repository, which is tracked by git
def testDefaultPlotsFolderIsInsideResults(monkeypatch):
    monkeypatch.undo()
    assert plotResults.defaultPlotsFolder.parts[-2:] == ("results", "plots")


# A folder without results.csv gives no plot instead of an error
def testNoResultsFileGivesNone(tmp_path):
    assert plotDetectionRun(tmp_path, "underwater", "none") is None


# The by-group plot (weather types) is drawn, and an empty result gives None
def testPlotGroupedMetrics(tmp_path):
    metrics = {"fog": {"map50": 0.4, "map50_95": 0.2, "precision": 0.5, "recall": 0.4, "images": 200}, "snow": {"map50": 0.3, "map50_95": 0.1, "precision": 0.4, "recall": 0.3, "images": 150}}
    plotPath = plotGroupedMetrics("weather", "student-fold0-seed0", "by-weather", metrics, groupTitle="weather")
    assert plotPath.parent.name == "weather" and plotPath.name.startswith("weather_student-fold0-seed0_by-weather_") and plotPath.exists()
    assert plotGroupedMetrics("weather", "student", "by-weather", {}) is None


# A CamoEdge run is drawn before and after evaluation
def testPlotCamoEdgeRun(tmp_path):
    history = [{"epoch": epoch, "loss": 1.0 / epoch, "maskLoss": 0.6 / epoch, "edgeLoss": 0.4 / epoch, "seconds": 30.0} for epoch in (1, 2, 3)]
    settings = {"variantName": "E5", "seed": 0, "encoderName": "pvt_v2_b2", "epochs": 3, "batchSize": 16, "imageSize": 352, "learningRate": 1e-4}
    assert plotCamoEdgeRun(tmp_path, history, settings, [1e-4, 7e-5, 3e-5]).parent == tmp_path / "saved-plots" / "camoedge"
    testMetrics = {"CAMO": {"sAlpha": 0.7, "eMean": 0.8, "eMax": 0.85, "fWeighted": 0.6, "mae": 0.08, "boundaryF": 0.5, "images": 250}}
    categoryMetrics = {"Aquatic": {"sAlpha": 0.7, "boundaryF": 0.5}, "Flying": {"sAlpha": 0.8, "boundaryF": 0.6}}
    assert plotCamoEdgeRun(tmp_path, history, settings, [1e-4, 7e-5, 3e-5], testMetrics, categoryMetrics, "subset=100").exists()
    assert plotCamoEdgeRun(tmp_path, [], settings) is None


# Per-class scores are read from an Ultralytics-like validation result, and anything unexpected gives None
def testExtractPerClassScores():
    validation = SimpleNamespace(box=SimpleNamespace(ap_class_index=[0, 2], ap50=[0.5, 0.7], ap=[0.3, 0.4]), names={0: "crab", 1: "fish", 2: "shrimp"}, nt_per_class=[10, 5, 7])
    assert extractPerClassScores(validation) == {"crab": {"mAP50": 0.5, "mAP50-95": 0.3, "instances": 10}, "shrimp": {"mAP50": 0.7, "mAP50-95": 0.4, "instances": 7}}
    assert extractPerClassScores(object()) is None


# Result rows lose their text columns and keep numbers
def testExtractRowMetrics():
    row = {"task": "underwater", "variant": "baseline", "seed": "0", "map50": "0.47", "gpu": "cpu", "notes": "x", "fps": ""}
    assert extractRowMetrics(row) == {"map50": 0.47}


# A failing plot function is reported and returns None instead of stopping the run
def testTrySavePlotSwallowsErrors():
    def failingPlot():
        raise RuntimeError("broken")
    assert trySavePlot(failingPlot) is None
