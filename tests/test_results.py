# Tests for the results log and the summary table (common/results.py, common/summariseResults.py)
import os
import subprocess
import sys
from pathlib import Path

from common.results import appendResultRow, buildResultRow, getGitCommit, logSpeedRow, readResultRows, resultsFolderVariable
from common.summariseResults import renderMarkdown, summariseRows


# A row carries the fixed columns and the metrics, and survives a round trip through the CSV file
def testAppendAndReadRow(tmp_path):
    row = buildResultRow("underwater", "yolo26n", "baseline", 0, {"map50": 0.5, "map50_95": 0.3})
    csvPath = appendResultRow(row, tmp_path)
    rows = readResultRows(csvPath)
    assert csvPath.name == "underwater.csv"
    assert len(rows) == 1 and rows[0]["variant"] == "baseline" and float(rows[0]["map50"]) == 0.5
    assert {"gpu", "commit", "ultralytics", "torch", "timestamp"} <= set(rows[0])


# A later row with a new metric widens the header without losing earlier rows
def testNewMetricColumnWidensFile(tmp_path):
    appendResultRow(buildResultRow("weather", "yolo26n", "student", 0, {"map50": 0.4}), tmp_path)
    csvPath = appendResultRow(buildResultRow("weather", "yolo26n", "student", 1, {"map50": 0.6, "fps": 90.0}), tmp_path)
    rows = readResultRows(csvPath)
    assert len(rows) == 2 and rows[0]["fps"] == "" and float(rows[1]["fps"]) == 90.0
    assert list(rows[0])[:4] == ["task", "model", "variant", "seed"]


# An unknown task name is rejected, so rows cannot land in a stray file
def testUnknownTaskIsRejected():
    try:
        buildResultRow("occlusion", "yolo26n", "baseline", 0, {})
    except ValueError:
        return
    raise AssertionError("an unknown task should raise ValueError")


# Seeds of one variant are averaged, and the standard deviation is the sample standard deviation
def testSummaryMeanAndDeviation():
    rows = [{"task": "underwater", "model": "yolo26n", "variant": "eca-a", "seed": str(seed), "map50": str(value), "gpu": "T4"} for seed, value in enumerate([0.50, 0.52, 0.54])]
    summaries, metricColumns = summariseRows(rows)
    meanValue, deviation, count = summaries[0]["map50"]
    assert metricColumns == ["map50"] and count == 3 and summaries[0]["seeds"] == 3
    assert abs(meanValue - 0.52) < 1e-9 and abs(deviation - 0.02) < 1e-9 and summaries[0]["runs"] == 3
    markdownText = renderMarkdown(summaries, metricColumns)
    assert "## underwater" in markdownText and "0.5200 ± 0.0200" in markdownText


# Each task gets its own table with only its own metric columns, and counts are printed as whole numbers
def testSummaryHasOneTablePerTask():
    rows = [{"task": "underwater", "model": "yolo26n", "variant": "baseline", "seed": "0", "map50": "0.5", "parameters": "2390950", "gpu": "T4"},
            {"task": "camoedge", "model": "pvt_v2_b2", "variant": "E5@CAMO", "seed": "0", "sAlpha": "0.8", "gpu": "T4"}]
    markdownText = renderMarkdown(*summariseRows(rows))
    underwaterTable, camoedgeTable = markdownText.split("## underwater")[1], markdownText.split("## underwater")[0]
    assert "sAlpha" in camoedgeTable and "map50" not in camoedgeTable
    assert "map50" in underwaterTable and "sAlpha" not in underwaterTable and "2,390,950" in underwaterTable


# Logging the speed of a run twice keeps one row, so re-running a notebook cell does not add extra values to the summary; text values are not stored
def testSpeedRowIsLoggedOnce(tmp_path):
    speedMetrics = {"fps": 120.5, "latencyMs": 8.3, "gpuName": "Tesla T4"}
    firstRow = logSpeedRow("underwater", "yolo26n", "eca-a", 0, speedMetrics, tmp_path)
    secondRow = logSpeedRow("underwater", "yolo26n", "eca-a", 0, speedMetrics, tmp_path)
    logSpeedRow("underwater", "yolo26n", "eca-a", 1, speedMetrics, tmp_path)
    rows = readResultRows(tmp_path / "underwater.csv")
    assert len(rows) == 2 and float(secondRow["fps"]) == float(firstRow["fps"]) == 120.5 and "gpuName" not in rows[0]
    assert rows[0]["notes"] == "speed protocol"


# The results folder follows the environment variable, so a trial run (scripts/trainAllPlans.py --results-folder) keeps its rows apart
def testResultsFolderFollowsEnvironmentVariable(tmp_path):
    repositoryRoot = Path(__file__).resolve().parent.parent
    command = [sys.executable, "-c", "from common.results import resultsFolder; print(resultsFolder)"]
    movedFolder = subprocess.run(command, cwd=repositoryRoot, capture_output=True, text=True, env={**os.environ, resultsFolderVariable: str(tmp_path)}).stdout.strip()
    defaultEnvironment = {name: value for name, value in os.environ.items() if name != resultsFolderVariable}
    defaultFolder = subprocess.run(command, cwd=repositoryRoot, capture_output=True, text=True, env=defaultEnvironment).stdout.strip()
    assert Path(movedFolder) == tmp_path.resolve() and Path(defaultFolder) == repositoryRoot / "results"


# New rows and plots in results/ and final weights in runs/ do not make a result "dirty"; an uncommitted change to the code does
def testGitCommitIgnoresGeneratedFolders(tmp_path):
    def runGit(*arguments): subprocess.run(["git", *arguments], cwd=tmp_path, capture_output=True, check=True)
    runGit("init")
    runGit("-c", "user.name=test", "-c", "user.email=test@example.com", "commit", "--allow-empty", "-m", "start")
    (tmp_path / "results").mkdir()
    (tmp_path / "results" / "underwater.csv").write_text("row\n", encoding="utf-8")
    (tmp_path / "runs" / "weights").mkdir(parents=True)
    (tmp_path / "runs" / "weights" / "underwater-baseline-seed0.pt").write_bytes(b"weights")
    assert getGitCommit(tmp_path) != "unknown" and not getGitCommit(tmp_path).endswith("-dirty")
    (tmp_path / "train.py").write_text("changed\n", encoding="utf-8")
    assert getGitCommit(tmp_path).endswith("-dirty")
