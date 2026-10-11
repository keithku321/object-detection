# Tests for the results log and the summary table (common/results.py, common/summariseResults.py)
from common.results import appendResultRow, buildResultRow, readResultRows
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
