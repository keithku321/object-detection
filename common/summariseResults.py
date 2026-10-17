# Merge results/*.csv into one summary table with mean ± std over seeds for every (task, model, variant)
# Run from the repository root:  python -m common.summariseResults
# Steward: plan B owner (plans/05-team-workflow.md)
import argparse
import statistics
from pathlib import Path

from common.results import leadingColumns, readResultRows, resultsFolder, trailingColumns

# Columns that identify a group of runs; rows with the same values are different seeds of one experiment
groupColumns = ("task", "model", "variant")
# File that the summary is written to unless another path is given
defaultSummaryPath = resultsFolder / "summary.md"


# Convert a CSV cell to a float, or None when the cell is empty or not a number
def toNumber(cellText):
    try:
        return float(cellText)
    except (TypeError, ValueError):
        return None


# Read every results/<task>.csv file (except the summary itself) into one list of rows
def readAllResultRows(folder=resultsFolder):
    allRows = []
    for csvPath in sorted(Path(folder).glob("*.csv")):
        allRows.extend(readResultRows(csvPath))
    return allRows


# Names of the numeric metric columns found in the rows, in order of first appearance
def findMetricColumns(rows):
    metricColumns = []
    for row in rows:
        for columnName, cellText in row.items():
            if columnName in leadingColumns or columnName in trailingColumns or columnName in metricColumns: continue
            if toNumber(cellText) is not None: metricColumns.append(columnName)
    return metricColumns


# Mean, standard deviation and count of a list of numbers; the standard deviation is 0 for a single value
def describeValues(values):
    meanValue = statistics.mean(values)
    deviation = statistics.stdev(values) if len(values) > 1 else 0.0
    return meanValue, deviation, len(values)


# Group rows by (task, model, variant) and compute mean ± std for each metric; returns a list of summary dicts
def summariseRows(rows):
    metricColumns = findMetricColumns(rows)
    groups = {}
    for row in rows:
        groups.setdefault(tuple(row[column] for column in groupColumns), []).append(row)
    summaries = []
    for groupKey, groupRows in groups.items():
        summary = dict(zip(groupColumns, groupKey))
        summary["seeds"] = len({row["seed"] for row in groupRows})
        summary["runs"] = len(groupRows)
        summary["gpus"] = ", ".join(sorted({row["gpu"] for row in groupRows if row.get("gpu")}))
        for metricName in metricColumns:
            values = [toNumber(row.get(metricName)) for row in groupRows]
            values = [value for value in values if value is not None]
            if values: summary[metricName] = describeValues(values)
        summaries.append(summary)
    return summaries, metricColumns


# Format one metric cell as "mean ± std"; a group with a single run shows only the mean
# Counts that are the same in every run (parameters, images) are printed as whole numbers
def formatMetric(metricStatistics, decimals=4):
    if metricStatistics is None: return ""
    meanValue, deviation, count = metricStatistics
    if deviation == 0 and abs(meanValue) >= 1 and float(meanValue).is_integer(): return f"{int(meanValue):,}"
    return f"{meanValue:.{decimals}f}" if count == 1 else f"{meanValue:.{decimals}f} ± {deviation:.{decimals}f}"


# Render the summaries as GitHub-flavoured markdown, one table per task with only the metric columns that task uses
def renderMarkdown(summaries, metricColumns):
    sections = []
    for taskName in sorted({summary["task"] for summary in summaries}):
        taskSummaries = [summary for summary in summaries if summary["task"] == taskName]
        taskMetrics = [metricName for metricName in metricColumns if any(metricName in summary for summary in taskSummaries)]
        headerColumns = ["model", "variant", "seeds", "runs"] + taskMetrics + ["gpus"]
        lines = [f"## {taskName}", "", "| " + " | ".join(headerColumns) + " |", "|" + "---|" * len(headerColumns)]
        for summary in taskSummaries:
            cells = [summary["model"], summary["variant"], str(summary["seeds"]), str(summary["runs"])]
            cells += [formatMetric(summary.get(metricName)) for metricName in taskMetrics]
            cells.append(summary["gpus"])
            lines.append("| " + " | ".join(cells) + " |")
        sections.append("\n".join(lines))
    return "\n\n".join(sections) + "\n"


# Build the summary table from all results files and write it to a markdown file
def writeSummary(folder=resultsFolder, outputPath=defaultSummaryPath):
    rows = [row for row in readAllResultRows(folder)]
    if not rows: return None
    summaries, metricColumns = summariseRows(rows)
    summaries.sort(key=lambda summary: tuple(summary[column] for column in groupColumns))
    markdownText = renderMarkdown(summaries, metricColumns)
    Path(outputPath).write_text(markdownText, encoding="utf-8")
    return markdownText


# Command line entry point
def main():
    parser = argparse.ArgumentParser(description="Merge results/*.csv into one summary table")
    parser.add_argument("--folder", default=str(resultsFolder), help="folder that holds the results CSV files")
    parser.add_argument("--out", default=str(defaultSummaryPath), help="markdown file to write")
    arguments = parser.parse_args()
    markdownText = writeSummary(Path(arguments.folder), Path(arguments.out))
    print(markdownText if markdownText else "No results rows found yet.")


if __name__ == "__main__":
    main()
