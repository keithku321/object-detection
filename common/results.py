# Results log: every run appends one row to results/<task>.csv (task, model, variant, seed, metrics, GPU, commit, versions)
# Steward: plan B owner (plans/05-team-workflow.md). Rows are committed to git; weights are not.
import csv
import subprocess
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

# Folder that holds one results CSV per task
resultsFolder = Path(__file__).resolve().parent.parent / "results"
# Repository root, used to find the git commit
repositoryRoot = Path(__file__).resolve().parent.parent
# Task names that own a results file
validTasks = ("underwater", "weather", "smallobject", "camoedge")
# Columns that start every row, in this order
leadingColumns = ["task", "model", "variant", "seed"]
# Columns that end every row, in this order; metric columns sit between the leading and trailing columns
trailingColumns = ["gpu", "commit", "ultralytics", "torch", "timestamp", "notes"]


# Name of the GPU used for this run, or "cpu" when no GPU is usable
def getGpuName():
    try:
        import torch
        if torch.cuda.is_available(): return torch.cuda.get_device_name(0)
    except ImportError:
        pass
    return "cpu"


# Short git commit of the code that produced a result; a "-dirty" suffix means there were uncommitted changes
def getGitCommit(repositoryPath=repositoryRoot):
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=repositoryPath, capture_output=True, text=True, check=True).stdout.strip()
        status = subprocess.run(["git", "status", "--porcelain"], cwd=repositoryPath, capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return f"{commit}-dirty" if status else commit


# Installed version of a Python package, or "not installed"
def getLibraryVersion(packageName):
    try:
        return metadata.version(packageName)
    except metadata.PackageNotFoundError:
        return "not installed"


# Build one result row; metrics is a dict of metric name to number, and every name becomes a CSV column
def buildResultRow(task, model, variant, seed, metrics, notes=""):
    if task not in validTasks: raise ValueError(f"Unknown task '{task}', expected one of {validTasks}")
    row = {"task": task, "model": model, "variant": variant, "seed": seed}
    for metricName, metricValue in metrics.items():
        row[metricName] = round(float(metricValue), 6) if isinstance(metricValue, (int, float)) else metricValue
    row.update({
        "gpu": getGpuName(),
        "commit": getGitCommit(),
        "ultralytics": getLibraryVersion("ultralytics"),
        "torch": getLibraryVersion("torch"),
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "notes": notes,
    })
    return row


# Read all rows of a results file as a list of dicts; an absent file gives an empty list
def readResultRows(csvPath):
    csvPath = Path(csvPath)
    if not csvPath.exists(): return []
    with csvPath.open(newline="", encoding="utf-8") as csvFile:
        return list(csv.DictReader(csvFile))


# Order the columns of a results file: leading columns, metric columns in order of first appearance, trailing columns
def orderColumns(rows):
    metricColumns = []
    for row in rows:
        for columnName in row:
            if columnName not in leadingColumns and columnName not in trailingColumns and columnName not in metricColumns:
                metricColumns.append(columnName)
    return leadingColumns + metricColumns + trailingColumns


# Write rows to a results file through a temporary file, so an interrupted write cannot leave a half-written CSV
def writeResultRows(csvPath, rows):
    csvPath = Path(csvPath)
    csvPath.parent.mkdir(parents=True, exist_ok=True)
    temporaryPath = csvPath.with_suffix(".csv.tmp")
    with temporaryPath.open("w", newline="", encoding="utf-8") as csvFile:
        writer = csv.DictWriter(csvFile, fieldnames=orderColumns(rows), restval="", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporaryPath.replace(csvPath)
    return csvPath


# Append a row to results/<task>.csv; when the row has a new metric column the whole file is rewritten with the wider header
def appendResultRow(row, folder=resultsFolder):
    csvPath = Path(folder) / f"{row['task']}.csv"
    return writeResultRows(csvPath, readResultRows(csvPath) + [row])


# Split a notes text such as "split=val; epochs=50" into its fields ("split=val", "epochs=50"), ignoring empty ones
def splitNoteFields(notes):
    return [field.strip() for field in str(notes or "").split(";") if field.strip()]


# Find an already logged row for the same task, model, variant and seed whose notes hold every required field; returns the row or None
# notesContain is one field ("split=val;" or "split=val") or a list of them, and each must equal a whole field of the row's notes, so
# "epochs=1" does not match a row of a 10-epoch run. Used so that re-running a cell after a Colab restart does not log the same run twice,
# while a run with other settings (for example more epochs) still gets its own row
def findResultRow(task, model, variant, seed, notesContain="", folder=resultsFolder):
    requiredFields = [field for item in ([notesContain] if isinstance(notesContain, str) else notesContain) for field in splitNoteFields(item)]
    for row in readResultRows(Path(folder) / f"{task}.csv"):
        sameRun = (row["model"], row["variant"], str(row["seed"])) == (model, variant, str(seed))
        if sameRun and set(requiredFields) <= set(splitNoteFields(row.get("notes", ""))): return row
    return None


# Text in the notes of a speed row; a re-run finds the row that is already there by this text
speedNote = "speed protocol"


# Log the outcome of the speed protocol (common/benchmark.py) as one row; a run that already has a speed row is not logged twice
# Only numbers are kept, because the benchmark also returns text such as the GPU name. Returns the new row or the existing one
def logSpeedRow(task, model, variant, seed, speedMetrics, folder=resultsFolder):
    existingRow = findResultRow(task, model, variant, seed, speedNote, folder)
    if existingRow: return existingRow
    row = buildResultRow(task, model, variant, seed, {key: value for key, value in speedMetrics.items() if isinstance(value, (int, float))}, notes=speedNote)
    appendResultRow(row, folder)
    return row


# Merge two copies of a task's results file (the repository copy and a copy on Google Drive) so that neither loses rows
# A fresh Colab clone only has the committed rows, so call this at the start of a session and after every run. Both files end up equal.
def syncResultFiles(task, otherFolder, folder=resultsFolder):
    csvPaths = [Path(folder) / f"{task}.csv", Path(otherFolder) / f"{task}.csv"]
    mergedRows, seenRows = [], set()
    for csvPath in csvPaths:
        for row in readResultRows(csvPath):
            rowKey = tuple(sorted((column, value) for column, value in row.items() if value != ""))
            if rowKey in seenRows: continue
            seenRows.add(rowKey)
            mergedRows.append(row)
    if not mergedRows: return 0
    mergedRows.sort(key=lambda row: row.get("timestamp", ""))
    for csvPath in csvPaths: writeResultRows(csvPath, mergedRows)
    return len(mergedRows)


# Turn an Ultralytics validation result (model.val) into the standard detection metric dict
def extractDetectionMetrics(validationMetrics):
    boxMetrics = validationMetrics.box
    return {"map50": boxMetrics.map50, "map50_95": boxMetrics.map, "precision": boxMetrics.mp, "recall": boxMetrics.mr}


# Convenience for detection runs: extract the metrics from model.val, add optional extras and append the row
def logDetectionRun(task, model, variant, seed, validationMetrics, extraMetrics=None, notes="", folder=resultsFolder):
    metrics = extractDetectionMetrics(validationMetrics)
    if extraMetrics: metrics.update(extraMetrics)
    return appendResultRow(buildResultRow(task, model, variant, seed, metrics, notes), folder)
