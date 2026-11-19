# Train the Phase 1 baselines of every plan one after the other, with one log file per run. Used for the one-epoch check and for the
# long runs. Run it from the repository root with the interpreter of the project's .venv (the runs use the same interpreter):
#   .venv\Scripts\python.exe -m scripts.trainAllPlans --epochs 50 --project runs
#   .venv\Scripts\python.exe -m scripts.trainAllPlans --epochs 1 --project runs/epoch1-check --results-folder runs/epoch1-check/results
# Runs per plan (plans/00-main-plan.md, Phase 1): A baseline (U0); B teacher yolo26m (W1) then student yolo26n (W0) on fold 0;
# C baseline (S0) on VisDrone; D plain encoder-decoder E1. Every run is its own "python -m <task>.train" process, so a crash in one
# plan does not stop the others, and re-running the same command resumes unfinished runs and skips finished ones.
# A trial run needs its own --project and --results-folder: Plans A, B and C find an existing results row by variant, seed and fold or split
# (not by epochs), so a one-epoch row in results/ would make the real run skip its evaluation. For the same reason Plan A runs under the
# label baseline-<epochs>ep: results/underwater.csv already holds the 20-epoch "baseline" row of seed 0 (PR #2).
# Output of every run is shown live in this console and written to its log file, so a training started in a visible terminal can be watched.
import argparse
import ctypes
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from common.results import resultsFolderVariable

# Repository root: every command runs from here, like the commands in cmd.md
repositoryRoot = Path(__file__).resolve().parent.parent
# Prepared dataset of Plan A (dataset.yaml is git-ignored and written for this machine; see cmd.md)
brackishYaml = "underwater/dataset/brackish/dataset.yaml"
# Prepared DAWN folder of Plan B (weather/prepareDawn.py)
dawnFolder = "weather/dataset/dawn"
# Arranged COD10K + CAMO folder of Plan D, with TrainDataset/ and TestDataset/
camoFolder = "camoedge/dataset/camo"
# Plans in the order they are trained when --plans is not given
defaultPlans = ["A", "B", "C", "D"]
# Windows SetThreadExecutionState flags (ES_CONTINUOUS | ES_SYSTEM_REQUIRED) that stop the PC from sleeping while this script runs
keepAwakeFlags = 0x80000000 | 0x00000001
# Time stamp format of the log file names, the same as the results plots
timeStampFormat = "%Y-%m-%d_%H-%M-%S"


# Command lines of one plan's runs: list of (run label, arguments after the Python interpreter)
def buildPlanCommands(planLetter, epochs, seed, project, workers):
    sharedArguments = ["--seed", str(seed), "--epochs", str(epochs), "--project", project]
    workerArguments = ["--workers", str(workers)] if workers is not None else []
    if planLetter == "A":
        return [(f"underwater-baseline-{epochs}ep", ["-m", "underwater.train", "--variant", "baseline", "--label", f"baseline-{epochs}ep", "--data", brackishYaml,
                                                     *sharedArguments, *workerArguments])]
    if planLetter == "B":
        return [(f"weather-{mode}-fold0", ["-m", "weather.train", "--mode", mode, "--fold", "0", "--dataset", dawnFolder, *sharedArguments, *workerArguments])
                for mode in ("teacher", "student")]
    if planLetter == "C":
        return [("smallobject-baseline", ["-m", "smallobject.train", "--variant", "baseline", *sharedArguments, *workerArguments])]
    if planLetter == "D":
        return [("camoedge-E1", ["-m", "camoedge.train", "--experiment", "E1", "--dataset", camoFolder, *sharedArguments])]
    raise ValueError(f"Unknown plan '{planLetter}'. Known: {defaultPlans}")


# Print a time-stamped line and append it to the summary log
def writeSummaryLine(summaryPath, text):
    line = f"{datetime.now():%Y-%m-%d %H:%M:%S} {text}"
    print(line, flush=True)
    with open(summaryPath, "a", encoding="utf-8") as summaryFile: summaryFile.write(line + "\n")


# Short git commit of the repository and whether tracked files have uncommitted changes, for the summary log
def describeGitState():
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=repositoryRoot, capture_output=True, text=True).stdout.strip()
        changes = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=repositoryRoot, capture_output=True, text=True).stdout.strip()
    except OSError:
        return "git not found"
    return f"commit {commit}" + (" with uncommitted changes" if changes else "")


# Run one command with this interpreter; its stdout and stderr are shown live in this console and copied to logPath. Returns (exit code, seconds taken)
# resultsFolder, when given, replaces results/ for the results rows and plots of this command (see common/results.py)
def runLogged(arguments, logPath, resultsFolder=None):
    startTime = time.time()
    # UTF-8 output, so the log file and the console agree whatever the Windows code page is
    environment = {**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8"}
    if resultsFolder is not None: environment[resultsFolderVariable] = str(resultsFolder)
    process = subprocess.Popen([sys.executable, *arguments], cwd=repositoryRoot, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=environment)
    with open(logPath, "wb") as logFile:
        # read1 returns whatever is available, so progress bars that redraw a line with "\r" stay live
        for chunk in iter(lambda: process.stdout.read1(65536), b""):
            sys.stdout.buffer.write(chunk)
            sys.stdout.buffer.flush()
            logFile.write(chunk)
            logFile.flush()
    return process.wait(), time.time() - startTime


# Train the chosen plans one after the other; returns the number of failed runs
def trainAllPlans(plans, epochs, seed, project, workers, resultsFolder=None):
    logFolder = (repositoryRoot / project / "logs").resolve()
    logFolder.mkdir(parents=True, exist_ok=True)
    if resultsFolder is not None: resultsFolder = (repositoryRoot / resultsFolder).resolve()
    startStamp = datetime.now().strftime(timeStampFormat)
    summaryPath = logFolder / f"{startStamp}_trainAllPlans.log"
    if sys.platform == "win32": ctypes.windll.kernel32.SetThreadExecutionState(keepAwakeFlags)
    writeSummaryLine(summaryPath, f"Plans {' '.join(plans)}, {epochs} epochs, seed {seed}, project {project}, results {resultsFolder or 'results/'}; "
                                  f"{describeGitState()}; Python {sys.executable}")
    failedRuns = []
    for planLetter in plans:
        for runLabel, arguments in buildPlanCommands(planLetter, epochs, seed, project, workers):
            logPath = logFolder / f"{datetime.now().strftime(timeStampFormat)}_{runLabel}.log"
            writeSummaryLine(summaryPath, f"Start {runLabel}: python {' '.join(arguments)} (log {logPath.name})")
            exitCode, seconds = runLogged(arguments, logPath, resultsFolder)
            writeSummaryLine(summaryPath, f"{'Done' if exitCode == 0 else 'FAILED'} {runLabel}: exit code {exitCode} after {seconds / 60:.1f} min")
            if exitCode != 0: failedRuns.append(runLabel)
    writeSummaryLine(summaryPath, f"Finished; failed runs: {', '.join(failedRuns) if failedRuns else 'none'}")
    return len(failedRuns)


# Command line entry point
def main():
    parser = argparse.ArgumentParser(description="Train the Phase 1 baselines of plans A, B, C and D one after the other")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--project", default="runs", help="folder for runs and logs, relative to the repository root or absolute")
    parser.add_argument("--plans", nargs="+", default=defaultPlans, choices=defaultPlans)
    parser.add_argument("--workers", type=int, default=None, help="dataloader workers for Plans A, B and C (default: the scripts' own)")
    parser.add_argument("--results-folder", default=None, help="write the results rows here instead of results/ (use it for trial runs)")
    arguments = parser.parse_args()
    failedCount = trainAllPlans(arguments.plans, arguments.epochs, arguments.seed, arguments.project, arguments.workers, arguments.results_folder)
    sys.exit(1 if failedCount else 0)


if __name__ == "__main__":
    main()
