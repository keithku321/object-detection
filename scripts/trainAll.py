# Train the first model of every plan one after the other, with the same number of epochs, and write a summary; used by the Windows
# scheduled task (scripts/registerTrainingTask.ps1) and for one-epoch checks of a new machine.
#   python -m scripts.trainAll --epochs 1                      (one-epoch check of all four plans, into runs/epochs1)
#   python -m scripts.trainAll --epochs 50                     (the long run, into runs/epochs50)
#   python -m scripts.trainAll --epochs 50 --plans A C         (only Plans A and C)
#   python -m scripts.trainAll --epochs 50 --dry-run           (print the commands only)
### Flow: for each chosen plan, prepare its dataset when it is missing (download scripts of the plan), then run its training commands as
### separate processes, so that a crash in one plan never stops the others. Every command is the documented one from cmd.md; its output is
### shown in this terminal and copied to <project>/logs/. Re-running the same command resumes unfinished runs and skips finished ones.
# Jobs per plan (the Phase 1 models of each plan, seed 0):
#   A  underwater baseline (YOLO26n on Brackish, clip split)
#   B  teacher (YOLO26m), plain student and fully distilled student (YOLO26n) on DAWN fold 0
#   C  small-object baseline (YOLO26n on VisDrone at 640 px; Ultralytics downloads VisDrone on first use)
#   D  CamoEdge E1 (PVTv2-B2 encoder, plain decoder, BCE) on COD10K + CAMO
# Each epoch budget gets its own project folder (runs/epochs<N>), because a run folder of another epoch count is refused (common/training.py).
import argparse
import ctypes
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# Repository root; every command runs from here so that the python -m module paths resolve
repositoryRoot = Path(__file__).resolve().parent.parent
# Plan letters in the order they run; Plan B is cheap and goes first so that a broken setup shows up quickly
planOrder = ("B", "D", "C", "A")
# Seed of every job; the multi-seed runs of the plans come later
jobSeed = 0
# Batch size of the Plan B teacher (YOLO26m). At the default 16 it needed 13.9 GB of memory on a CPU and was killed in a 16 GB machine, and it
# would not fit an 8 GB laptop GPU either; the teacher is a reference model, so its batch size does not change the student comparison
teacherBatchSize = 8
# Dataset paths that the jobs read, relative to the repository root
brackishRoot = Path("underwater/dataset/brackish")
dawnRoot = Path("weather/dataset/dawn")
camoRoot = Path("camoedge/dataset/camo")
# Windows SetThreadExecutionState flags: ES_CONTINUOUS keeps the setting until it is cleared, ES_SYSTEM_REQUIRED stops idle sleep
continuousFlag, systemRequiredFlag = 0x80000000, 0x00000001
# Size of the pieces in which a job's output is passed through to the terminal and the log file
outputChunkSize = 4096


# Ask Windows not to sleep while the jobs run (a laptop otherwise pauses training after its idle timeout); does nothing elsewhere
def keepComputerAwake(enabled):
    if sys.platform != "win32": return
    ctypes.windll.kernel32.SetThreadExecutionState(continuousFlag | systemRequiredFlag if enabled else continuousFlag)


# Write underwater/dataset/brackish/dataset.yaml when it is missing; it is git-ignored because it holds the absolute path of this machine
def ensureBrackishYaml():
    yamlPath = repositoryRoot / brackishRoot / "dataset.yaml"
    if yamlPath.exists() or not (repositoryRoot / brackishRoot / "images" / "train").exists(): return
    from common.datasets import getManifestEntry, writeDatasetYaml
    writeDatasetYaml((repositoryRoot / brackishRoot).resolve(), getManifestEntry("brackish")["classNames"], testSplit="images/test")
    print(f"Wrote {yamlPath}")


# Preparation command of a plan when its dataset is missing, otherwise None (Plan C needs none: Ultralytics downloads VisDrone itself)
def buildPreparation(planLetter):
    python = sys.executable
    if planLetter == "A" and not (repositoryRoot / brackishRoot / "images" / "train").exists():
        return [python, "-m", "underwater.prepareBrackish", "--output", str(brackishRoot.parent)]
    if planLetter == "B" and not (repositoryRoot / dawnRoot / "images").exists():
        return [python, "-m", "weather.prepareDawn", "--download-to", str(dawnRoot.parent), "--output", str(dawnRoot)]
    if planLetter == "D" and not (repositoryRoot / camoRoot / "TrainDataset" / "Imgs").exists():
        return [python, "-m", "camoedge.prepareCamo", "--output", str(camoRoot.parent)]
    return None


# Training commands of one plan as (job name, command) pairs; device and workers are passed on when given
def buildJobs(planLetter, epochs, project, device=None, workers=None):
    python = sys.executable
    common = ["--seed", str(jobSeed), "--epochs", str(epochs), "--project", str(project)]
    deviceArguments = ["--device", str(device)] if device else []
    workerArguments = ["--workers", str(workers)] if workers is not None else []
    if planLetter == "A":
        return [("A underwater baseline", [python, "-m", "underwater.train", "--variant", "baseline", "--data", str(brackishRoot / "dataset.yaml"),
                                           *common, *deviceArguments, *workerArguments])]
    if planLetter == "B":
        return [(f"B weather {mode}", [python, "-m", "weather.train", "--mode", mode, "--fold", "0", "--dataset", str(dawnRoot), *common,
                                       *(["--batch", str(teacherBatchSize)] if mode == "teacher" else []), *deviceArguments, *workerArguments])
                for mode in ("teacher", "student", "full")]
    if planLetter == "C":
        return [("C smallobject baseline", [python, "-m", "smallobject.train", "--variant", "baseline", *common, *deviceArguments, *workerArguments])]
    if planLetter == "D":
        return [("D camoedge E1", [python, "-m", "camoedge.train", "--experiment", "E1", "--dataset", str(camoRoot), *common, *deviceArguments])]
    raise ValueError(f"Unknown plan {planLetter}")


# Run one command from the repository root, showing its output live and copying it to a log file; returns (exit code, seconds)
def runCommand(command, logPath):
    logPath.parent.mkdir(parents=True, exist_ok=True)
    environment = {**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8"}
    startTime = time.time()
    with logPath.open("wb") as logFile:
        logFile.write((" ".join(command) + "\n\n").encode("utf-8"))
        process = subprocess.Popen(command, cwd=repositoryRoot, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=environment)
        try:
            for chunk in iter(lambda: process.stdout.read1(outputChunkSize), b""):
                sys.stdout.buffer.write(chunk)
                sys.stdout.buffer.flush()
                logFile.write(chunk)
        except KeyboardInterrupt:
            process.terminate()
            raise
        returnCode = process.wait()
    return returnCode, time.time() - startTime


# Markdown summary of the finished jobs: one table row per job with its status, minutes, minutes per epoch and log file
def renderSummary(records, epochs, project, startedAt):
    lines = [f"# Training summary: {epochs} epoch(s), started {startedAt}", "", f"Project folder: `{project}`. Python: `{sys.executable}`.", "",
             "| Job | Status | Minutes | Minutes per epoch | Log |", "|---|---|---|---|---|"]
    for record in records:
        minutes = record["seconds"] / 60
        perEpoch = f"{minutes / epochs:.1f}" if record["kind"] == "train" and record["status"] == "ok" else "-"
        lines.append(f"| {record['name']} | {record['status']} | {minutes:.1f} | {perEpoch} | `{record['log']}` |")
    return "\n".join(lines) + "\n"


# Command line entry point
def main():
    parser = argparse.ArgumentParser(description="Train the Phase 1 model of every plan one after the other")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--plans", nargs="+", default=list(planOrder), choices=list(planOrder), help="plan letters to run (default: all)")
    parser.add_argument("--project", default=None, help="folder for the runs (default: runs/epochs<N>)")
    parser.add_argument("--device", default=None, help="for example 0 for the first GPU or cpu (default: chosen by the scripts)")
    parser.add_argument("--workers", type=int, default=None, help="dataloader workers of the YOLO plans (default: 2 on Windows, 4 elsewhere)")
    parser.add_argument("--dry-run", action="store_true", help="print the commands and exit")
    arguments = parser.parse_args()
    project = Path(arguments.project or f"runs/epochs{arguments.epochs}")
    startedAt = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    logFolder = repositoryRoot / project / "logs"
    chosenPlans = [planLetter for planLetter in planOrder if planLetter in arguments.plans]
    if arguments.dry_run:
        for planLetter in chosenPlans:
            preparation = buildPreparation(planLetter)
            if preparation: print("prepare:", " ".join(preparation))
            for jobName, command in buildJobs(planLetter, arguments.epochs, project, arguments.device, arguments.workers): print(f"{jobName}:", " ".join(command))
        return 0
    keepComputerAwake(True)
    records = []
    try:
        ensureBrackishYaml()
        for planLetter in chosenPlans:
            preparation = buildPreparation(planLetter)
            if preparation:
                logPath = logFolder / f"{startedAt}_{planLetter}-prepare.log"
                print(f"\n===== {planLetter}: preparing the dataset =====\n", flush=True)
                returnCode, seconds = runCommand(preparation, logPath)
                records.append({"name": f"{planLetter} prepare dataset", "kind": "prepare", "status": "ok" if returnCode == 0 else f"failed ({returnCode})",
                                "seconds": seconds, "log": logPath.relative_to(repositoryRoot).as_posix()})
                if returnCode != 0:
                    print(f"Preparation of plan {planLetter} failed; its training is skipped", flush=True)
                    continue
                if planLetter == "A": ensureBrackishYaml()
            for jobName, command in buildJobs(planLetter, arguments.epochs, project, arguments.device, arguments.workers):
                logPath = logFolder / f"{startedAt}_{jobName.replace(' ', '-')}.log"
                print(f"\n===== {jobName}: {arguments.epochs} epoch(s) =====\n{' '.join(command)}\n", flush=True)
                returnCode, seconds = runCommand(command, logPath)
                records.append({"name": jobName, "kind": "train", "status": "ok" if returnCode == 0 else f"failed ({returnCode})", "seconds": seconds,
                                "log": logPath.relative_to(repositoryRoot).as_posix()})
                print(f"\n----- {jobName}: {records[-1]['status']} after {seconds / 60:.1f} minutes -----", flush=True)
    finally:
        keepComputerAwake(False)
        if records:
            summary = renderSummary(records, arguments.epochs, project.as_posix(), startedAt)
            logFolder.mkdir(parents=True, exist_ok=True)
            (logFolder / f"{startedAt}_summary.md").write_text(summary, encoding="utf-8")
            (logFolder / f"{startedAt}_summary.json").write_text(json.dumps(records, indent=1), encoding="utf-8")
            print("\n" + summary, flush=True)
    return 0 if all(record["status"] == "ok" for record in records) else 1


if __name__ == "__main__":
    sys.exit(main())
