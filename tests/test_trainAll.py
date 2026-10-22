# Tests for the all-plans training driver (scripts/trainAll.py); no training runs, the commands are only built or replaced by stand-ins
import sys

from scripts import trainAll


# Every plan builds the documented command, with the epochs and project passed on, and the Plan B teacher runs at the smaller batch size
def testJobsOfEveryPlan():
    jobsByPlan = {planLetter: trainAll.buildJobs(planLetter, 50, "runs/epochs50") for planLetter in "ABCD"}
    modules = {planLetter: [command[command.index("-m") + 1] for _, command in jobs] for planLetter, jobs in jobsByPlan.items()}
    assert modules == {"A": ["underwater.train"], "B": ["weather.train"] * 3, "C": ["smallobject.train"], "D": ["camoedge.train"]}
    for jobs in jobsByPlan.values():
        for _, command in jobs:
            assert command[command.index("--epochs") + 1] == "50" and command[command.index("--project") + 1] == "runs/epochs50"
    teacherCommand = dict(jobsByPlan["B"])["B weather teacher"]
    assert teacherCommand[teacherCommand.index("--batch") + 1] == str(trainAll.teacherBatchSize)
    assert "--batch" not in dict(jobsByPlan["B"])["B weather student"]
    assert "--workers" not in trainAll.buildJobs("D", 1, "runs", workers=2)[0][1]


# When the teacher fails, the distilled student is skipped instead of failing, the student still runs, and the summary says so
def testDistillationIsSkippedWhenTheTeacherFails(tmp_path, monkeypatch):
    calledJobs = []

    # Stand-in for runCommand: the teacher fails as if the system killed it, every other job succeeds
    def fakeRunCommand(command, logPath):
        calledJobs.append(command[command.index("--mode") + 1])
        return (-9 if "teacher" in command else 0), 1.0

    monkeypatch.setattr(trainAll, "runCommand", fakeRunCommand)
    monkeypatch.setattr(trainAll, "buildPreparation", lambda planLetter: None)
    monkeypatch.setattr(trainAll, "ensureBrackishYaml", lambda: None)
    monkeypatch.setattr(sys, "argv", ["trainAll", "--epochs", "1", "--plans", "B", "--project", str(tmp_path / "runs")])
    assert trainAll.main() == 1
    assert calledJobs == ["teacher", "student"]
    summaryText = next((tmp_path / "runs" / "logs").glob("*_summary.md")).read_text(encoding="utf-8")
    assert "| B weather teacher | failed (-9) |" in summaryText and "| B weather full | skipped (B weather teacher failed) |" in summaryText


# The summary gives minutes per epoch for finished training jobs only
def testSummaryMinutesPerEpoch():
    records = [{"name": "A underwater baseline", "kind": "train", "status": "ok", "seconds": 600.0, "log": "runs/x.log"},
               {"name": "B prepare dataset", "kind": "prepare", "status": "ok", "seconds": 60.0, "log": "runs/y.log"}]
    summaryText = trainAll.renderSummary(records, 5, "runs/epochs5", "2026-10-22_10-00-00")
    assert "| A underwater baseline | ok | 10.0 | 2.0 | `runs/x.log` |" in summaryText
    assert "| B prepare dataset | ok | 1.0 | - | `runs/y.log` |" in summaryText
