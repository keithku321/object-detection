# Setup log: environment, data, one-epoch run and the 50-epoch schedule

What was set up on **2026-10-22**, what went wrong, and how it was fixed. The commands themselves are in [`cmd.md`](cmd.md); the state of the project is in
[`CLAUDE.md`](CLAUDE.md).

**Where it ran:** a Linux cloud container with 4 CPU cores, 16 GB of memory and **no GPU**. The one-epoch numbers below are CPU timings for checking that
every plan trains end to end; they are not results, and their rows were not committed. The 50-epoch runs are meant for the Windows PC with the
RTX 4060 Laptop GPU, through the scheduled task described at the end, which has to be started on that PC.

## Environment

| Item | Value |
|---|---|
| Python | 3.12.3 (the Colab reference) in `.venv`, made with `uv venv --python 3.12 .venv` |
| PyTorch | 2.11.0 CPU build (`--index-url https://download.pytorch.org/whl/cpu`), torchvision 0.26.0 |
| Shared requirements | `uv pip install -r requirements.txt`: Ultralytics 8.4.175, timm 1.0.30, pysodmetrics 1.6.2, gdown 6.4.2 and the rest; `uv pip check` reports no conflicts |
| Tests | `python -m pytest -q`: 90 fast tests passed (the 15 `slow` tests were not run) |

On the Windows PC the existing `.venv` (Python 3.14, CUDA build of PyTorch) stays as it is; run `pip install -r requirements-local.txt` once more so that
`gdown` (new in `requirements.txt`) is installed for the Plan D download.

## Data

| Plan | Dataset | How | Size | Check |
|---|---|---|---|---|
| A | Brackish | already in git (`underwater/dataset/brackish`, clip split); the driver writes the machine's `dataset.yaml` | 0.76 GB | 11,612 / 1,575 / 1,487 images |
| B | DAWN | `python -m weather.prepareDawn --download-to weather/dataset --output weather/dataset/dawn` | 139 MB | 1,027 images, five folds from `common/splits/dawn/` |
| C | VisDrone-DET | Ultralytics `VisDrone.yaml` on first use (downloaded and converted in 56 s here) | 1.94 GB download, 3.7 GB in `datasets/VisDrone` (git-ignored) | 6,471 train and 548 val images |
| D | COD10K + CAMO | `python -m camoedge.prepareCamo --output camoedge/dataset` (new; two Google Drive zips, under a minute here) | 1.8 GB download, 3.5 GB unpacked | 4,040 training pairs, COD10K 2,026 and CAMO 250 test pairs |

## One-epoch run of all four plans (CPU)

`python -m scripts.trainAll --epochs 1` runs the Phase 1 model of every plan into `runs/epochs1/`:

| Job | Status | Minutes (CPU) | Notes |
|---|---|---|---|
| B weather teacher (YOLO26m, fold 0) | PENDING | | first attempt at batch 16 killed for lack of memory (issue 3) |
| B weather student (YOLO26n, fold 0) | ok | 5.8 | pretrained transfer 708/708 |
| B weather full (distilled YOLO26n) | PENDING | | first attempt failed because the teacher had been killed |
| D camoedge E1 (PVTv2-B2) | PENDING | | 25.1 M parameters |
| C smallobject baseline (YOLO26n, VisDrone, 640 px) | PENDING | | |
| A underwater baseline (YOLO26n, Brackish) | PENDING | | |

## Issues found and fixed

1. **A 50-epoch run would have silently reused the 20-epoch Plan A run.** `runs/underwater/baseline-seed0` on the Windows PC holds the finished 20-epoch
   run. `trainDetector` skipped any run folder that had a `best.pt` ("already finished"), and the results log treated any row of the same variant and
   seed as already logged, so `--epochs 50` would have trained nothing and handed back the 20-epoch weights under the new name. Now a run folder (or
   CamoEdge checkpoint) made for another number of epochs is refused with a clear message (`common/training.py`, `camoedge/train.py`), the epoch
   count is part of the duplicate check of every plan, and the driver gives each epoch budget its own project folder (`runs/epochs50`).
2. **"epochs=1" matched "epochs=10".** `findResultRow` compared notes as plain text, so a 1-epoch row could stand in for a 10-epoch run. It now compares
   whole `;`-separated note fields (`common/results.py`), with tests.
3. **The Plan B teacher was killed for lack of memory.** YOLO26m at batch 16 reached 13.9 GB of resident memory on the CPU and the kernel killed it
   (exit code -9, no Python error). An 8 GB laptop GPU is unlikely to fit it either. The driver trains the teacher at batch 8; the teacher is a reference
   model, so this does not change the comparison between the students.
4. **The distillation job failed with a confusing message when its teacher had failed.** The driver now marks it "skipped (B weather teacher failed)".
5. **Plan D had no download script.** `camoedge/prepareCamo.py` downloads the two zips of the SINet release with `gdown` (added to `requirements.txt`;
   a new shared dependency needs two owners' approval), checks the SHA-256 values now recorded in the manifest and the protocol counts. It also settled
   an open question: the "COD10K-train" zip already holds the 1,000 CAMO training images, so it is exactly the 4,040-pair training set.
6. **gdown 6 removed `--fuzzy`.** The script passes the Google Drive file ids instead of share links.
7. **A CamoEdge epoch was silent until it ended**, which looks like a hang in a terminal or scheduled task (an epoch takes minutes on a GPU and well over
   an hour on this CPU). It now prints ten progress lines per epoch with the running loss and the time left.
8. **Driver details found while testing it:** its log files were only written at the end (now flushed as the output arrives), and a `--project` outside
   the repository made the summary crash.
9. **Windows specifics of the scheduled task:** a laptop sleeps after its idle timeout and pauses training (the driver now holds a "system required"
   request while it runs); `powershell -File` passes `-Plans A,C` as one text (the script splits it); paths with spaces or apostrophes are quoted for
   PowerShell (checked with PowerShell 7 and stand-ins for the Windows-only cmdlets).
10. **Git:** PR #1 had been squash-merged, so the branch was restarted from `main` (plus the open PR #2, so that the newest plotting code was tested).
    Rewriting the published branch was not allowed, so the old commits were joined with an `ours` merge, which changes no file.

Not fixed, because it is a data question and not a bug: **crab is never detected in the 20-epoch Plan A run.** The crab labels mostly mark real
crabs, but at 640 px a median crab (76 x 62 px in the 1920 x 1080 frame) shrinks to about 25 x 21 px of low-contrast blur, the Roboflow images are
compressed to about 52 KB per frame, and a few crab boxes look wrong (one marks a fish-like shape, one an empty patch). Compare with the original
Aalborg release before blaming the model, and look at crab at 1280 px; changing the size or dataset for the report is a protocol decision.

## The 50-epoch runs on the Windows PC

The cloud container cannot reach the PC, so the task has to be registered there. In PowerShell, from the repository root:

```powershell
git pull
.venv\Scripts\Activate.ps1
pip install -r requirements-local.txt
powershell -ExecutionPolicy Bypass -File scripts\registerTrainingTask.ps1 -StartNow
```

A PowerShell window opens in the repository root and runs `python -m scripts.trainAll --epochs 50`: Plan B (teacher, student, distilled student),
then D, C and A, each resumable. Logs and the summary table go to `runs\epochs50\logs\`. Keep the laptop plugged in and the lid open.

Expected duration on the RTX 4060 Laptop GPU (only Plan A is measured: 20 epochs took 2.0 hours, so 6 minutes per epoch; the others are scaled from it by
image count and model size, so treat them as rough):

| Plan | 50 epochs, roughly |
|---|---|
| A underwater baseline | 5 hours (measured rate) |
| B teacher + student + distilled student | PENDING |
| C small-object baseline | PENDING |
| D CamoEdge E1 | PENDING |

**No plan except possibly Plan D fits in 2 hours, so the session was finished instead of waiting.**
