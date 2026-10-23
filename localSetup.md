# Local training setup

How the local Windows GPU PC below was set up on 2026-10-22 to train all four plans locally (nothing in the cloud), what went wrong on the way and how it was
fixed, and what the one-epoch check of every plan measured. Commands are in [`cmd.md`](cmd.md); the project rules are in [`CLAUDE.md`](CLAUDE.md).

## This PC

| Part | Specification |
|---|---|
| CPU | Intel Core i7-14700KF, 20 cores (8 performance + 12 efficiency), 28 threads |
| Memory | 32 GB DDR5 at 4,800 MT/s (2 × 16 GB) |
| GPU | NVIDIA GeForce RTX 5060 Ti, 16 GB (16,311 MiB), compute capability 12.0 (Blackwell, `sm_120`), power limit 180 W |
| NVIDIA driver | 616.64 (CUDA driver API 13.4) |
| Mainboard | MSI MEG Z790 ACE (MS-7D86) |
| Operating system | Windows 11 Pro 10.0.26200 (Traditional Chinese display language) |
| Disk of the repository | `X:`, WD Red 2 TB **hard disk** (WD20EFRX, SATA), 1.4 TB free |
| Other disks | NVMe SSDs: Samsung 990 PRO 2 TB (`S:`), two XPG GAMMIX S50 Core 1 TB (`C:` and `I:`); older SATA disks `E:`, `F:`, `G:` |

The repository sits on a spinning disk, and Ultralytics warned "Slow image access detected" in its quick read test (about 5.7 MB/s on DAWN and 33 MB/s on
VisDrone). Because a game shared the GPU during the one-epoch check (see below), the check cannot tell how much the disk costs. If epochs are slow with the
GPU free, a copy of the repository and the VisDrone folder on the Samsung 990 PRO (`S:`) is the first thing to try.

## Software

| What | Version | Notes |
|---|---|---|
| Python | 3.12.0 | The reference version (Colab runs 3.12). Created with `py -3.12 -m venv .venv` in the repository root |
| PyTorch | 2.11.0+cu128, torchvision 0.26.0+cu128 | CUDA 12.8 build from `requirements-local.txt`; includes `sm_120` kernels for the RTX 50 series; cuDNN 9.19 |
| Ultralytics | 8.4.175 | The pin. Ultralytics prints "New 8.4.183 available": ignore it (decision D5) |
| PySODMetrics | 1.6.2 | Installs normally on Python 3.12; the Python 3.14 workaround in `requirements-local.txt` is not needed |

`pip check` reports no broken requirements, and `python -m pytest -q` passes 87 fast tests (the 83 of `main` plus four new ones: the results-folder variable,
the `-dirty` check, and two for Plan D's final weights). The 15 `slow` tests were not run.

## Datasets on this PC

| Plan | Dataset | Location | How it got there |
|---|---|---|---|
| A | Brackish, clip-based split (11,612 / 1,575 / 1,487 images) | `underwater/dataset/brackish/` | Tracked in git; only `dataset.yaml` was regenerated with the command in `cmd.md` (it holds this machine's path) |
| B | DAWN, 1,027 images, five folds (210 / 201 / 202 / 203 / 211 images) | `weather/dataset/dawn/` | `python -m weather.prepareDawn --download-to weather/dataset --output weather/dataset/dawn`; the Mendeley SHA-256 values matched, one `train` box was dropped as expected, and the fold lists came out identical to `common/splits/dawn/` |
| C | VisDrone2019-DET (6,471 train, 548 val images) | `X:\yolo-fyp\repos\keithku321\datasets\VisDrone` | Ultralytics' own `VisDrone.yaml` download (see issue 2 for why it lands next to the repository) |
| D | COD10K + CAMO, arranged as `TrainDataset/` (4,040 pairs) and `TestDataset/` (COD10K 2,026, CAMO 250, CHAMELEON 76) | `camoedge/dataset/camo/` (git-ignored, about 1.8 GB) | The two zips of the SINet repository, fetched with `gdown` (commands in `cmd.md`); D12 keeps COD10K a manual download, and none of it is committed |

SHA-256 of the SINet zips as downloaded on 2026-10-22, for the `cod10k` and `camo` entries of `datasets/manifest.json` once their owner confirms them:

| Zip | Bytes | SHA-256 |
|---|---|---|
| `TrainDataset.zip` (Google Drive id `1D9bf1KeeCJsxxri6d2qAC7z6O1X_fxpt`) | 1,100,100,433 | `f2042b827bcecf1169f965385862e742fb25840808fff3cc30313e04c015ae02` |
| `TestDataset.zip` (Google Drive id `1QEGnP9O7HbN_2tH999O3HRIsErIVYalx`) | 699,895,455 | `436937ccafef4f98a63798199529e83423d307977de8363aa8d1d294cc245575` |

## Issues found during the setup, and their fixes

1. **Three Python versions, none of them the default you want.** `python` on `PATH` is 3.11 from the Microsoft Store, and the `py` launcher defaults to 3.15. The
   environment was created with `py -3.12 -m venv .venv`, and every command uses `.venv\Scripts\python.exe` (or an activated `.venv`).
2. **Ultralytics thinks it is a git checkout.** Because `.venv` lies inside the repository, Ultralytics finds the repository's `.git` above its own files and takes the
   folder *above* the repository as the root of its datasets: `datasets_dir` became `X:\yolo-fyp\repos\keithku321\datasets`, so VisDrone is stored next to the
   repository, not in it (which also keeps the 1.8 GB out of the repository folders, in line with decision D12). Its `weights_dir` is `weights/` in the repository,
   where the AMP check stores a second `yolo26n.pt`; `*.pt` is git-ignored. To put datasets somewhere else: `yolo settings datasets_dir=<folder>` (before the
   first Plan C run, or move the folder afterwards).
3. **A trial run would have blocked the real run.** Plans A, B and C decide whether a run is "already logged" by task, model, variant, seed and split or fold, but
   **not by the number of epochs**. A one-epoch row of `baseline` seed 0 in `results/smallobject.csv` (or `student`, fold 0, seed 0 in `results/weather.csv`)
   would make the later 50-epoch run skip its evaluation, log nothing, and draw its plot with the one-epoch numbers. Plans C and D have no `--label` to rename a
   trial. Fix: `common/results.py` reads an optional environment variable `FYP_RESULTS_FOLDER`, and the new runner `scripts/trainAllPlans.py` sets it with
   `--results-folder`, so the one-epoch check wrote its rows and plots to `runs/epoch1-check/results/` and `results/` stayed clean.
4. **The 20-epoch Plan A row of PR #2 collides with a 50-epoch run.** `results/underwater.csv` (merged on 2026-10-22) holds `baseline`, seed 0, 20 epochs. For the
   reason in issue 3 a 50-epoch `baseline` seed 0 run would train for hours and then log nothing. Fix: the runner names Plan A's run `baseline-<epochs>ep`
   (`--label`), so the 50-epoch run is logged as `baseline-50ep` and the summary does not average a 20-epoch and a 50-epoch run as if they were one variant.
   **For the team:** the root cause (rows matched without the epoch count) is still there for anyone who re-runs a variant with another length; the closed PR #3
   proposed matching on the whole `epochs=` note. Plan B's runs keep their plain names, because the distilled students find their teacher by the run name
   `teacher-fold<k>-seed0`.
5. **Training output must be visible** (rule "Do" in `CLAUDE.md`, added by PR #2). The runner streams each run's output live to its console and copies it to a log
   file, and forces UTF-8 output in the runs (`PYTHONIOENCODING`), so the log and the console agree whatever the Windows code page is.
6. **Sleep.** A long run would stop when Windows goes to sleep; the runner asks Windows to stay awake (`SetThreadExecutionState`) until the last run ends. The
   screen may still turn off.
7. **Every result after the first was marked `-dirty`.** `common/results.py` added `-dirty` to the commit of a results row whenever `git status` showed any
   change. Since PR #2 the results files and plots are tracked, so the first run of a session made every later row "dirty" although no code had changed (and
   the untracked `UpgradeLog.htm` that Visual Studio writes when it upgrades the project file did so even before). Fix: `results/` and `runs/` no longer count
   as changes of the code, and `UpgradeLog*.htm` is git-ignored.
8. **Plan D had no final weights file and printed nothing for minutes.** Plans A to C copy their final weights to `<project>/weights/`; Plan D did not, although
   the demo expects `camoedge-<variant>-seed<k>.pt`. `python -m camoedge.train` now writes that file (weights in half precision, about 48 MiB instead of
   96 MiB, below GitHub's 100 MB limit), and prints progress ten times per epoch and per test set, so the terminal shows that training and the twelve-minute
   evaluation are moving.
9. **The GitHub repository is public**, although CLAUDE.md asks the team and supervisor to decide that first (decision D9). Committing weights therefore means
   publishing them, so only Plans A and D may be committed (decision D8, `runs/weights/README.md`); the weights of Plans B and C stay off GitHub, because their
   data terms (DAWN non-commercial, VisDrone unconfirmed) conflict with the AGPL-3.0 of Ultralytics-trained weights.

## One-epoch check of every plan

Run on 2026-10-22 from commit `831a783e` plus the changes of this setup, with the runner in a visible terminal:
`python -m scripts.trainAllPlans --epochs 1 --project runs/epoch1-check --results-folder runs/epoch1-check/results` (Plans A, B and D in one call, Plan C in a
second call once VisDrone had downloaded). Every run finished with exit code 0, logged its results rows to `runs/epoch1-check/results/` (not `results/`) and saved
its plots there, and copied its final weights to `runs/epoch1-check/weights/`. Seed 0, 640 px (Plan D 352 px), batch 16, 2 dataloader workers, AMP on.

**Timing caveat:** a game (`HTGame.exe`) used about 80 % of the GPU and 5.6 GB of its memory during the whole check, so all times below are for a shared GPU.

| Plan | Run | Time of the epoch (train + val) | Whole run (setup, epoch, evaluation, plots) | Result after one epoch |
|---|---|---|---|---|
| A | `baseline-1ep`, YOLO26n, Brackish | 7.3 min | 9.7 min (includes building the label cache on the hard disk, done once) | `val`: mAP50 0.462, mAP50-95 0.238, precision 0.473, recall 0.452; transfer 696/708 (class layers new) |
| B | `teacher`, YOLO26m, DAWN fold 0 | 2.0 min | 4.8 min | `val`: mAP50 **0.001**, mAP50-95 0.000 (see below); transfer 768/768; batch 16 fitted in GPU memory beside the game |
| B | `student`, YOLO26n, DAWN fold 0 | 1.1 min | 2.2 min | `val`: mAP50 0.217, mAP50-95 0.116; per weather mAP50 fog 0.217, rain 0.330, snow 0.384, sand 0.200 |
| C | `baseline`, YOLO26n, VisDrone | 3.9 min | 5.9 min | `val`: mAP50 0.148, mAP50-95 0.077; COCO AP small 0.037, medium 0.122, large 0.227 (maxDets 500) |
| D | `E1`, PVTv2-B2, COD10K + CAMO | 4.2 min (training only; Plan D has no validation split) | 16.3 min (the test-set evaluation takes about 12 min on one CPU core) | COD10K: S-measure 0.699, mean E-measure 0.707, weighted F 0.325, MAE 0.116; CAMO: S-measure 0.755, weighted F 0.466, MAE 0.150 |

These are pipeline checks, **not numbers for the report**. Observations:
- **Plan B's teacher scored almost zero after one epoch** while the smaller student reached 0.217, and its losses were higher than the student's (validation class
  loss 8.0 against 1.9). The pretrained weights did load (768/768, and because all six DAWN classes are COCO classes, Ultralytics' `cls_remap` also carries
  the COCO class weights over), so this is not a broken transfer. One epoch of DAWN is only about 50 iterations; check that the teacher's mAP rises within the
  first epochs of the 50-epoch run before using it for distillation.
- Pretrained transfer: Plan C also transferred 708/708 (some VisDrone classes match COCO names); Plan A transferred 696/708, as on the first PC.
- Each detection evaluation leaves an empty `runs/detect/val-<n>` folder (Ultralytics' default output folder for validation). It is harmless and git-ignored.
- `huggingface_hub` warns on Windows that its cache cannot use symbolic links without Developer Mode, and that requests are unauthenticated. Both are harmless:
  the PVTv2-B2 weights of Plan D download once and are copied instead of linked.

Plan D after the setup changes: the final weights are now copied automatically and in half precision. Re-exporting the check's `last.pt` gave a 47.9 MiB file
(the old full-precision copy was 95.7 MiB, `last.pt` with the optimiser state 287 MiB) that loads back as float32 and predicts within 0.002 of the full-precision
model on eight CAMO images.

## 50-epoch training

The Phase 1 baselines of all four plans, 50 epochs each, seed 0, run one after the other by the Windows scheduled task `ObjectDetection-TrainAllPlans`,
registered and started on the evening of 2026-10-22 with `scripts/registerTrainingTask.ps1` (commands in `cmd.md`) right after this file was committed. It runs
`python -m scripts.trainAllPlans --epochs 50 --project runs` in a visible Windows PowerShell window in the repository root, as the signed-in user; the exact start
time and every run's start, end and exit code are in `runs/logs/<date>_trainAllPlans.log`. If the PC restarts or the user signs out, the task stops; start it
again with `Start-ScheduledTask -TaskName ObjectDetection-TrainAllPlans` and it resumes the unfinished run. The runs are A `baseline-50ep`, B `teacher`
then `student` on fold 0, C `baseline`, D `E1`; results rows go to `results/<task>.csv`, plots to `results/plots/<task>/`, final weights to `runs/weights/`, and
one log per run plus a summary log to `runs/logs/`.

50 epochs is the default length of Plans A and C; Plan B's scripts default to 100 epochs and Plan D's to 30, so the B and D rows of this run do not use their
plans' default length (the notes column of each row records `epochs=50`). **Result (2026-10-22 21:59 to 2026-10-23 01:38, 3 h 40 min, every run exit code 0, all rows from the clean commit `8b527eca`).** The game was closed soon
after the start, so the GPU was free and the runs were much faster than the estimate below. Seed 0 only, so these are baselines to compare against, not
final numbers (those need three seeds).

| Plan | Run | Time | Reported result |
|---|---|---|---|
| A | `baseline-50ep`, YOLO26n, Brackish `val` (`best.pt`) | 62.9 min | mAP50 **0.597**, mAP50-95 0.344, precision 0.580, recall 0.566. Best epoch 6 (val mAP50 0.601), then flat: the overfitting seen in the 20-epoch run. Per class mAP50: starfish 0.976, jellyfish 0.860, fish 0.847, small_fish 0.796, shrimp 0.099, **crab 0.000** (1,226 boxes, still never detected) |
| B | `teacher`, YOLO26m, DAWN fold 0 (`last.pt`) | 16.2 min | mAP50 **0.647**, mAP50-95 0.382; fog 0.654, rain 0.706, snow 0.866, sand 0.630 |
| B | `student`, YOLO26n, DAWN fold 0 (`last.pt`) | 5.7 min | mAP50 **0.635**, mAP50-95 0.405; fog 0.585, rain 0.636, snow 0.835, sand 0.732 |
| C | `baseline`, YOLO26n, VisDrone `val` | 45.2 min | mAP50 **0.291**, mAP50-95 0.159; COCO AP small **0.083**, medium 0.245, large 0.439 |
| D | `E1`, PVTv2-B2, test sets (`last.pt`) | 89.5 min | COD10K: S-measure **0.853**, mean E-measure 0.882, weighted F 0.717, MAE 0.029; CAMO: S-measure **0.854**, weighted F 0.780, MAE 0.055 |

- **The Plan B teacher learned:** val mAP50 0.571 after its first epoch of this run, best 0.664 (epoch 45), 0.646 at the end. The 0.001 of the one-epoch check
  was an effect of training for a single epoch, not a broken teacher. But the teacher is only 0.012 mAP50 above the student on fold 0, and lower than it on
  sand (0.630 against 0.732) and in mAP50-95 (0.382 against 0.405): with this little headroom, distillation has little to gain from this teacher. Plan B's owner
  should check this on the other folds and consider the 100 epochs of the plan before Phase 2.
- Committed with this result: the rows in `results/*.csv`, the plots in `results/plots/`, and the final weights `runs/weights/underwater-baseline-50ep-seed0.pt`
  (5.4 MB) and `runs/weights/camoedge-E1-seed0.pt` (50 MB). The Plan B and C weights stay on this PC (decision D8).

Expected duration before the run, from the one-epoch times above (shared GPU): A about 6.1 h, B about 2.6 h, C about 3.3 h, D about 3.7 h, **about 16 hours in total**. The
one-epoch runs had mosaic augmentation switched off (Ultralytics turns it off for the last 10 epochs, which was the only epoch here), so the long runs of A and C
may be somewhat slower per epoch; with the GPU free they should be faster.
