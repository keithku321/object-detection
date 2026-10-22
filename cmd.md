# Command reference

Copy-paste commands for this repository. Run everything from the repository root. The blocks use PowerShell (Windows); on Colab or Linux the same
`python -m ...` lines work in a shell, with `/` in paths. **Keep this file in step with the code: when a command line, flag or default changes, update it here.**

Contents: [Setup](#setup) · [Datasets](#datasets) · [Train](#train-locally) · [All plans and the Windows scheduled task](#all-plans-in-one-go-and-the-windows-scheduled-task) · [Results and plots](#results-and-plots) · [Evaluate and export](#evaluate-speed-and-export) · [Demo](#demo) · [Tests](#tests) · [Troubleshooting](#troubleshooting)

Conventions used below:
- `--project runs` puts all runs in `runs/` (git-ignored). A run folder is `runs/<task>/<run name>/`, with `weights/`, Ultralytics' own charts and `results.csv`. The
  **results plot is saved straight into the tracked folder `results/plots/<task>/`**, so it can go into a pull request (see [Results and plots](#results-and-plots)).
- Re-running the same training command **resumes** an unfinished run and **skips** a finished one. Use `--label` (Plans A) or a new `--seed` to start a separate run.
- Every finished run appends a row to `results/<task>.csv`. A short trial run goes into the same file, so give trials a `--label` or delete the row afterwards.
- Defaults: image size 640, batch 16, 2 dataloader workers on Windows, YOLO26 `n` scale. Add `--workers 0` if the dataloader hangs, `--batch 8` on out-of-memory.

## Setup

Create the virtual environment (Python 3.12 is the reference; 3.14 also works with the CUDA wheels):

```powershell
python -m venv .venv
```

```powershell
.venv\Scripts\Activate.ps1
```

Install for a PC with an NVIDIA GPU (CUDA build of PyTorch, about 2.7 GB). Colab uses `requirements.txt` instead:

```powershell
pip install -r requirements-local.txt
```

Switch an existing CPU-only environment to the CUDA build:

```powershell
pip install torch==2.11.0+cu128 torchvision==0.26.0+cu128 --index-url https://download.pytorch.org/whl/cu128
```

Check that PyTorch sees the GPU (should print `True` and the GPU name):

```powershell
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '-')"
```

Print Python, PyTorch, Ultralytics and GPU details (for bug reports):

```powershell
python -c "from common.environment import printEnvironment; printEnvironment()"
```

GPU usage while training (second terminal):

```powershell
nvidia-smi -l 2
```

Python 3.14 only: install PySODMetrics for Plan D (see `requirements-local.txt`):

```powershell
pip install scikit-image scikit-learn scipy
pip install pysodmetrics==1.6.2 --no-deps
```

Linux machine without a GPU (for example a cloud container), Python 3.12 with `uv`, CPU build of PyTorch, then the shared requirements (bash):

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cpu
uv pip install --python .venv/bin/python -r requirements.txt
```

## Datasets

Datasets are never committed (Brackish excepted). Full terms and links: `datasets/README.md`.

Plan A, Brackish (0.7 GB download, clip-based split, writes `underwater/dataset/brackish/` with `dataset.yaml`):

```powershell
python -m underwater.prepareBrackish --output underwater/dataset
```

Same, but keep the Roboflow split (leaks video frames across splits; say so in any report):

```powershell
python -m underwater.prepareBrackish --output underwater/dataset --keep-original-split
```

Build `brackish.zip` for Google Drive as well:

```powershell
python -m underwater.prepareBrackish --output underwater/dataset --zip
```

Regenerate only `underwater/dataset/brackish/dataset.yaml` when the images are already there (the file is git-ignored and holds this machine's path):

```powershell
python -c "import json; from pathlib import Path; from common.datasets import writeDatasetYaml; names = next(e for e in json.loads(Path('datasets/manifest.json').read_text(encoding='utf-8'))['datasets'] if e['name'] == 'brackish')['classNames']; print(writeDatasetYaml(Path('underwater/dataset/brackish').resolve(), names, testSplit='images/test'))"
```

Plan B, DAWN (139 MB, downloads from Mendeley Data, verifies SHA-256, renumbers classes, writes the five folds):

```powershell
python -m weather.prepareDawn --download-to weather/dataset --output weather/dataset/dawn
```

DAWN from raw folders you already have:

```powershell
python -m weather.prepareDawn --source weather/dataset/dawn_raw --output weather/dataset/dawn
```

Plan C, VisDrone: Ultralytics downloads and converts it by itself on the first training run (1.8 GB, goes to its own datasets folder); nothing to prepare.

Plan D, COD10K and CAMO (1.8 GB in two zips from the SINet authors' Google Drive, checked against the SHA-256 in the manifest, unzipped to `camoedge/dataset/camo` and counted: 4,040 training pairs, 2,026 COD10K and 250 CAMO test pairs). Needs `gdown` from `requirements.txt`:

```powershell
python -m camoedge.prepareCamo --output camoedge/dataset
```

If Google Drive refuses the file (it limits popular files for about a day), download the two zips by hand from `datasets/README.md`, put them in `camoedge/dataset/downloads/` and run the same command. The examples below pass `camoedge/dataset/camo` as `--dataset`.

## Train locally

### Plan A: underwater (`underwater/train.py`)

Variants: `baseline`, `yolo11`, `yolov8`, and `<block>-<placement>` with blocks `cbam`, `eca`, `simam`, `ca` and placements `a`, `b`, `c` (for example `eca-a`).

Baseline, 50 epochs, seed 0:

```powershell
python -m underwater.train --variant baseline --seed 0 --epochs 50 --data underwater/dataset/brackish/dataset.yaml --project runs
```

Attention variant:

```powershell
python -m underwater.train --variant eca-a --seed 0 --epochs 50 --data underwater/dataset/brackish/dataset.yaml --project runs
```

One-epoch check that everything works (separate run name, so it never blocks a real run):

```powershell
python -m underwater.train --variant baseline --seed 0 --epochs 1 --label epoch1-check --data underwater/dataset/brackish/dataset.yaml --project runs
```

Final multi-seed runs (seeds 0, 1, 2), one after the other:

```powershell
foreach ($seed in 0,1,2) { python -m underwater.train --variant baseline --seed $seed --epochs 50 --data underwater/dataset/brackish/dataset.yaml --project runs }
```

Evaluate on the test split instead of `val`:

```powershell
python -m underwater.train --variant baseline --seed 0 --epochs 50 --eval-split test --data underwater/dataset/brackish/dataset.yaml --project runs
```

Write the model YAML files of every attention variant and exit:

```powershell
python -m underwater.train --write-configs
```

### Plan B: weather distillation (`weather/train.py`)

Modes: `student` (W0), `teacher` (W1), `lambda0`, `feat`, `cls`, `box`, `full`, `full-fg`, `full-both-heads`. A distilled student needs the teacher of the same fold.
Reported numbers come from `last.pt`, not `best.pt`.

Teacher, then plain student, then the full distillation student, on fold 0:

```powershell
python -m weather.train --mode teacher --fold 0 --seed 0 --epochs 100 --dataset weather/dataset/dawn --project runs
```

```powershell
python -m weather.train --mode student --fold 0 --seed 0 --epochs 100 --dataset weather/dataset/dawn --project runs
```

```powershell
python -m weather.train --mode full --fold 0 --seed 0 --epochs 100 --dataset weather/dataset/dawn --project runs
```

Sweep point with changed distillation weights (`--label` keeps sweep points apart in the results):

```powershell
python -m weather.train --mode full --fold 0 --seed 0 --epochs 100 --lambda-feat 4 --lambda-cls 1 --lambda-box 2 --temperature 2 --label full-sweep1 --dataset weather/dataset/dawn --project runs
```

All five folds of one mode (resumable; finished runs are skipped):

```powershell
foreach ($fold in 0..4) { python -m weather.train --mode student --fold $fold --seed 0 --epochs 100 --dataset weather/dataset/dawn --project runs }
```

### Plan C: small objects (`smallobject/train.py`)

A variant is features joined by `+`: `baseline`, `p2`, one of `cbam|eca|simam|ca`, `dysample`, `wfusion`, `nwd`. VisDrone is the default data.

Baseline:

```powershell
python -m smallobject.train --variant baseline --seed 0 --epochs 50 --project runs
```

P2 head:

```powershell
python -m smallobject.train --variant p2 --seed 0 --epochs 50 --project runs
```

Everything combined:

```powershell
python -m smallobject.train --variant p2+eca+dysample+wfusion+nwd --seed 0 --epochs 50 --project runs
```

NWD with an explicit constant (pixels) and weight, instead of the constant computed from the training labels:

```powershell
python -m smallobject.train --variant nwd --nwd-constant 12.8 --nwd-weight 0.5 --seed 0 --epochs 50 --project runs
```

Larger input size (the run is named `<variant>@<size>`):

```powershell
python -m smallobject.train --variant baseline --imgsz 960 --batch 8 --seed 0 --epochs 50 --project runs
```

Write the model YAML of a variant and exit:

```powershell
python -m smallobject.train --variant p2+eca --write-config
```

### Plan D: CamoEdge (`camoedge/train.py`, plain PyTorch)

Experiments `E1` to `E5` (the E6 and E7 sweeps use `--edge-weight`, `--gating` and `--encoder`). Trains, evaluates on the test sets, appends results rows and saves the plot. A checkpoint is written every epoch and the same command resumes.

Quick local trial on 100 images:

```powershell
python -m camoedge.train --experiment E5 --subset 100 --epochs 2 --dataset camoedge/dataset/camo --project runs
```

E1, 30 epochs:

```powershell
python -m camoedge.train --experiment E1 --epochs 30 --dataset camoedge/dataset/camo --project runs
```

Full model, then with concatenation gating, an explicit edge-loss weight, or the ResNet-50 encoder (E6 and E7 sweeps):

```powershell
python -m camoedge.train --experiment E5 --epochs 30 --dataset camoedge/dataset/camo --project runs
```

```powershell
python -m camoedge.train --experiment E5 --gating concat --epochs 30 --dataset camoedge/dataset/camo --project runs
```

```powershell
python -m camoedge.train --experiment E5 --edge-weight 0.5 --epochs 30 --dataset camoedge/dataset/camo --project runs
```

```powershell
python -m camoedge.train --experiment E5 --encoder resnet50 --epochs 30 --dataset camoedge/dataset/camo --project runs
```

Train only, evaluate later (the plot then shows the loss curves only):

```powershell
python -m camoedge.train --experiment E1 --epochs 30 --skip-evaluation --dataset camoedge/dataset/camo --project runs
```

## All plans in one go and the Windows scheduled task

`scripts/trainAll.py` runs the Phase 1 model of every plan one after the other, each in its own process, so a crash in one plan does not stop
the others: Plan B teacher (batch 8), student and full distillation on fold 0, Plan D E1, Plan C baseline, then Plan A baseline, all seed 0.
A missing DAWN, Brackish or COD10K/CAMO copy is downloaded first (VisDrone is downloaded by Ultralytics). The output stays visible in the
terminal and is copied to `runs/epochs<N>/logs/`, which also gets a summary table (`<date>_summary.md`) with the minutes per job and per epoch.
Each epoch budget has its own project folder (`runs/epochs1`, `runs/epochs50`), because a run folder made for another number of epochs is refused.
Re-running the same command resumes unfinished runs and skips finished ones. On Windows it keeps the computer from idle sleep while it runs.

One-epoch check of all four plans:

```powershell
python -m scripts.trainAll --epochs 1
```

All four plans for 50 epochs, or only some of them, or just print the commands:

```powershell
python -m scripts.trainAll --epochs 50
```

```powershell
python -m scripts.trainAll --epochs 50 --plans A C
```

```powershell
python -m scripts.trainAll --epochs 50 --dry-run
```

Windows scheduled task: registers the task `ObjectDetectionTraining` for the current user and starts it now; it opens a PowerShell window in the
repository root that runs `python -m scripts.trainAll --epochs 50` with the `.venv` Python and stays open with the summary at the end. It only runs
while you are logged on (that is what makes the window visible), may run on battery and has no time limit; keep the laptop plugged in and the lid open:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\registerTrainingTask.ps1 -StartNow
```

Start at a set time, or other epochs or plans (`-Plans A,C` is fine):

```powershell
powershell -ExecutionPolicy Bypass -File scripts\registerTrainingTask.ps1 -At "2026-10-23 01:00" -Epochs 50 -Plans A,C
```

See whether it is running, and the latest summary:

```powershell
Get-ScheduledTask -TaskName ObjectDetectionTraining | Get-ScheduledTaskInfo
```

```powershell
Get-ChildItem runs/epochs50/logs/*_summary.md | Sort-Object LastWriteTime | Select-Object -Last 1 | Get-Content
```

Stop it, or remove it:

```powershell
Stop-ScheduledTask -TaskName ObjectDetectionTraining
```

```powershell
powershell -ExecutionPolicy Bypass -File scripts\registerTrainingTask.ps1 -Remove
```

## Results and plots

Every training command saves a detailed Matplotlib figure to `results/plots/<task>/<task>_<run>_<plot kind>_<YYYY-MM-DD_HH-MM-SS>.png`, for example
`results/plots/underwater/underwater_baseline-seed0_training-results_2026-10-18_01-55-03.png`. This folder is tracked by git (it is not ignored), so nothing has to be copied
by hand before a commit. For a detection run (Plans A, B and C) Ultralytics' normalised confusion matrix is copied next to it as `..._confusion-matrix-normalised_...png`.
The plot shows box, class and other losses (training solid, validation dashed), validation mAP with the best epoch marked, precision and recall, the learning rate, per-class
scores (or COCO AP by object size for Plan C), and a text summary of the run. Plan B also saves `..._by-weather_...png`; Plan D shows test-set scores, MAE and per-super-class
scores. A plotting problem prints a warning and never stops a run. Every re-run or redraw adds a new file with a new date and time: delete the older ones you do not want
to commit. On Colab the folder is inside the cloned repository, so copy `results/plots/` to Drive before the session ends, as for `results/*.csv`.

Redraw the plot of a finished detection run (training curves from its `results.csv`; per-class bars need the evaluation, which runs inside the training command):

```powershell
python -m common.plotResults --task underwater --run runs/underwater/baseline-seed0
```

Redraw the plot of a CamoEdge run:

```powershell
python -m common.plotResults --task camoedge --checkpoint runs/camoedge/E1-seed0/last.pt
```

Save the PNG somewhere else instead of `results/plots/<task>/`:

```powershell
python -m common.plotResults --task underwater --run runs/underwater/baseline-seed0 --output plots
```

Summary table of all results (mean ± std over seeds) as markdown:

```powershell
python -m common.summariseResults
```

```powershell
python -m common.summariseResults --folder results --out results/summary.md
```

Show a results file in the terminal:

```powershell
Get-Content results/underwater.csv
```

## Evaluate, speed and export

Speed protocol (the only source of speed numbers; NMS-free `end2end` mode by default):

```powershell
python -m common.benchmark --weights runs/weights/underwater-baseline-seed0.pt --mode end2end
```

```powershell
python -m common.benchmark --weights runs/weights/underwater-baseline-seed0.pt --mode nms --image sample.jpg
```

Plan B, accuracy per weather type on a held-out fold:

```powershell
python -m weather.evalByWeather --weights runs/weather/student-fold0-seed0/weights/last.pt --dataset weather/dataset/dawn --fold 0
```

Plan B, INT8 export (works) and pruning (does not finish yet, see the header of `weather/prune.py`):

```powershell
python -m weather.quantise --weights runs/weather/student-fold0-seed0/weights/last.pt --data weather/dataset/dawn/dawn-fold0.yaml
```

```powershell
python -m weather.prune --weights runs/weather/student-fold0-seed0/weights/last.pt --output pruned.pt --ratio 0.4
```

Plan C, COCO-style AP by object size, and the objects-per-image statistics of VisDrone:

```powershell
python -m smallobject.evalCoco --weights runs/smallobject/baseline-seed0/weights/best.pt --data VisDrone.yaml --split val
```

```powershell
python -m smallobject.evalCoco --weights runs/smallobject/baseline-seed0/weights/best.pt --data VisDrone.yaml --stats-only
```

Plan D, evaluate a checkpoint on the test sets (appends rows and saves the plot; `--no-log` only prints):

```powershell
python -m camoedge.evaluate --checkpoint runs/camoedge/E1-seed0/last.pt --dataset camoedge/dataset/camo
```

```powershell
python -m camoedge.evaluate --checkpoint runs/camoedge/E1-seed0/last.pt --dataset camoedge/dataset/camo --no-log
```

Validate any Ultralytics weights with the Ultralytics command line (accuracy on the same NMS-free head as the protocol):

```powershell
yolo val model=runs/underwater/baseline-seed0/weights/best.pt data=underwater/dataset/brackish/dataset.yaml imgsz=640 nms=False
```

## Demo

Gradio app with one tab per task (opens at http://127.0.0.1:7860):

```powershell
python -m demo.app
```

```powershell
python -m demo.app --port 7861
```

Temporary public link (shares the app: do not use this while the repository and datasets are private):

```powershell
python -m demo.app --share
```

Live webcam demo:

```powershell
python demo/webcam.py
```

## Tests

Fast tests (default, about a minute):

```powershell
python -m pytest -q
```

Slow tests: real one-epoch training on `coco8`, weight downloads (several minutes):

```powershell
python -m pytest -m slow -q
```

One file, or the plotting and results tests only:

```powershell
python -m pytest tests/test_plotResults.py tests/test_results.py -q
```

## Troubleshooting

Run folder of the last training and its files:

```powershell
Get-ChildItem runs/underwater | Sort-Object LastWriteTime -Descending | Select-Object -First 1 | Get-ChildItem -Recurse -Depth 1
```

Delete one trial run so its name can be used again (check the path first; this cannot be undone):

```powershell
Remove-Item -Recurse -Force runs/underwater/epoch1-check-seed0
```

The log line "Transferred N/M items from pretrained weights" must not collapse after inserting a layer; read it at the start of every attention or P2 run.

Dataloader hangs or is slow on Windows:

```powershell
python -m underwater.train --variant baseline --seed 0 --epochs 1 --label trial --workers 0 --data underwater/dataset/brackish/dataset.yaml --project runs
```

CUDA out of memory:

```powershell
python -m underwater.train --variant baseline --seed 0 --epochs 1 --label trial --batch 8 --data underwater/dataset/brackish/dataset.yaml --project runs
```

"... holds a run started with 20 epochs, not 50": the run folder belongs to a run with another epoch count (for example your 20-epoch
`runs/underwater/baseline-seed0`). Resuming or skipping it would report that run under the new epoch count, so the scripts stop. Use another
project folder, which keeps both runs:

```powershell
python -m underwater.train --variant baseline --seed 0 --epochs 50 --data underwater/dataset/brackish/dataset.yaml --project runs/epochs50
```

A training process ends with exit code -9 or "Killed" and no Python error: the operating system ran out of memory. Lower `--batch` (the Plan B
teacher, YOLO26m, needed 13.9 GB at batch 16 on a CPU; `scripts/trainAll.py` trains it at batch 8).
