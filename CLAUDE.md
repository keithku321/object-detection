# CLAUDE.md

## What this project is
A final-year project (Hong Kong Metropolitan University, Computer Science, 2026) by a team of four: **robust object detection under challenging visual conditions**. Each team member owns one subtask, and the four are merged into one repo and one demo app:

| Plan | Subtask | Task type | Folder | Plan file |
|---|---|---|---|---|
| A | Underwater detection: attention-enhanced lightweight YOLO, real-time (≥ 30 FPS) | Detection (Ultralytics YOLO26) | `underwater/` | `plans/01-underwater.md` |
| B | WeatherDistill: knowledge distillation, YOLO26m teacher → YOLO26n student, on DAWN | Detection (Ultralytics YOLO26, custom trainer) | `weather/` | `plans/02-weather-distill.md` |
| C | Small objects: multi-scale fusion + attention, P2 head, NWD loss, on VisDrone | Detection (Ultralytics YOLO26, custom neck/loss) | `smallobject/` | `plans/03-small-object.md` |
| D | CamoEdge: boundary-guided camouflaged object segmentation, on COD10K/CAMO | Segmentation (plain PyTorch, **not** Ultralytics) | `camoedge/` | `plans/04-camoedge.md` |

Everything is planned in `plans/`. **Read the relevant plan (A, B, C or D) before changing anything.** Plans are always referred to by letter; the number prefix in a file name only sorts the folder.

Cross-cutting documents, which are not plans:
- `plans/00-main-plan.md`: phase checklist, decisions, storage and Colab strategy, evaluation protocol
- `plans/05-team-workflow.md`: owners, stewards of shared files, repo rules
- `plans/decisions.md`: decisions log, including any fallback from YOLO26

## Current state
Only the starter files exist: `YOLO26 Occlusion.ipynb` (Colab notebook), `webcam.py` (live webcam demo), an empty `Object_Detection.py`, and the Visual Studio project files. The task folders, `common/`, `demo/`, `results/` and `tests/` described in the plans are created as the plans are carried out. The occlusion example inside the notebook is not one of the four subtasks.

## The notebook is a template
`YOLO26 Occlusion.ipynb` is the starting point for every Colab notebook in this project. You may modify it to implement the plans.

- **Per-task notebooks:** copy it to `<task>/<task>_colab.ipynb` (for example `underwater/underwater_colab.ipynb`) and change the copy. Edit the template itself only for improvements that every task benefits from.
- **Keep its structure:** install and GPU check → configuration cell → mount Drive and unpack the dataset → helper functions → dataset YAML and sanity checks → baseline → train (results written to Drive) → resume → evaluate → look at predictions → export. Keep the Drive resume cell: free Colab sessions end unexpectedly.
- **Replace the example-specific parts:** dataset zip and folder names, class names, model name, run names, occlusion-level evaluation. Dataset acquisition should go through `common/datasets.py` once it exists.
- **Keep notebooks thin:** as modules appear in `common/` and the task folders, move logic there and import it after `git clone` in the first cells. Notebooks configure and run; modules hold the code so it can be tested locally.
- **Colab-only code** (`google.colab`, `!pip`, `/content` paths) stays in notebooks, never in modules.
- **Before committing a notebook, strip its outputs.** Never put tokens in a notebook; use Colab Secrets.

## Conventions (match the existing code)
- Python, camelCase names for variables and functions, descriptive and unabbreviated (`confidenceThreshold`, `writeDatasetYaml`).
- A one-line comment above every function and every configuration constant. Configuration constants at the top of the file.
- `pathlib.Path` for paths; the development machines run Windows, Colab runs Linux.
- Add each new `.py` file to `Object Detection.pyproj` (Visual Studio lists files explicitly).
- Keep changes inside the owning task's folder. Shared files (`common/`, `requirements.txt`, `datasets/manifest.json`, `demo/app.py`) change only as described in `plans/05-team-workflow.md`.

## Technical rules
- **Model family:** YOLO26 by default. Fall back to YOLO11 or YOLOv8 only under the fallback rule in `plans/00-main-plan.md`, and record it in `plans/decisions.md`.
- **One pinned Ultralytics version** (see `requirements.txt`; initial pin `8.4.175`). Do not upgrade it casually; custom trainers, losses and layers depend on its internals.
- **YOLO26 specifics that affect code:** no DFL (`reg_max = 1`); two heads in training (`one2many` and `one2one`); NMS-free inference capped at 300 detections; unknown custom layers are built from their YAML arguments only, with no width scaling. Details are in the plans.
- **Python environment:** use a Python 3.11 or 3.12 virtual environment. The machine default (3.14) may lack PyTorch support.
- **Fair comparison:** baseline and variants share data split, image size, epochs and augmentation. Exploration may use one seed; every reported number uses 3 seeds (mean ± std).
- **Results:** every run appends a row to `results/<task>.csv` (task, model, variant, seed, metrics, GPU, git commit, Ultralytics version). Speed numbers come only from `common/benchmark.py`.
- **Check pretrained loading:** after inserting a layer into a model YAML, confirm the log line "Transferred N/M items from pretrained weights" did not collapse.

## Data and licensing
- Never commit datasets, weights, `*.zip`, `*.onnx` or `runs/` (already in `.gitignore`), and never commit tokens or Drive links with edit access.
- Datasets are used for educational purposes. **Do not republish third-party datasets**; the repo ships download scripts and SHA-256 checksums (`datasets/manifest.json`), not images. Terms per dataset are in `plans/00-main-plan.md`.
- Storage: code on GitHub; processed datasets on a Hugging Face dataset repo (private unless the license allows); checkpoints on each member's Google Drive; final small weights on a GitHub release or Hugging Face.
- Ultralytics is AGPL-3.0 while this repo is MIT: do not make the repo or the demo public without the team and supervisor deciding first.

## Working with Colab
Training runs on free Colab (T4 typically, sessions that end after a few hours) and on local GPUs. Make every long run resumable, write checkpoints to Drive every epoch, and log the GPU model in every results row.

## Do not
- Do not change the model family, the dataset, or the evaluation protocol to make a number look better. If a result is negative, report it.
- Do not patch or copy a shared file (`common/` and the like) to work around a problem inside one task's folder; propose the change as described in `plans/05-team-workflow.md`.
