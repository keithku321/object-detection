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
A first draft of everything the plans list has been written (2026-10-11). It is scaffolding to work from, not finished research: **no model has been trained on a real dataset yet, nothing has run on Colab or a GPU, and `results/` is still empty.** The plans in `plans/` are unchanged.

What has been checked, on Windows with Python 3.14 and the CPU build of PyTorch 2.11.0:
- `pytest` passes 62 fast tests; `pytest -m slow` passes 15 tests that run real one-epoch Ultralytics training on the 8-image `coco8` set (custom layers, pretrained transfer through the layer map, the combined Plan C model, NWD, distillation, the speed protocol).
- Every task's train, resume, evaluate and log path ran once on tiny made-up data, and the demo app starts and serves its four tabs.

What has **not** been checked: the four notebooks have never been run on Colab; no real dataset (Brackish, DAWN, VisDrone, COD10K, CAMO) has been downloaded or trained on; nothing has run on a GPU, so the AMP and FP16 paths and TensorRT export are untested; `weather/prune.py` does not finish yet (see its header).

Starter files: `YOLO26 Occlusion.ipynb` (the notebook template, untouched), `demo/webcam.py` (live webcam demo, moved there from the repository root), `Object_Detection.py` (Visual Studio startup file; it now starts the demo app) and the Visual Studio project files.

| Area | Files | What it is |
|---|---|---|
| Environment | `requirements.txt`, `requirements-local.txt`, `pytest.ini` | Pinned shared requirements (no PyTorch, so Colab keeps its own); the local-PC variant adds the CUDA build of PyTorch and the Python 3.14 note |
| Shared code | `common/datasets.py`, `results.py`, `summariseResults.py`, `benchmark.py`, `modules.py`, `registerModules.py`, `training.py`, `demoSupport.py`, `environment.py` | Manifest-driven download and checks; results log and mean ± std summary; speed protocol; custom layers; YAML and pretrained-transfer tools; the one train, resume, evaluate and log loop used by Plans A, B and C; demo building blocks |
| Data | `datasets/manifest.json` | One entry per dataset with source, licence notes and instructions. **Download locations and SHA-256 values are still empty** and must be filled in by the owners |
| Plan A | `underwater/train.py`, `enhance.py`, `degrade.py`, `demoTab.py`, `underwater_colab.ipynb` | Variants `baseline`, `yolo11`, `yolov8`, `<block>-<placement>` (for example `eca-a`); model YAMLs are generated into `underwater/configs/` |
| Plan B | `weather/prepareDawn.py`, `distillTrainer.py`, `train.py`, `evalByWeather.py`, `prune.py`, `quantise.py`, `demoTab.py`, `weather_colab.ipynb` | 5-fold preparation, distillation on top of Ultralytics' built-in feature, per-weather evaluation. Phase 6 extras: `quantise.py` produced a working INT8 model in a smoke test; `prune.py` is not working yet |
| Plan C | `smallobject/train.py`, `evalCoco.py`, `nwdLoss.py`, `demoTab.py`, `smallobject_colab.ipynb` | Variants are features joined by `+` (for example `p2+eca+nwd`); model YAMLs are generated into `smallobject/configs/` |
| Plan D | `camoedge/dataset.py`, `model.py`, `losses.py`, `metrics.py`, `train.py`, `evaluate.py`, `demoTab.py`, `camoedge_colab.ipynb` | Experiments E1 to E5 from one network class; plain PyTorch |
| Demo and tests | `demo/app.py`, `tests/` | Gradio app with one tab per task; fast tests run by default, tests that download weights or train an epoch are marked `slow` (`pytest -m slow`) |

Not created yet: `results/*.csv` (written by the first run), `common/splits/` (written by `weather/prepareDawn.py`), trained weights, and the Phase 0 entries in `plans/decisions.md` (the findings below are proposals for the team to record there).

### Findings from reading Ultralytics 8.4.175 that differ from the plans
The team should confirm these and record them in `plans/decisions.md`; the code already follows them.
- **Validation and prediction are not NMS-free by default.** They use the one-to-many head with external NMS unless `nms=False` is passed (the default `nms=None` means external NMS). `common/training.py` passes `nms=False` to every train, val and predict call (`endToEndArguments`), so accuracy is measured on the same NMS-free head that the speed protocol times, and every results row notes the head used. Changing this is a protocol decision for the whole team.
- **Knowledge distillation is built in** (`model.train(distill_model="teacher.pt", dis=6.0)`): hooked neck features, a projector, a teacher-confidence-weighted L2 loss, and a final `best.pt` that contains only the student. `weather/distillTrainer.py` subclasses it and adds only the class and box terms, so Plan B needs much less custom code than the plan assumed.
- **`CBAM` is not usable from a YAML as shipped**: it exists in `ultralytics.nn.modules` but `tasks.py` does not import it, and its channel argument is not scaled with the model width. `common/registerModules.py` registers it and fills in the real channel count.
- **Custom layers:** the approach chosen for the Phase 0 spike is lazy channel inference (layers build on the first forward pass) plus an explicit layer map for pretrained transfer after insertions (`insertLayersIntoYaml`, `loadPretrainedWithLayerMap`). A multi-input custom layer cannot be written as a YAML row, so the weighted Concat is swapped in after the model is built.
- **Small-object evaluation:** `val(save_json=True)` already reports `mAP_small` for custom datasets, but with COCO's limit of 100 detections per image; `smallobject/evalCoco.py` raises the limit. The validator also raises `max_det` to the densest image by itself. The metrics object no longer has a `save_dir` attribute.
- **`half=` and `end2end=` are deprecated** in favour of `quantize=16` and `nms=`.

### Things to check that are not code
- **`LICENSE.txt` is the GNU AGPL v3** since commit `986de2a` (it was MIT before). The sentences below and in the plans that call this repository MIT are out of date; the team and supervisor should confirm which licence is intended.
- The template notebook installs Ultralytics with `pip install -U ultralytics`, which ignores the pin. The four task notebooks install from `requirements.txt` instead.

The occlusion example inside the template notebook is not one of the four subtasks.

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
- British English in file names, function and variable names, and comments (`summariseResults.py`, `normaliseFeatures`, `greyWorldWhiteBalance`, `xCentre`, "licence" as a noun). Names that belong to a library keep the library's spelling (`quantize_static`, `ColorJitter`, `torch.cuda.synchronize`, Ultralytics arguments such as `quantize=16`, and the checkpoint key `"optimizer"`). The field name `license` in `datasets/manifest.json` follows the column name used in the plans.
- `pathlib.Path` for paths; the development machines run Windows, Colab runs Linux.
- Add each new `.py` file to `Object Detection.pyproj` (Visual Studio lists files explicitly).
- Keep changes inside the owning task's folder. Shared files (`common/`, `requirements.txt`, `datasets/manifest.json`, `demo/app.py`) change only as described in `plans/05-team-workflow.md`.

## Technical rules
- **Model family:** YOLO26 by default. Fall back to YOLO11 or YOLOv8 only under the fallback rule in `plans/00-main-plan.md`, and record it in `plans/decisions.md`.
- **One pinned Ultralytics version** (see `requirements.txt`; initial pin `8.4.175`). Do not upgrade it casually; custom trainers, losses and layers depend on its internals.
- **YOLO26 specifics that affect code:** no DFL (`reg_max = 1`); two heads in training (`one2many` and `one2one`); NMS-free inference capped at 300 detections; unknown custom layers are built from their YAML arguments only, with no width scaling. Details are in the plans.
- **Python environment:** use a virtual environment. Python 3.12 is the reference, because Colab runs it. Python 3.14 also works with PyTorch 2.11 (checked with the CPU build; CUDA wheels exist for it), except that `pysodmetrics` needs the manual step described in `requirements-local.txt`. Install with `requirements.txt` on Colab and `requirements-local.txt` on a PC with an NVIDIA GPU.
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
