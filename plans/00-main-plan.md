# Main Plan: Robust Object Detection under Challenging Conditions

Team of four, one proposal each, merged into one repo and one demo. This plan is a **phase checklist**: the work is grouped by technique, from easiest to hardest, and every owner works through the same phases in parallel. Each phase ends with a **sync gate** where the team reviews and merges together. Week estimates were dropped on purpose: with four people working in parallel a single sequential schedule no longer applies. Set dates yourselves once the deadlines are known.

| Doc | Content |
|---|---|
| [01-underwater.md](01-underwater.md) | Plan A: attention-enhanced lightweight YOLO for underwater detection |
| [02-weather-distill.md](02-weather-distill.md) | Plan B: WeatherDistill, knowledge distillation on DAWN |
| [03-small-object.md](03-small-object.md) | Plan C: multi-scale fusion + attention for small objects |
| [04-camoedge.md](04-camoedge.md) | Plan D: CamoEdge, boundary-guided camouflaged object segmentation (not YOLO) |
| [05-team-workflow.md](05-team-workflow.md) | Roles, branches, merge rules, shared interfaces |
| [decisions.md](decisions.md) | Log of decisions and any fallback from YOLO26 |

---

## 1. What is in the repo today

| File | What it is | Reuse |
|---|---|---|
| `YOLO26 Occlusion.ipynb` | Colab notebook for a fifth, occlusion-themed project. Mounts Drive, copies a zip to local disk, writes `dataset.yaml`, checks labels, trains `yolo26n.pt`, resumes, evaluates, exports. | Template for every task notebook. |
| `webcam.py` | Live webcam demo that loads `best.pt`. | Becomes the webcam path of the demo. |
| `Object_Detection.py` | Empty Visual Studio startup stub. | Replace with the demo entry point or delete. |
| `Object Detection.pyproj` | Visual Studio project; files listed explicitly. | Add new `.py` files when created. |
| `.gitignore` | Excludes `datasets/`, `runs/`, `*.pt`, `*.onnx`, `*.zip`. | Data and weights stay out of git. |
| `LICENSE.txt` | MIT. | See the AGPL note in section 5. |

No shared code, tests, dependency pins or results tracking exist yet. The occlusion project is not one of the four proposals and stays untouched.

**Coding conventions to keep** (from the existing code): camelCase names, descriptive and unabbreviated; a one-line comment above every function and constant; configuration constants at the top of a file; `pathlib` for paths.

---

## 2. Decisions

| # | Decision | Detail |
|---|---|---|
| D1 | **YOLO26 is the default for Plans A, B and C** | Fall back only on a *hard technical issue* (rules below). Reference rows with YOLO11 or YOLOv8 are still welcome because the proposals mention them. |
| D2 | **Free Colab** (one account per person), plus local GPUs | Every run must be resumable. Small-object compute is trimmed accordingly (Plan C). |
| D3 | **VisDrone allowed for educational use** | See the dataset terms in section 4. |
| D4 | **CamoEdge stays segmentation** | The only plan that does not use Ultralytics. |
| D5 | **One pinned Ultralytics version for everyone** | Start with `ultralytics==8.4.175` (what pip resolves today). A pin change needs approval from at least two other owners. |
| D6 | Accuracy targets set at the Phase 1 gate | The proposals fix only one number (≥ 30 FPS). |

### What YOLO26 changes (checked in the Ultralytics 8.4.175 source)
- **No DFL** (`reg_max: 1`). Boxes are four direct distances per anchor and the box loss uses L1 instead of DFL. WeatherDistill's "KL on the regression head" cannot be applied literally; Plan B replaces it with a box-level distillation loss. This is a method adaptation, not a reason to downgrade.
- **Two heads in training:** a one-to-many head (feeds the backbone) and a one-to-one head (NMS-free inference; its input features are detached). The loss is a weighted sum that shifts from one-to-many towards one-to-one during training.
- **The training output already contains the neck feature maps** (`feats`) next to boxes and class logits, so feature distillation does not need forward hooks.
- **STAL (small-target-aware assignment) and ProgLoss are already in the baseline.** The small-object baseline is stronger than a YOLOv8 one, so expect smaller gains from additions (Plan C).
- **P2 variant exists as a YAML only, with no pretrained weights.** The backbone still loads pretrained weights; the head does not.
- **End-to-end inference is capped at 300 detections per image.** Dense VisDrone scenes need checking (Plan C).
- **Custom layers:** layer names are looked up in the `ultralytics.nn.tasks` namespace, so a custom layer can be registered without editing the library. But a layer unknown to the parser is built with only its YAML arguments and keeps the input channel count, so the width multiplier is *not* applied to its arguments. Each owner must handle this (Phase 0 spike).

### Fallback rule ("hard technical issue")
A fallback is allowed only when **all** hold: (1) the problem is reproduced on the pinned version with a minimal example; (2) at most three working days were spent on a workaround; (3) the owner posted the finding in `decisions.md` and at least two other owners approved it. Typical qualifying cases:
- inserting a custom layer makes pretrained transfer collapse and no wrapper approach fixes it;
- the λ = 0 distillation check does not reproduce the baseline;
- a bug in the pinned release that blocks training or validation.

The fallback ladder is YOLO26 → YOLO11 → YOLOv8 (all share the Ultralytics API). It is **per plan**, not project-wide. Not qualifying: "YOLOv8 has a feature the proposal mentions" (for example DFL-KL): add that as an optional extra row on YOLO11 and keep YOLO26 as the main result.

---

## 3. Compute on free Colab

- **Expect:** no guaranteed GPU (usually a T4), sessions that end after a few hours or when idle, and usage limits that vary. Plan for resumable chunks, not one long run.
- **Local GPUs matter.** One team machine has an RTX 4060 laptop GPU (8 GB); everyone lists their hardware in the table in `05-team-workflow.md`. Debug and run small experiments locally; use Colab for the long sweeps.
- **Second free source (optional):** Kaggle notebooks also offer a free GPU with a weekly quota; check the current quota before relying on it.
- **Rules:** save to Drive every epoch; start every long run with `resume`; no run should need more than one session to finish *without* a checkpoint; log the GPU model in every result row.
- **Phase 1 compute gate:** each owner times one epoch of every baseline, multiplies out `runs × seeds × epochs × seconds`, and trims the plan (smaller image size, fewer epochs, nano models, one seed for exploration) before Phase 2. Final reported numbers still use 3 seeds.

---

## 4. Colab, storage and dataset hosting

### Where each kind of thing lives
| Thing | Where | Why |
|---|---|---|
| Code, configs, notebooks (outputs stripped), plans, small split lists | **GitHub** (this repo) | Versioned; Colab clones it and imports its modules. |
| Processed datasets (YOLO-format zips) | **Hugging Face dataset repo** (private unless the license allows public) | Public repos allow about 1 TB, one file up to 200 GB (under 50 GB recommended), under 10k files per folder, so ship zips. Versioned, no per-file download lock. |
| Checkpoints and runs while training | **Each member's own Google Drive** (`MyDrive/yoloProjectResults/<task>/<run>`) | Survives disconnects; the existing notebook already does this. |
| Final small weights | **GitHub release** or a Hugging Face model repo | The demo downloads them. |
| Archive with a DOI (optional) | **Zenodo** | Only for data you may publish. |

**Do not use a public Google Drive link as the main distribution channel.** Google locks heavily downloaded shared files for about 24 hours, and `gdown` cannot bypass it. Mounted Drive is also slow for thousands of small images, so notebooks always copy one zip to `/content` and unzip locally.

### Dataset terms: educational use
Using these datasets for a university final-year project is consistent with their stated academic, non-commercial terms. The restrictions that matter for the team are **redistribution and commercial use**, so the repo ships download scripts and checksums, not images.

| Dataset | Terms found | Team rule |
|---|---|---|
| **VisDrone** | Copyright AISKYEYE (Tianjin University). Secondary sources describe CC BY-NC-SA 3.0 (attribution, non-commercial, share-alike); I could not retrieve the official license text. | Allowed for the educational project. Attribute (Zhu et al., TPAMI 2021). Do not republish. Confirm the license file in the official VisDrone GitHub repo and note it in `datasets/manifest.json`. Ultralytics downloads it itself. |
| DAWN | Mendeley page: all rights reserved, commercial use and distribution prohibited; images came from web searches. | Educational use fine; no public mirror. |
| Brackish | Roboflow mirror lists CC BY 4.0 (confirm at vap.aau.dk). | May be mirrored with attribution if confirmed. |
| URPC | Official download closed after the contest; many near-duplicate images. | Use a Roboflow copy or the cleaned DUO benchmark; deduplicate; no mirror. |
| COD10K / CAMO | Research distribution by the authors. | Check each README; no mirror. |

This is a practical reading of the published terms, not legal advice. If the supervisor or the university requires formal clearance, ask them.

### Colab rules for every notebook
1. First cells: `git clone` the repo, `pip install -r requirements.txt`, print the GPU.
2. Mount Drive for results only; download the dataset zip to `/content`, check its SHA-256 against `datasets/manifest.json`, unzip locally.
3. Tokens (Hugging Face, Kaggle) go in **Colab Secrets**, never in a notebook or the repo.
4. Training writes to Drive and supports `resume`.
5. Notebooks are thin; logic lives in `.py` modules so it can be tested locally.

---

## 5. Repo layout and licensing

```
plans/                  this folder
common/                 shared code (each file has a steward, see 05-team-workflow.md):
                        datasets.py, benchmark.py, results.py, modules.py,
                        registerModules.py, splits/
underwater/             Plan A      weather/       Plan B
smallobject/            Plan C      camoedge/      Plan D
  each task folder also holds: configs, train.py, <task>_colab.ipynb, demoTab.py
demo/app.py             Gradio app that mounts every task's demoTab
datasets/manifest.json  source, license, SHA-256 per dataset
results/<task>.csv      one row per run (committed; weights are not)
requirements.txt        pinned versions
tests/                  smoke and metric tests
```

**Licensing:** Ultralytics (and YOLO26) is AGPL-3.0 or an enterprise license; this repo's license is MIT. For a private university project this is not a problem. If you publish the repo or serve the Gradio demo to the public, AGPL obligations can apply to code built on Ultralytics. Decide with your supervisor before making anything public.

---

## 6. Shared evaluation protocol (defined once)

- **Fair comparison:** baseline and variants share split, image size, epochs, augmentation and pretrained start; only the change under test differs.
- **Seeds:** exploration uses one seed; every reported number uses **3 seeds**, as mean ± std. Differences under about one mAP point on small test sets are noise.
- **Detection metrics:** mAP@0.5, mAP@0.5:0.95, precision, recall, parameters, GFLOPs.
- **Speed protocol** (`common/benchmark.py`): batch 1, FP16, 640 px, 50 warm-up and 300 timed iterations, GPU name printed. YOLO26's inference is NMS-free by default, so report **end-to-end latency** (one-to-one head, no NMS) and, where the one-to-many path is used, the NMS-inclusive figure.
- **Results log:** each run appends a row (task, model, variant, seed, metrics, GPU, git commit, ultralytics version) to `results/<task>.csv`.

---

## 7. Phase checklist

Owner tags: **A** underwater, **B** weather, **C** small object, **D** CamoEdge. Shared files have a steward from this list (see `05-team-workflow.md`). Step IDs (U1, W3, S2, E4 …; E = CamoEdge steps) refer to the experiment tables in each plan. Sizes: S = a few days, M = about a week, L = more than a week; they are rough and per owner.

### Phase 0: Foundations — gate G0
- [ ] **A (M, steward)** `common/datasets.py` (download, convert, checksum) and `datasets/manifest.json` with a license column.
- [ ] **B (S, steward)** `common/results.py` and a small script that merges `results/*.csv` into one summary table.
- [ ] **D (S, steward)** `common/benchmark.py` (speed protocol below).
- [ ] **B (S, steward)** Keep `YOLO26 Occlusion.ipynb` as the notebook template (see `CLAUDE.md`); Hugging Face repo; Colab Secrets instructions.
- [ ] **A, B, C (S each) YOLO26 spike**, results written to `decisions.md`:
  - [ ] `yolo26n` trains 1 epoch on a tiny dataset locally and on Colab, and appends a results row;
  - [ ] **A, C:** register a channel-preserving custom layer, insert it in a YAML, and compare the log line "Transferred N/M items from pretrained weights" with and without it; decide the approach (explicit channel arguments per scale, a wrapper applied after loading weights, or lazy channel inference);
  - [ ] **B:** subclass the end-to-end loss, add a dummy term, and confirm the training output exposes the neck features and both heads;
  - [ ] end-to-end latency measured with `benchmark.py`.
- [ ] **All** Read your plan, fill in the hardware table, confirm scope.
- **G0 exit:** spike results logged; fallback decisions (if any) approved; everyone can run a 1-epoch job and log a row.

### Phase 1: Data and baselines — gate G1
- [ ] **A (M)** Brackish → YOLO format; baselines `yolo26n/s` plus YOLO11n and YOLOv8n reference rows (U0). URPC/DUO conversion and deduplication.
- [ ] **B (M)** DAWN download and conversion; stratified 5-fold splits; per-weather evaluation script; student `yolo26n` and teacher `yolo26m` baselines (W0, W1).
- [ ] **C (M)** VisDrone via Ultralytics; check the objects-per-image statistics against the 300-detection cap; COCO-style AP_small evaluation script; `yolo26n` baseline (S0).
- [ ] **D (M)** COD10K + CAMO loaders; metrics via PySODMetrics with unit tests; plain encoder-decoder baseline (E1); published reference quoted (E0).
- [ ] **All** Time one epoch per baseline and write the compute budget.
- **G1 exit:** a baseline row for every task in `results/`; compute budget agreed; numeric targets set; data code merged.

### Phase 2: Quick wins (config and loss changes only) — gate G2
- [ ] **A (S)** Built-in CBAM variant (U2); classical-enhancement comparison with gray-world + CLAHE (U1).
- [ ] **B (M)** Fine-tune the teacher on all folds; write the distillation trainer skeleton and pass its checks (λ = 0 reproduces the baseline; smoke test).
- [ ] **C (S)** P2-head config (S1); image-size comparison.
- [ ] **D (M)** Weighted BCE + weighted IoU structure loss (E2).
- **G2 exit:** first improvement over baseline in at least some tasks; `common/` additions merged.

### Phase 3: New modules — gate G3
- [ ] **C (steward) + A** Finalise `common/modules.py` (Coordinate Attention, ECA, SimAM, DySample, weighted fusion). A and C use the same registration mechanism from G0.
- [ ] **A (L)** Attention variants and placement ablation (U3, U4).
- [ ] **B (M)** Channel adapters and feature distillation (W3).
- [ ] **C (L)** Attention, DySample and weighted fusion, one at a time (S2, S3, S4).
- [ ] **D (L)** Edge Guidance Branch and Edge-Attention Gating modules; run with the branch as an auxiliary output (E3).
- **G3 exit:** every owner has ablation rows for their modules; `common/modules.py` merged.

### Phase 4: Custom training objectives — gate G4
- [ ] **A (M)** Synthetic turbidity robustness test sets and curves; URPC/DUO second-dataset runs (U6).
- [ ] **B (L)** Head distillation (classification KL + box-level regression distillation), combined with features; λ and temperature sweep on fold 0 (W4, W5, W6).
- [ ] **C (L)** NWD box loss (S5) and the combined model (S6).
- [ ] **D (L)** Edge loss, edge ground truth, λ sweep, full CamoEdge (E4, E5, E6).
- **G4 exit:** each plan has its "proposed method" result and a first ablation.

### Phase 5: Ablations and final multi-seed runs — gate G5
- [ ] **A** Final 3-seed runs, baseline vs best (U7).
- [ ] **B** 5-fold × 3-seed final runs, per-weather tables (W9).
- [ ] **C** Final 3-seed runs; leave-one-out ablation (S7).
- [ ] **D** Encoder comparison (E7), final 3-seed runs, per-category and boundary analysis (E8).
- **G5 exit:** every number planned for the report exists with mean ± std.

### Phase 6: Compression, export and speed — gate G6
- [ ] **B (L)** Structured pruning (about 40%), fine-tune with distillation, **static INT8** quantization (W7, W8).
- [ ] **A, C (S)** Export final models (ONNX, TensorRT FP16) and run the speed protocol; confirm the ≥ 30 FPS requirement for A.
- [ ] **D (S)** Parameters, GFLOPs, FPS of CamoEdge.
- [ ] **D** Edge-hardware statement: the project has no Jetson, so edge claims rest on parameters, GFLOPs and measured latency on available hardware.
- **G6 exit:** an accuracy–size–speed table for every final model.

### Phase 7: Integration, demo and report — final gate
- [ ] **Each owner** `demoTab.py` with `buildTab()` (see `05-team-workflow.md`); upload final weights to the chosen release location.
- [ ] **A (steward)** `demo/app.py` mounting all four tabs; webcam path from `webcam.py`; baseline vs proposed toggle.
- [ ] **All** Failure-case gallery, figures, report sections, presentation.
- **Final exit:** a fresh clone plus `requirements.txt` reproduces the headline tables and starts the demo.

---

## 8. Risks

| Risk | Effect | Mitigation |
|---|---|---|
| Free Colab limits | Lost or delayed runs | Resume everywhere; Phase 1 compute gate; use local GPUs; optional Kaggle |
| YOLO26 is new and its internals move | Custom code breaks | One pinned version; smoke tests; fallback rule |
| Custom layers break pretrained loading | Variants look worse for the wrong reason | G0 spike; check "Transferred N/M" in every log |
| Merge conflicts in shared files | Time lost | Ownership rules in `05-team-workflow.md`; per-task results files |
| Four people pinned to different versions | Results not comparable | Single `requirements.txt`, changed only by agreement |
| URPC near-duplicates | Inflated mAP | Deduplicate; consider DUO |
| Tiny DAWN test set | Noisy conclusions | 5-fold cross-validation, 3 seeds |
| VisDrone cost on free Colab | Ablation does not fit | Gate G1; nano model and 640 px for exploration |
| CamoEdge scope | Overruns | Minimum viable scope in Plan D |
| Dataset terms | Cannot publish data | Scripts and checksums only; section 4 |
| AGPL | Constraints on publishing code or demo | Decide with supervisor before going public |
| Uneven shared-code load | Bottleneck on one person | Stewardship is split across all four owners; the owner of the lightest plan carries more of it (see `05-team-workflow.md`) |
