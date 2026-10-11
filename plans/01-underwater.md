# Plan A: Attention-Enhanced Lightweight YOLO26 for Underwater Object Detection

Proposal: *An Attention-Enhanced Lightweight YOLO Framework for Real-Time Underwater Object Detection* (Subtask, supervisor Dr. Wang Weiming Wyman). Main plan: [00-main-plan.md](00-main-plan.md). Team workflow: [05-team-workflow.md](05-team-workflow.md). Difficulty rank: **1 of 4**. Owner: _TBD_.

## Goal and success criteria
Embed lightweight attention modules in a YOLO26 backbone/neck so it detects underwater objects better than an unmodified YOLO26, **without** a separate image-enhancement stage.

| Criterion | How it is judged |
|---|---|
| Real-time | ≥ 30 FPS under the shared speed protocol (batch 1, FP16, 640 px, end-to-end) on a named GPU |
| Accuracy | mAP@0.5 above the same-size baseline, mean ± std over 3 seeds. Target size set at gate G1 |
| Turbidity robustness | mAP stays higher than baseline as synthetic degradation increases |
| Thesis check | "Classical enhancement + baseline" versus "attention model, no enhancement" |
| Prototype | Image and video inference in the demo app |

## Data
| Dataset | Facts (verified) | Role |
|---|---|---|
| **Brackish** | Aalborg University, Limfjorden (Denmark). Roboflow mirror: 14,674 images, YOLO format, 80/10/10 split, 6 classes (crab, fish, jellyfish, shrimp, small_fish, starfish), listed CC BY 4.0. Frames come from video, so neighbouring frames look alike. | **Primary dataset: start here** |
| **URPC** | 4 classes (holothurian, echinus, scallop, starfish). Counts differ by year. Official downloads closed after the contest; many near-duplicate images. | Second dataset. Use a Roboflow copy or the cleaned **DUO** benchmark (arXiv 2106.05681) |

Steps (Phase 1):
1. `common/datasets.py` downloads Brackish, writes `dataset.yaml`, runs the label checks from the existing notebook, records the checksum.
2. URPC: convert VOC XML to YOLO if needed, **deduplicate with perceptual hashes**, confirm no train/val overlap before training.
3. Keep Brackish's original split. Commit file lists under `common/splits/`.
4. Check that Brackish's splits do not put near-identical frames from the same video on both sides.

## Method
- **Baselines (Phase 1):** `yolo26n` and `yolo26s` (main). Reference rows: `yolo11n` and `yolov8n`, since the proposal mentions them.
- **Attention candidates:** CBAM (built into Ultralytics, Phase 2); Coordinate Attention, ECA, SimAM (Phase 3, from `common/modules.py`). SimAM adds no parameters, which suits the "lightweight" claim. YOLO26's backbone already ends with a C2PSA self-attention block, so the new blocks go **earlier** (feeding the neck) and the gain must be shown on top of it.
- **Placement ablation:** (a) after the last block of each of the three backbone stages feeding the neck (layers 4, 6, 8 in the base YAML), (b) in the neck before the head, (c) both.
- **Inserting layers (from the Phase 0 spike):** an unknown layer receives only its YAML arguments and keeps the input channel count, and the width multiplier is not applied to those arguments. Use the approach chosen at G0 (explicit channels per scale, a wrapper applied after loading weights, or lazy channel inference). Inserting a layer in the YAML shifts all later layer indices, so pretrained weights may silently stop matching; check "Transferred N/M items" in every log. If the count collapses, the comparison is invalid.
- **Training:** same recipe as baseline (epochs, image size, augmentation including mosaic); only the attention changes. Tune hyperparameters on the baseline first, then reuse them.

## Experiment matrix
| ID | Model | Change | Phase |
|---|---|---|---|
| U0 | yolo26n/s (+ yolo11n, yolov8n reference) | none | 1 |
| U1 | yolo26n + enhancement | gray-world white balance + CLAHE on train and test images | 2 |
| U2 | yolo26n + CBAM | built-in block | 2 |
| U3 | yolo26n + CA / ECA / SimAM | each block, best placement from U2 | 3 |
| U4 | placement ablation | positions (a), (b), (c) for the best block | 3 |
| U5 | turbidity robustness curves | baseline vs U1 vs best | 4 |
| U6 | repeat on URPC/DUO | second dataset | 4 |
| U7 | final 3-seed runs on both datasets | baseline vs best; also on `yolo26s` | 5 |
| U8 | export + speed | ONNX, TensorRT FP16, ≥ 30 FPS check | 6 |

## Robustness test (turbidity)
Create degraded copies of the **test** set at 5 severity levels (0–4): haze (`I = J·t + A·(1 − t)` with increasing strength), red-channel attenuation (colour cast), Gaussian blur. Plot mAP@0.5 against severity for the baseline, the enhancement pipeline and the attention model. This is the "robustness against turbidity" metric the proposal asks for. Cross-water-body generalisation (the Folkman et al. point) is a stretch goal; the class sets of Brackish and URPC differ, so it would have to be class-agnostic.

## Speed and prototype
- Measure with `common/benchmark.py`. YOLO26 inference is NMS-free by default, so report end-to-end latency. If an attention variant falls below 30 FPS, try fewer insertion points or SimAM before dropping the idea.
- Export the best model to ONNX and TensorRT FP16 for the deployment number.
- Demo tab (`underwater/demoTab.py`): image and video upload, detections with class and confidence, FPS shown; reuses the loop in `webcam.py`.

## Files to create
```
underwater/configs/        model YAMLs per variant
underwater/enhance.py      classical enhancement (U1)
underwater/degrade.py      synthetic turbidity test sets
underwater/train.py        runs one variant, appends to results CSV
underwater/demoTab.py
underwater/underwater_colab.ipynb
```

## Free-Colab notes
Brackish is large (about 14.7k images). `yolo26n` at 640 px is feasible on a T4 but not in one session at 100 epochs; train in resumable chunks with checkpoints on Drive, and time one epoch at gate G1 before choosing the epoch count. Do the Phase 2–3 exploration with a reduced epoch count and one seed.

## Risks
- Brackish frames from the same video can be near-identical across splits; if so, accuracy is inflated.
- Attention gains on a small model can be within seed noise. The 3-seed final runs decide, not single runs.
- URPC availability is the least certain part; fall back to DUO.
- The base model already has C2PSA attention, so gains from extra attention may be smaller than the proposal expects; report that honestly.
