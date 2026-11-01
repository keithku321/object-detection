# Plan C: Small Object Detection — Multi-Scale Fusion + Attention (YOLO26)

Proposal: the "Small Object" subtask in the project deck (enhanced multi-scale feature fusion, lightweight attention, preserved high-resolution features, improved AP_s). Only the deck describes it; there is no separate write-up, so this plan fills the gaps. Main plan: [00-main-plan.md](00-main-plan.md). Team workflow: [05-team-workflow.md](05-team-workflow.md). Difficulty rank: **3 of 4**. Owner: **Ku Sui Ki** (assigned 2026-10-22, see decision D7).

## Goal and success criteria
Raise the small-object AP (AP_s) of a lightweight YOLO26 on a standard small-object benchmark while staying lightweight and fast.

| Criterion | How it is judged |
|---|---|
| AP_s improvement | COCO-style AP_small above baseline, mean ± std over 3 seeds (target size set at gate G1) |
| No collapse elsewhere | mAP@0.5, mAP@0.5:0.95 and AP_medium/large do not drop meaningfully |
| Still lightweight | Parameters, GFLOPs and end-to-end latency against baseline, shared protocol |

## Important: YOLO26 already helps small objects
YOLO26 ships **STAL** (small-target-aware assignment: ground-truth boxes with a side smaller than the stride are enlarged for assignment, so tiny objects keep positive anchors) and **ProgLoss**. The baseline is therefore stronger than the YOLOv8 baseline the proposal assumed. Consequences:
- Expect **smaller gains** from P2 and NWD than the literature reports for older YOLO versions, and say so. A gain that survives against the STAL baseline is the stronger result.
- NWD (S5) overlaps in spirit with STAL; keep it as an experiment, not a promise.
- The P2 YAML exists but has **no pretrained weights**: the backbone loads pretrained, the head does not, so compare against a baseline trained for the same epochs and read the "Transferred N/M" line.

## Data
- **Primary: VisDrone2019-DET** (educational use; see terms in the main plan). 6,471 train / 548 val / 1,610 test-dev images, 10 classes. Ultralytics' `VisDrone.yaml` downloads (about 2 GB) and converts it; allow about 4 GB free disk. Cite Zhu et al., TPAMI 2021. Do not republish.
- Images are large (roughly 2000 × 1500), so image size matters: 640 px shrinks small objects heavily.
- **Optional second dataset:** TinyPerson or AI-TOD, only if time remains.

## Evaluation: AP_s
Ultralytics reports mAP but not AP by object size. Build `smallobject/evalCoco.py`:
1. Predict on the validation set with a very low confidence threshold; save detections.
2. Convert YOLO labels to a COCO ground-truth JSON using box area in **original image pixels**.
3. Run `pycocotools` COCOeval; report AP_small (< 32² px), AP_medium and AP_large.
4. **Raise `maxDets`** above COCO's default of 100.
5. Check that the overall AP matches Ultralytics' mAP@0.5:0.95 within a small tolerance; if not, the converter is wrong.

**Detection cap:** YOLO26's NMS-free (one-to-one) inference returns at most 300 detections per image, and dense VisDrone scenes can contain more objects than that. In Phase 1 compute the objects-per-image statistics; if the cap matters, evaluate the one-to-many + NMS path with a higher `max_det` and report both.

## Method (add one idea at a time)
| Step | Change | Cost |
|---|---|---|
| S1 | **P2 head:** stride-4 detection layer (`yolo26-p2.yaml`; weights not pretrained for the head) | Config only, Phase 2 |
| S2 | **Attention** (CBAM, then CA/ECA/SimAM from `common/modules.py`) in the neck | Phase 3 |
| S3 | **DySample** content-aware upsampling replacing the nearest-neighbour upsampling in the neck | Phase 3, new module |
| S4 | **Weighted multi-scale fusion** (BiFPN-style learnable weights) in place of plain concatenation | Phase 3, new module |
| S5 | **NWD loss:** model each box as a 2D Gaussian; `NWD = exp(−√W₂² / C)`; box loss = (1 − α)·IoU-loss + α·(1 − NWD). YOLO26 uses direct box regression, so wrap the box-loss term | Phase 4, custom loss |
| S6 | Combined model: the best of the above | Phase 4 |

Notes:
- C in NWD relates to typical object size; compute it from the training labels. Try α ∈ {0.3, 0.5}. Using NWD in the label assigner is higher risk; do the loss only first.
- Optional strong baseline that needs no training: sliced inference (SAHI-style tiling). It shows whether the architecture changes beat the cheap alternative; report the speed cost.
- Custom layers: use the mechanism chosen at G0 (see the underwater plan for the details of unknown-layer handling).

## Experiment matrix
| ID | Model | Phase |
|---|---|---|
| S0 | `yolo26n` baseline at 640 (and `yolo26s` if affordable); 1280 for baseline only | 1 |
| S1 | + P2 head | 2 |
| S2 | + attention | 3 |
| S3 | + DySample | 3 |
| S4 | + weighted fusion | 3 |
| S5 | + NWD loss (on S0 and on the best architecture) | 4 |
| S6 | Combined model | 4 |
| S7 | Final 3-seed runs; leave-one-out ablation of the combined model | 5 |
| S8 | Export and speed | 6 |

## Compute on free Colab
This is the heaviest YOLO task, and free Colab will not fit seven variants × 3 seeds × 100 epochs on 6.5k large images. Plan for it:
- **Exploration:** `yolo26n`, 640 px, reduced epochs, one seed. Run the ablation on the local GPU when available.
- **1280 px** only for the baseline and the final model.
- **Gate G1:** time one epoch at each size, compute total GPU hours, then cut variants or epochs before Phase 2.
- Train in resumable chunks with checkpoints on Drive; log the GPU in every result row.
- The final 3-seed comparison is limited to baseline versus best model.

## Files to create
```
smallobject/evalCoco.py     AP_s evaluation
smallobject/nwdLoss.py      NWD box loss and trainer hook
smallobject/configs/        model YAMLs per step
smallobject/train.py
smallobject/demoTab.py
smallobject/smallobject_colab.ipynb
common/modules.py           DySample, weighted fusion (shared with Plan A)
```

## Risks
- Several changes together can cancel out; the one-at-a-time ablation is required.
- STAL may absorb part of the benefit of P2 and NWD; report honestly.
- A custom loss touches Ultralytics internals; pin the version and smoke-test.
- Gains at 1280 px may vanish at 640 px; report the setting used.
