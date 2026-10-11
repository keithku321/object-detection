# Plan D: CamoEdge — Boundary-Guided Camouflaged Object Detection

Proposal: *CamoEdge: Boundary-Guided Feature Fusion for Accurate Camouflaged and Mimetic Object Detection*. Main plan: [00-main-plan.md](00-main-plan.md). Team workflow: [05-team-workflow.md](05-team-workflow.md). Difficulty rank: **4 of 4**. Owner: _TBD (needs someone comfortable with plain PyTorch)_.

**This is the one proposal that is not a YOLO bounding-box task, so the YOLO26 decision does not apply.** It produces a per-pixel mask scored with segmentation metrics, so it uses a custom PyTorch model (PVTv2 or ResNet-50 encoder) and not Ultralytics. It shares only the repo, Colab workflow, results logging and demo with the others. It is not affected by the pinned Ultralytics version.

## Goal and success criteria
Show that an explicit edge branch plus an edge-supervised compound loss improves camouflaged object segmentation over the same network without them.

| Criterion | How it is judged |
|---|---|
| Edge guidance helps | Ablation: S_α, E_φ, F_β^w up and MAE down when the edge branch and edge loss are added |
| Boundary quality specifically | A boundary-focused metric improves (see Metrics) |
| Across categories | Results split by COD10K super-class (for example amphibian, aquatic, flying, terrestrial) |
| Complexity manageable | Parameters, GFLOPs and FPS reported |
| Interactive demo | Gradio tab showing mask, edge map and overlay |

## Data (standard protocol, verified)
- **Train:** COD10K-Tr (3,040) + CAMO-Tr (1,000) = **4,040 images**.
- **Test:** COD10K-Te (2,026) and CAMO-Te (250). NC4K (4,121, test-only) is an optional extra for generalisation.
- COD10K contains 10,000 images in total, but only the 5,066 camouflaged ones (3,040 train + 2,026 test) are used for this task; the rest are background and non-camouflaged images.
- COD10K file names encode the super-class, which allows the per-category analysis; confirm the naming pattern in the downloaded data.
- Check each dataset's README for license terms before storing copies anywhere public (see the main plan).

## Metrics (Phase 1)
- Use the **PySODMetrics** package (`pip install pysodmetrics`, import `py_sod_metrics`) for S-measure (S_α), E-measure (E_φ), weighted F-measure (F_β^w) and MAE rather than re-implementing them. Its maintainers report agreement with the standard MATLAB code, with very small differences on E-measure; note the version used.
- Write unit tests with tiny hand-made masks (perfect prediction → S = 1, MAE = 0; all-wrong prediction → MAE = 1).
- **Boundary metric (recommended extra):** edge pixels are under 5% of the mask, so global scores hide boundary gains. Add a boundary F-score (precision and recall of predicted versus true boundary within a few pixels). This directly tests the proposal's thesis.

## Architecture
1. **Semantic encoder:** PVTv2-B2 (the `timm` library provides PVTv2 variants; confirm the model name) or ResNet-50, ImageNet-pretrained. Take features at strides 4, 8, 16, 32.
2. **Edge Guidance Branch (EGB):** a shallow conv stream on the high-resolution feature (stride 4) plus optionally the raw image gradient; outputs an edge feature map and an edge prediction.
3. **Edge-Attention Gating (EAG):** turn the edge feature into a gate (`σ(conv(edge))`) and inject it into the decoder features, for example `f' = f ⊙ (1 + gate)` so the gate can only emphasise, not erase, features. Alternative: concatenate then convolve (ablation).
4. **Decoder:** multi-scale fusion from deep to shallow (FPN-style), with side outputs at several scales for deep supervision.

## Loss
Total = Σ over outputs [ L_wBCE + L_wIoU ] + λ_e · L_edge

- **Weighted BCE + weighted IoU** (the "structure loss" used in earlier camouflaged and polyp segmentation work): pixel weight `w = 1 + 5·|avgpool(mask) − mask|`, which up-weights boundary and hard pixels.
- **Edge loss:** BCE between the edge prediction and an edge ground truth derived from the mask with Sobel or Canny as the proposal states (thicken to about 2–3 px). Edge pixels are rare, so use a positive-class weight, or add a Dice term; tune λ_e (ablation).

## Experiment matrix
| ID | Setup | Phase |
|---|---|---|
| E0 | Published reference (for example SINet-V2) run or quoted with citation | 1 |
| E1 | Encoder + plain decoder, BCE only (data loader, metrics, training loop) | 1 |
| E2 | E1 + weighted BCE + weighted IoU | 2 |
| E3 | E2 + EGB (and the EAG module built), edge used as an auxiliary output only (no gating) | 3 |
| E4 | E3 + edge loss | 4 |
| E5 | E4 + EAG gating (the full CamoEdge) | 4 |
| E6 | λ_e sweep; gating variant (multiplicative versus concatenate) | 4 |
| E7 | PVTv2 versus ResNet-50 encoder | 5 |
| E8 | Final 3-seed runs; per-category and boundary analysis | 5 |
| E9 | Parameters, GFLOPs, FPS | 6 |

## Training setup (starting point, adjust after a trial)
Input about 352 × 352; random flip, crop, rotation and colour jitter; Adam; mixed precision; a few tens of epochs on 4,040 images. PVTv2-B2 at this size should fit on a free Colab T4 with a modest batch size; confirm with a one-epoch trial and check memory. Develop with a 100-image subset locally before spending Colab time, and train in resumable chunks with checkpoints on Drive.

## Demo and stretch
- Gradio tab (`camoedge/demoTab.py`, `buildTab()`): upload an image, show the predicted mask, edge map, overlay, and a slider for the threshold.
- Stretch: train a `yolo26-seg` baseline on COD10K for a "general detector versus specialised model" comparison. It is not in the proposal, so do it only if time remains.

## Files to create
```
camoedge/dataset.py     loader, augmentation, edge ground-truth generation
camoedge/model.py       encoder, EGB, EAG, decoder
camoedge/losses.py      weighted BCE, weighted IoU, edge loss
camoedge/metrics.py     wrapper around PySODMetrics + boundary metric
camoedge/train.py       training, checkpointing to Drive, resume
camoedge/evaluate.py    per-dataset and per-category tables
camoedge/demoTab.py
camoedge/camoedge_colab.ipynb
tests/test_camoedge_metrics.py
```

## Risks
- Largest unknown in the project: a from-scratch two-branch network. Mitigation: build in the order of the experiment matrix so there is a working model (E1) after the first few days and every later step is a small change.
- Reproducing published numbers is hard; compare against the same-protocol reference rather than claiming state of the art.
- Aggregate metrics may show small gains; the boundary metric and the category breakdown are where the effect should be visible.
- Minimum viable scope if time runs short: E1 → E4 plus metrics. EAG gating and the boundary metric are the first things to cut.
