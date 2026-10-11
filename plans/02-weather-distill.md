# Plan B: WeatherDistill — Knowledge Distillation for Adverse-Weather Detection (YOLO26)

Proposal: *WeatherDistill: Knowledge Distillation for Lightweight Object Detection in Adverse Weather* (DAWN). Main plan: [00-main-plan.md](00-main-plan.md). Team workflow: [05-team-workflow.md](05-team-workflow.md). Difficulty rank: **2 of 4**. Owner: _TBD_.

## Goal and success criteria
Transfer accuracy from a **YOLO26m teacher to a YOLO26n student** on DAWN with no increase in student size or inference cost. (The proposal says YOLOv8m → YOLOv8n; the model family changes, the method does not.)

| Criterion | How it is judged |
|---|---|
| Distilled student beats the plain student | mAP50 and mAP50-95, mean ± std over 5 folds × seeds |
| Per-weather effect | Same metrics for fog, rain, snow and sand separately |
| Cost unchanged | Student parameters and GFLOPs identical to a standard `yolo26n` |
| Compression (extension) | Pruned + INT8 student: accuracy versus size versus latency |

## Data: DAWN
- About 1,000 real-traffic images in four weather sets (fog, rain, snow, sand). Classes: car, bus, truck, person, motorcycle, bicycle. Annotations in XML and TXT (YOLO). Verify the class order against the YOLO files.
- **Terms:** the Mendeley page reserves all rights and prohibits commercial use and distribution; the images came from web searches. Educational use is fine; do not publish a mirror. `common/datasets.py` downloads from the original source; keep any processed copy private.
- Do not train from scratch on 1,000 images: start every model from COCO-pretrained weights.
- **Splits:** a single 70/15/15 split leaves about 150 test images, too noisy. Use **stratified 5-fold cross-validation** (stratify by weather type); commit each fold's file list. Check for near-duplicate images across folds with perceptual hashing.

## Method

### What the model gives us (YOLO26, from the source)
- Training output has two heads, `one2many` and `one2one`. Each holds `boxes` (four direct distances per anchor, since `reg_max = 1`), `scores` (class logits) and `feats` (the neck feature maps). The one-to-one head receives *detached* features, so only the one-to-many head sends gradients into the backbone.
- The criterion is a weighted sum of two detection losses whose weights shift from one-to-many to one-to-one as training proceeds.
- There is **no box-distribution output**, so the proposal's "KL divergence on the regression head" cannot be applied as written. This is a change of method, not a blocker: record it in `decisions.md` and tell the supervisor.

### Distillation terms (training-only)
Total loss = standard end-to-end loss + λ_feat · L_feat + λ_cls · L_cls + λ_box · L_box.

1. **L_feat — feature MSE.** On the three neck feature maps in `feats` (no hooks needed). A 1×1 convolution adapter maps student channels to teacher channels; adapters exist only during training. Normalise features before the MSE. Ablation: restrict to foreground regions (ground-truth boxes, slightly dilated), because background dominates dense detectors.
2. **L_cls — classification distillation.** Class scores are independent sigmoid outputs, not a softmax, so use a temperature-softened **binary KL** per class (or soft-target BCE) between teacher and student logits. State this choice in the report.
3. **L_box — regression distillation.** Replaces the DFL-based KL. On the anchors the student's assigner marks as positive, penalise the L1 (and optionally GIoU) distance between student and teacher box predictions, weighted by teacher confidence.
4. **Which head:** apply L_feat, L_cls and L_box to the **one-to-many** head (it trains the backbone). Test as an ablation whether adding them to the one-to-one head helps. Optionally follow the same decaying weight schedule as the base loss.
5. Tune λ on fold 0 validation only, over a small grid.

### Implementation notes
- Subclass the end-to-end loss class and the detection trainer/model so the student's criterion is the wrapped one. Run the teacher in eval mode under `no_grad` on the **same augmented batch**.
- Keep the **teacher and adapters out of the student's `state_dict`**; otherwise checkpoints bloat and parameter counts are wrong.
- Pin the Ultralytics version; the loss internals change between releases.

### Correctness checks before any real run
- All λ = 0 reproduces the plain student within seed noise.
- Teacher used as its own student gives distillation loss near 0.
- A 1-epoch smoke test on 8 images passes (`tests/`).
- Final `best.pt` has exactly the parameter count of a standard `yolo26n`.

## Experiment matrix
| ID | Setup | Phase |
|---|---|---|
| W0 | Student `yolo26n`, no distillation, 5 folds | 1 |
| W1 | Teacher `yolo26m`, 5 folds (upper reference); optional `yolo26s` | 1 |
| W2 | Distillation trainer skeleton and checks (λ = 0, identity) | 2 |
| W3 | Feature distillation only | 3 |
| W4 | Classification distillation only; box distillation only | 4 |
| W5 | Feature + classification + box (full WeatherDistill) | 4 |
| W6 | Foreground-masked features; λ and temperature sweep (fold 0); one-to-one head ablation | 4 |
| W7 | Pruned (about 40% channels) + fine-tune with distillation | 6 |
| W8 | W7 + static INT8 | 6 |
| W9 | Final runs, all folds × 3 seeds, per-weather tables | 5 |
| W10 | _Optional:_ proposal-faithful DFL-KL variant on a YOLO11 pair (`yolo11m` → `yolo11n`) | 5 |

Per-weather evaluation: validate on each weather subset of the held-out fold so fog, rain, snow and sand each get their own mAP.

## Compression (Phase 6, optional)
- **Pruning:** structured channel pruning to about 40% with a library built for it (for example torch-pruning). C3k2/concat/split layers make this fiddly; first prove it on the plain baseline with a short fine-tune, then add distillation.
- **Quantization:** the proposal says "dynamic quantization". PyTorch dynamic quantization covers Linear and recurrent layers, not Conv2d, so it barely changes a YOLO model. Use **static post-training INT8** with calibration images from the training set, through an exporter (ONNX Runtime, OpenVINO or TensorRT). YOLO26's removal of DFL is reported to ease quantization and export; verify it on your own model.
- **Edge evidence:** there is no Jetson in the project. The proposal's Jetson Nano figures (24.1 FPS, 7.2 GFLOPs, 69.10 mAP) are for YOLOv8n; cite their source or re-measure. For your own models report parameters, GFLOPs, GPU latency and CPU (ONNX Runtime) latency.

## Files to create
```
weather/prepareDawn.py     convert, stratified 5-fold manifests
weather/distillTrainer.py  trainer, wrapped loss, adapters
weather/evalByWeather.py   per-weather metrics
weather/prune.py           pruning (Phase 6)
weather/demoTab.py
weather/weather_colab.ipynb
tests/test_distill.py      λ = 0, identity and smoke tests
```

## Free-Colab notes
Nano and medium models on 800 images per fold are cheap; most of this plan can be developed locally and only the 5-fold × 3-seed sweeps need Colab, in resumable chunks.

## Risks
- Gains on a small dataset can be smaller than noise. Cross-validation and seeds make the result credible; a negative result is still reportable if measured properly.
- DAWN images are web-scraped and may overlap with COCO pretraining data; mention it as a limitation.
- Pruning is the most likely place to lose time; keep it behind gate G5.
- The wrapped end-to-end loss touches Ultralytics internals; pin the version and keep the smoke test in CI-style checks.
