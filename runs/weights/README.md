# Final weights

The training scripts copy the final weights of every finished run into this folder as `<task>-<variant>-seed<k>.pt` (for example
`underwater-baseline-50ep-seed0.pt` or `camoedge-E1-seed0.pt`). Only the files in the first two rows below may be committed (`.gitignore`, decision D8 in
[`plans/decisions.md`](../../plans/decisions.md)); everything else under `runs/` stays on the machine that trained it. The results row of a file in
`results/<task>.csv` gives its metrics, settings and git commit.

| Files | Plan | Trained on | Terms of the weights |
|---|---|---|---|
| `underwater-*.pt` | A | Brackish (Aalborg University, the CC BY 4.0 copy described in `underwater/dataset/brackish/LICENSE.md`), starting from Ultralytics' COCO-pretrained YOLO26 weights | GNU AGPL-3.0, as Ultralytics' own weights. Credit Brackish: Pedersen et al., "Detection of Marine Animals in a New Underwater Dataset with Varying Visibility", CVPR Workshops 2019 |
| `camoedge-*.pt` | D | The COD10K and CAMO training sets (4,040 images), with an ImageNet-pretrained PVTv2-B2 encoder from `timm` | **Non-commercial research and education only**, following the terms of COD10K and CAMO. Cite Fan et al., "Camouflaged Object Detection", CVPR 2020, and Le et al., "Anabranch Network for Camouflaged Object Segmentation", CVIU 2019 |
| `weather-*.pt`, `smallobject-*.pt` | B, C | DAWN, VisDrone | **Not committed.** DAWN is non-commercial ("all rights reserved" in its description) and VisDrone's terms are unconfirmed, while weights trained with Ultralytics fall under the AGPL-3.0, which allows commercial use and forbids adding a non-commercial condition. Share them inside the team (Google Drive) until decisions D8 and D9 settle it |

Plan A files hold the model in half precision as Ultralytics saves it (about 5 MB); Plan D files hold the model weights in half precision and the run settings,
without the optimiser state (`camoedge/train.py exportFinalWeights`, about 50 MB). The checkpoints to resume from (`last.pt`) stay in the run folders. GitHub refuses files over 100 MB, and every new
version of a file stays in the git history, so commit only the weights of runs that the report or the demo uses.
