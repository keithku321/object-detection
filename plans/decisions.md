# Decisions Log

Add one row per decision. A change of model version needs the evidence listed in the fallback rule in [00-main-plan.md](00-main-plan.md) and approval from two other owners.

| ID | Date | Decision | Decided by | Reason / evidence |
|---|---|---|---|---|
| D1 | 2026-10-11 | YOLO26 is the default model family for Plans A, B, C | Team | Team preference; fall back only on a hard technical issue |
| D2 | 2026-10-11 | Free Colab, one account per person, plus local GPUs | Team | Team constraint |
| D3 | 2026-10-11 | VisDrone is used for educational purposes; no redistribution | Team | Secondary sources describe CC BY-NC-SA 3.0. Checked 2026-10-17: the official VisDrone-Dataset GitHub repo has no LICENSE file and its README states no licence, so the official terms are still unconfirmed; ask AISKYEYE or the supervisor |
| D4 | 2026-10-11 | CamoEdge remains a segmentation project (PyTorch, not Ultralytics) | Team | Proposal defines mask metrics and a PVTv2/ResNet encoder |
| D5 | 2026-10-11 | Ultralytics pinned to `8.4.175` initially | Team | Version used to read the YOLO26 source for these plans |
| D6 | _open_ | Numeric accuracy targets per plan | Each owner | Set at gate G1 after baselines exist |
| D7 | _open_ | Plan owners and stewardship of shared files | Team | See `05-team-workflow.md`. 2026-10-22: Plan C (small object) is owned by Ku Sui Ki; Plans A, B and D are still to be assigned |
| D8 | _open_ | Where final weights are published | Team | GitHub release or Hugging Face model repo. 2026-10-22, proposed, team to confirm: the final weights that the training scripts copy to `runs/weights/` are committed for Plan A (`underwater-*.pt`: Brackish is CC BY 4.0, weights AGPL-3.0 like Ultralytics') and Plan D (`camoedge-*.pt`: the team's own code, published for non-commercial research only as COD10K and CAMO require); terms in `runs/weights/README.md`. Plans B and C are not committed: DAWN is non-commercial and VisDrone's terms are unconfirmed, while Ultralytics-trained weights are AGPL-3.0, which allows commercial use and forbids adding a non-commercial condition. Note that the GitHub repository is public (see D9) |
| D9 | _open_ | Whether the repo/demo will be public (AGPL implication) | Team + supervisor | Ultralytics is AGPL-3.0; repo is MIT. Checked 2026-10-22: the GitHub repository `keithku321/object-detection` is already **public** and its `LICENSE.txt` is AGPL-3.0, so everything committed (results, plots, the weights of D8) is published; the team and supervisor should confirm this or make it private |
| D10 | _open_ | Which split the reported accuracy comes from | Team | Today: Plans A and C log `val` (Brackish `test` is used only for the turbidity curves; VisDrone test-dev is not used); Plan B logs the held-out fold from `last.pt`; Plan D logs the test sets from the last epoch. Record the final choice per plan before the final runs |
| D13 | 2026-10-17 | Brackish is re-split by video clip instead of using the Roboflow split | Proposed, team to confirm | The Roboflow split put all 89 source clips into train, val and test (every val and test image had frames from its own clip in train; a perceptual-hash check at distance 4 linked 14,446 of 14,674 images across splits). `underwater/prepareBrackish.py` now keeps each clip whole in one split (11,612 train, 1,575 val, 1,487 test images; seed 231 chosen from 500 clip orders by class balance of val and test, never by model results; no clip in two splits). Lists are in `common/splits/brackish/` (`original-<split>.txt` keeps the Roboflow assignment). Accuracy is lower but honest; frames of different clips from the same fixed camera still look alike. Any run on the Roboflow split must say so |
| D11 | 2026-10-17 | DAWN class numbers are renumbered by class name when converting | Proposed, team to confirm | The YOLO files of the real download number classes like COCO counting from 1 (1 person, 2 bicycle, 3 car, 4 motorcycle, 6 bus, 7 train, 8 truck), checked against the PASCAL VOC names of all 1,027 images. Copying them unchanged would train on wrong and out-of-range classes. `sourceYoloClassIds` in the manifest holds the table; the one `train` box is dropped |
| D12 | 2026-10-17 | Local dataset copies live in `<plan folder>/dataset/` (git-ignored) and are made by scripts | Proposed, team to confirm | Datasets are never committed. Only datasets under 1 GB with terms that allow private educational use are fetched this way (DAWN, Brackish). VisDrone (1.8 GB, Ultralytics downloads it) and COD10K (about 1.5 GB) stay manual |

## Spike results (Phase 0, fill in)
| Date | Owner | Question | Result |
|---|---|---|---|
| | | Custom layer insertion: how is width scaling handled, and what is "Transferred N/M"? | |
| | | Can the end-to-end loss be subclassed with an extra distillation term? | |
| | | End-to-end latency measured with `benchmark.py` | |
