# Decisions Log

Add one row per decision. A change of model version needs the evidence listed in the fallback rule in [00-main-plan.md](00-main-plan.md) and approval from two other owners.

| ID | Date | Decision | Decided by | Reason / evidence |
|---|---|---|---|---|
| D1 | 2026-10-11 | YOLO26 is the default model family for Plans A, B, C | Team | Team preference; fall back only on a hard technical issue |
| D2 | 2026-10-11 | Free Colab, one account per person, plus local GPUs | Team | Team constraint |
| D3 | 2026-10-11 | VisDrone is used for educational purposes; no redistribution | Team | Secondary sources describe CC BY-NC-SA 3.0; confirm official license text and record it in the manifest |
| D4 | 2026-10-11 | CamoEdge remains a segmentation project (PyTorch, not Ultralytics) | Team | Proposal defines mask metrics and a PVTv2/ResNet encoder |
| D5 | 2026-10-11 | Ultralytics pinned to `8.4.175` initially | Team | Version used to read the YOLO26 source for these plans |
| D6 | _open_ | Numeric accuracy targets per plan | Each owner | Set at gate G1 after baselines exist |
| D7 | _open_ | Plan owners and stewardship of shared files | Team | See `05-team-workflow.md` |
| D8 | _open_ | Where final weights are published | Team | GitHub release or Hugging Face model repo |
| D9 | _open_ | Whether the repo/demo will be public (AGPL implication) | Team + supervisor | Ultralytics is AGPL-3.0; repo is MIT |

## Spike results (Phase 0, fill in)
| Date | Owner | Question | Result |
|---|---|---|---|
| | | Custom layer insertion: how is width scaling handled, and what is "Transferred N/M"? | |
| | | Can the end-to-end loss be subclassed with an extra distillation term? | |
| | | End-to-end latency measured with `benchmark.py` | |
