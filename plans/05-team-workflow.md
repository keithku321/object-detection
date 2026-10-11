# Team Workflow: Roles, Merging and Shared Interfaces

Companion to [00-main-plan.md](00-main-plan.md). Four people, one proposal each, equal say. Nobody sits above anyone else: shared files are split between the four owners as **stewards**, and reviews are peer reviews.

## Plans and owners
| Plan | Task | Owner | Hardware (fill in) |
|---|---|---|---|
| A | Underwater ([01](01-underwater.md)) | _TBD (the underwater proposal is written by Chan Sze Ham; confirm)_ | |
| B | WeatherDistill ([02](02-weather-distill.md)) | _TBD_ | |
| C | Small object ([03](03-small-object.md)) | _TBD_ | |
| D | CamoEdge ([04](04-camoedge.md)) | _TBD_ | |

Assign plans by interest and skill. Two practical notes: Plan C needs the most GPU time, so it suits whoever has the strongest local GPU; Plan D is plain PyTorch with the least reuse of the others' code, so it suits someone comfortable working independently. The plans differ in difficulty, not in importance; stewardship (below) is spread so that the total workload comes out similar.

## Shared files and their stewards
A steward looks after a shared file: keeps it working, answers questions about it, and reviews changes to it. Stewards do not have final say; changes are decided by review (see Repository rules).

| Shared file or area | Steward |
|---|---|
| Environment, `requirements.txt`, `common/datasets.py`, `datasets/manifest.json`, `demo/app.py` | A |
| `common/results.py`, results summary script, `decisions.md`, the notebook template | B |
| `common/modules.py`, `common/registerModules.py` | C |
| `common/benchmark.py` | D |

The easiest plan (A) carries the most shared files, and the heaviest compute plan (C) the fewest. Swap stewardships if the load turns out uneven.

## Repository rules
1. **One folder per owner.** Owners edit only their own task folder, their own `results/<task>.csv` and their own notebook. This avoids most merge conflicts.
2. **Shared files** (`common/`, `requirements.txt`, `datasets/manifest.json`, `demo/app.py`) change through a pull request that **one other owner** approves; the steward is the natural reviewer, and a steward's own change is reviewed by someone else. A pin change in `requirements.txt` or a model-version fallback needs **two** other owners.
3. **Branches:** `main` holds merged phases; owners work on `owner/<task>-<topic>` branches and open a PR at each phase gate. The person who merges and tags `phase-N-done` rotates each gate (A, B, C, D, then A again).
4. **Notebooks:** strip outputs before committing; otherwise notebook diffs become unreadable. One notebook per owner per task, named `<task>_colab.ipynb`, created by copying the template notebook.
5. **Never commit** datasets, weights, API tokens or Drive links with edit access. `.gitignore` already excludes data and weights.
6. **Version pin:** everyone installs from `requirements.txt`. Nobody upgrades Ultralytics locally "to try it".
7. **Reproducibility:** every result row carries the git commit and Ultralytics version; the report cites rows, not screenshots.

## Shared interfaces (the contract between owners)
| Interface | Definition | Steward |
|---|---|---|
| Results row | `task, model, variant, seed, metrics…, gpu, commit, ultralytics` appended via `common/results.py` | B |
| Dataset entry | name, source URL, license/terms, SHA-256, conversion script, in `datasets/manifest.json` | A; owners add their own datasets |
| Custom layers | `common/modules.py` classes registered through `common/registerModules.py`; each layer documents its channel arguments | C |
| Speed numbers | only via `common/benchmark.py` | D |
| Demo tab | each task folder has `demoTab.py` exposing `buildTab() -> gradio.Blocks`; it loads weights from a path or release URL set in a config | each owner |
| Weights naming | `<task>-<variant>-seed<k>.pt`; final ones uploaded to the release location named in `decisions.md` | each owner |

## Phase gates (how merging works)
1. Owners finish their phase checklist items and push results rows.
2. Each owner writes a 5–10 line status in their PR: what was done, the table row(s), surprises, and what changes next phase.
3. The team reviews the PRs together, merging shared-code changes first. The rotating merger tags the gate.
4. Decisions made at the gate go into `decisions.md`.

If an owner is behind at a gate, the others continue; gates only block *shared-code* changes, not individual work.

## Communication
- Surprises that change a plan go in `decisions.md` with date, owner and reason.
- A fallback from YOLO26 requires the evidence listed in the main plan and approval from two other owners.
- Ask for help when stuck for more than a day on environment or Colab problems; these are the most common time sinks.
