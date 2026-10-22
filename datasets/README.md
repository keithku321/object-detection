# Dataset downloads

Where to get every dataset of the project, how big it is, what its terms say, and what this repository already does for you. The machine-readable version (names, class lists, checksums) is [`manifest.json`](manifest.json); the plans give the reasons ([plans/00-main-plan.md](../plans/00-main-plan.md) section 4).

**Checked on 2026-10-17 from a cloud container.** "Opened" means the link answered when this list was written; "not tested" means it is a Google Drive or Baidu link that was copied from the authors' own page and not downloaded. Google Drive locks files that are downloaded very often, and Baidu needs an account, so those links can fail. Terms are a practical reading of what the authors publish, not legal advice.

## At a glance

| Dataset | Plan | Size | Terms in short | Easiest route here |
|---|---|---|---|---|
| [Brackish](#brackish-plan-a) | A | 0.76 GB | CC BY 4.0 (read from the copies, see below) | Already in the repository: `underwater/dataset/brackish/` |
| [URPC / DUO](#urpc-and-duo-plan-a-second-dataset) | A | not checked | contest data / not stated | Manual download of DUO |
| [DAWN](#dawn-plan-b) | B | 139 MB | Mendeley: CC BY-NC 3.0, but "all rights reserved, no distribution" | `python -m weather.prepareDawn --download-to weather/dataset --output weather/dataset/dawn` |
| [VisDrone-DET](#visdrone-det-plan-c) | C | 1.94 GB | Copyright AISKYEYE; terms unconfirmed | Ultralytics downloads it when you train with `VisDrone.yaml` |
| [COD10K](#cod10k-plan-d) | D | 1.8 GB (two zips, with CAMO) | non-commercial only | `python -m camoedge.prepareCamo --output camoedge/dataset` |
| [CAMO](#camo-plan-d) | D | inside the COD10K zips | research use | same command as COD10K |
| [NC4K](#nc4k-plan-d-optional) | D (optional) | not checked | not stated | Manual download |

## Brackish (Plan A)

Underwater video frames with boxes for crab, fish, jellyfish, shrimp, small_fish and starfish, recorded by Aalborg University in the Limfjord (Denmark). 14,674 images.

| What | Link |
|---|---|
| Project page (authors, citation) | https://vap.aau.dk/the-brackish-dataset/ (opened) |
| Dataset index of the group | https://vap.aau.dk/dataset/ (opened) |
| Original release on Kaggle, linked from the project page | https://www.kaggle.com/aalborguniversity/brackish-dataset (answered 404 from the container, not tested) |
| Paper | http://openaccess.thecvf.com/content_CVPRW_2019/papers/AAMVEM/Pedersen_Detection_of_Marine_Animals_in_a_New_Underwater_Dataset_with_CVPRW_2019_paper.pdf |
| Code | https://bitbucket.org/aauvap/brackish-dataset/src/master/ |
| Roboflow export (needs a free login) | https://universe.roboflow.com/brad-dwyer/brackish-underwater (answered 403 from the container, not tested) |
| **Hugging Face copy used here** (unofficial, CC BY 4.0, 712 MB) | https://huggingface.co/datasets/dronefreak/Brackish (opened and downloaded) |

- **Terms:** CC BY 4.0. That statement is printed in the Roboflow README ("Provided by Aalborg University") and on the Hugging Face card; Aalborg's own project page states none. Attribution and the changes made are in [`underwater/dataset/brackish/LICENSE.md`](../underwater/dataset/brackish/LICENSE.md).
- **In this repository:** the prepared copy is tracked in git, split by video clip (decision D13). To rebuild it: `python -m underwater.prepareBrackish --output underwater/dataset --zip`.
- **Catch:** the Roboflow split puts every clip into train, val and test; the prepared copy fixes that. Hugging Face stores the training split in four shard folders, which the script merges.

## URPC and DUO (Plan A, second dataset)

- **URPC** (holothurian, echinus, scallop, starfish): the contest downloads are closed and no verified link exists. Many images are near-duplicates, so use DUO instead.
- **DUO**, the cleaned benchmark of the same four classes:

| What | Link |
|---|---|
| Repository | https://github.com/chongweiliu/DUO (opened) |
| Google Drive | https://drive.google.com/file/d/1w-bWevH7jFs7A1bIBlAOvXOxe2OFSHHs/view?usp=sharing (not tested) |
| Baidu (key `4bfl`) | https://pan.baidu.com/s/1Be8zc9UdR_Pdsyotg_vR2Q (not tested) |
| Paper (arXiv 2106.05681) | https://arxiv.org/abs/2106.05681 (opened) |

DUO's README states no licence, so read it as research use and do not republish it. Its labels are probably COCO JSON or VOC XML: convert with `convertVocXmlToYolo` in `common/datasets.py` and check the class order against the label files.

## DAWN (Plan B)

About 1,000 real-traffic images in four weather sets (fog, rain, sand, snow), 1,027 images in the download.

| What | Link |
|---|---|
| Dataset page (version 3, DOI 10.17632/766ygrbt8y.3) | https://data.mendeley.com/datasets/766ygrbt8y/3 (opened) |
| Fog.zip, 30.6 MB | https://data.mendeley.com/public-files/datasets/766ygrbt8y/files/c286e791-88c9-4bec-b43d-2ed276dd11ba/file_downloaded (downloaded) |
| Rain.zip, 29.9 MB | https://data.mendeley.com/public-files/datasets/766ygrbt8y/files/7b37087a-db6b-4ce9-8ec4-ee64f963f073/file_downloaded (downloaded) |
| Sand.zip, 34.9 MB | https://data.mendeley.com/public-files/datasets/766ygrbt8y/files/51071d4d-5505-4e0e-acf7-495bc3606b02/file_downloaded (downloaded) |
| Snow.zip, 43.9 MB | https://data.mendeley.com/public-files/datasets/766ygrbt8y/files/d5abcff0-484f-49c9-b9f7-bf36c3b325a2/file_downloaded (downloaded) |
| Paper (arXiv 2008.05402) | https://arxiv.org/abs/2008.05402 (opened) |

- **Terms:** the Mendeley licence field says CC BY-NC 3.0, but the description says all rights are reserved and commercial use and distribution are strictly prohibited. Read it as the stricter one: private, educational use, **never commit or mirror it**.
- **Catch:** the download needs a User-Agent header (`common/datasets.py` sends one), and its YOLO files number classes like COCO counting from 1, so `weather/prepareDawn.py` renumbers them (decision D11). The checksums Mendeley publishes are in `manifest.json`.

## VisDrone-DET (Plan C)

Drone images, 10 classes: 6,471 train, 548 val and 1,610 test-dev images.

| What | Link |
|---|---|
| Official page with all download links | https://github.com/VisDrone/VisDrone-Dataset (opened) |
| Authors' train set, 1.44 GB | Google Drive https://drive.google.com/file/d/1a2oHjcEcwXP8oUF95qiwrqzACb2YlUhn/view?usp=sharing · Baidu https://pan.baidu.com/s/1K-JtLnlHw98UuBDrYJvw3A (not tested) |
| Authors' val set, 0.07 GB | Google Drive https://drive.google.com/file/d/1bxK5zgLn0_L8x276eKkuYA_FzwCIjb59/view?usp=sharing · Baidu https://pan.baidu.com/s/1jdK_dAxRJeF2Xi50IoML1g (not tested) |
| Authors' test-dev set, 0.28 GB (labels included) | Google Drive https://drive.google.com/open?id=1PFdW_VFSCfZ_sTSZAGjQdifF_Xd5mf0V · Baidu https://pan.baidu.com/s/1RdRfSWV-1IFK7aWljLU_LQ (not tested) |
| Ultralytics copy of the train set, 1,549,875,511 bytes | https://github.com/ultralytics/assets/releases/download/v0.0.0/VisDrone2019-DET-train.zip (size checked) |
| Ultralytics copy of the val set, 81,638,851 bytes | https://github.com/ultralytics/assets/releases/download/v0.0.0/VisDrone2019-DET-val.zip (size checked) |
| Ultralytics copy of the test-dev set, 311,251,787 bytes | https://github.com/ultralytics/assets/releases/download/v0.0.0/VisDrone2019-DET-test-dev.zip (size checked) |
| Ultralytics documentation | https://docs.ultralytics.com/datasets/detect/visdrone |
| Paper | Zhu et al., "Detection and tracking meet drones challenge", IEEE TPAMI 44(11), 2021 |

- **Terms:** copyright AISKYEYE (Tianjin University). The official repository has no LICENSE file and its README states no licence, so the terms are unconfirmed (decision D3). Use it for the project only, cite the paper, and do not republish it.
- **Why it downloads by itself:** see the next section.

### Why VisDrone downloads automatically

The other datasets sit behind Google Drive, Baidu or a login, so a script cannot fetch them. VisDrone is different only because **Ultralytics** (the YOLO library) ships a dataset file, `VisDrone.yaml`, with a `download:` script inside it:

1. When you train with `data="VisDrone.yaml"` and the folder `VisDrone/` does not exist yet, Ultralytics runs that script.
2. The script downloads the three zips above from Ultralytics' own GitHub release (`ASSETS_URL`, `github.com/ultralytics/assets/releases/download/v0.0.0`). Those are copies that Ultralytics hosts, not the authors' links, and GitHub release files have no login or quota.
3. It converts the annotations to YOLO format (class number minus 1, ignored regions skipped), moves the images into `images/<split>` and `labels/<split>`, and deletes the original folders.

It is the library's feature, not something this repository added: `smallobject/train.py` and the notebook just pass the name `VisDrone.yaml`, and the manifest entry has `"access": "ultralytics"`. It needs about 2 GB to download and about 4 GB free in total, which is why the team did not put it in `plans/*/dataset/` (above the 1 GB limit). Because Ultralytics redistributes the data itself, its terms are not clearer than the authors'; our rule of not republishing still applies.

## COD10K (Plan D)

Camouflaged object segmentation: 10,000 images, of which the 5,066 camouflaged ones are used (3,040 train and 2,026 test). File names carry the super-class (for example `COD10K-CAM-1-Aquatic-1-BatFish-1.jpg`).

| What | Link |
|---|---|
| Repository with the download links | https://github.com/DengPingFan/SINet (opened) |
| Full COD10K | Google Drive https://drive.google.com/file/d/1pVq1rWXCwkMbEZpTt4-yUQ3NsnQd_DNY/view?usp=sharing · Baidu https://pan.baidu.com/s/15GnPrMd6TGzMufEQ9IC66Q (code `aq4i`) (not tested) |
| Training set | Google Drive https://drive.google.com/file/d/1D9bf1KeeCJsxxri6d2qAC7z6O1X_fxpt/view?usp=sharing · Baidu https://pan.baidu.com/s/1XL6OjpDF-MVnXOY6-bdaBg (code `djq2`) (Google Drive link downloaded 2026-10-22; Baidu not tested) |
| Test sets (COD10K-test + CAMO-test + CHAMELEON in one zip) | Google Drive https://drive.google.com/file/d/1QEGnP9O7HbN_2tH999O3HRIsErIVYalx/view?usp=sharing · Baidu https://pan.baidu.com/s/143yHFLAabMBT7wgXA0LrMg (code `z83z`) (Google Drive link downloaded 2026-10-22; Baidu not tested) |
| Unofficial Hugging Face copy, 1.57 GB, tagged cc-by-nc-4.0 | https://huggingface.co/datasets/nobg/COD10K (listed, not downloaded) |

- **Terms:** the repository says COD10K is available for non-commercial purposes only and may not be reproduced, used or conveyed for commercial purposes. Do not republish.
- **Checked on 2026-10-22:** the training zip (`TrainDataset.zip`, 1.14 GB) holds **4,040 pairs: COD10K-Tr 3,040 plus CAMO-Tr 1,000** (named `camourflage_*`), with `Imgs/`, `GT/` and `Edge/`, although the SINet README calls it "COD10K-train". The test zip (`TestDataset.zip`, 0.72 GB) holds COD10K-Te 2,026, CAMO-Te 250 and CHAMELEON 76. Together they are exactly the Plan D layout, so nothing has to be rearranged.
- **In this repository:** `python -m camoedge.prepareCamo --output camoedge/dataset` downloads both zips with `gdown` (in `requirements.txt`; Google Drive needs its confirmation step), checks the SHA-256 values recorded in the `camoedge` entry of the manifest, unzips them into `camoedge/dataset/camo` (git-ignored) and checks the counts. Both downloads took under a minute from the cloud container.

## CAMO (Plan D)

1,250 images (1,000 train and 250 test) of camouflaged objects.

| What | Link |
|---|---|
| Official project page (the download link could not be read from the container: the page is built with scripts) | https://sites.google.com/view/ltnghia/research/camo (opened) |
| CAMO-test inside the SINet test zip and CAMO-train inside the SINet training zip (checked 2026-10-22) | see COD10K above; `python -m camoedge.prepareCamo` fetches both |
| Unofficial Hugging Face copy, 214 MB, no licence stated | https://huggingface.co/datasets/PassbyGrocer/CAMO (listed, not downloaded) |
| Paper | Le et al., "Anabranch network for camouflaged object segmentation", Computer Vision and Image Understanding, 2019 |

Terms: research distribution by the authors; read the README that comes with the download, and do not republish.

## NC4K (Plan D, optional)

4,121 test-only images for checking generalisation.

| What | Link |
|---|---|
| Repository | https://github.com/JingZhang617/COD-Rank-Localize-and-Segment (opened) |
| NC4K, Google Drive | https://drive.google.com/file/d/1kzpX_U3gbgO9MuwZIWTuRVpiB7V6yrAQ/view?usp=sharing (not tested) |
| NC4K, Baidu (password `d581`) | https://pan.baidu.com/s/1bG4F2KJ_4UJG_7XG6ZNBHA (not tested) |
| Paper | Lyu et al., "Simultaneously Localize, Segment and Rank the Camouflaged Objects", CVPR 2021 |

No licence is stated in the README, so treat it as research use and do not republish.

## Rules that go with these links

- **Never commit** DAWN, VisDrone, COD10K, CAMO, NC4K, URPC or DUO. Only Brackish (CC BY 4.0, with its `LICENSE.md`) may sit in git; `.gitignore` enforces this.
- **Colab:** get a zip to `/content` and unzip it there; Drive is for checkpoints and results only. A hand-made zip has one top-level folder named like the notebook's `datasetFolderName`, and its SHA-256 belongs in the manifest entry whose `zipName` it is.
- **Record a change:** when a link here stops working or a licence is confirmed, update this file and `manifest.json` together, and add a line to `plans/decisions.md` if it changes a rule.
