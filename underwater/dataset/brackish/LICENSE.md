# Brackish underwater dataset: licence and attribution

This folder holds a copy of the **Brackish** dataset. It is **not** covered by the AGPL-3.0 licence of the code in this repository; the data is
shared under **Creative Commons Attribution 4.0 International (CC BY 4.0)**: https://creativecommons.org/licenses/by/4.0/

## Attribution

Pedersen, M., Haurum, J. B., Gade, R., Moeslund, T. B., Madsen, N. (2019). *Detection of Marine Animals in a New Underwater Dataset with
Varying Visibility.* IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops. Aalborg University, Denmark.
Project page: https://vap.aau.dk/the-brackish-dataset/

```
@InProceedings{pedersen2019brackish,
  title     = {Detection of Marine Animals in a New Underwater Dataset with Varying Visibility},
  author    = {Pedersen, Malte and Haurum, Joakim Bruslund and Gade, Rikke and Moeslund, Thomas B. and Madsen, Niels},
  booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
  year      = {2019}
}
```

## Where this copy comes from

1. Aalborg University recorded the video and annotated it.
2. Roboflow extracted the frames and exported them in YOLO format (https://universe.roboflow.com/brad-dwyer/brackish-underwater, version 1,
   stated there as "Provided by Aalborg University", licence CC BY 4.0). Their notes are kept in `README.dataset.txt` and `README.roboflow.txt`.
3. The Hugging Face dataset `dronefreak/Brackish` (licence tag cc-by-4.0) is an unofficial redistribution of that export; this copy was
   downloaded from it on 2026-10-17.

The CC BY 4.0 statement was read from the Roboflow README and the Hugging Face card. The Aalborg University project page does not state a
licence on the page itself and the original Kaggle release could not be reached when this was written, so confirm it there if it matters.

## Changes made in this project (CC BY 4.0 asks that changes are indicated)

- The training split was stored in four shard folders; they were merged, and the loader files `metadata.jsonl` and the Roboflow `data.yaml`
  files were removed.
- Folders were renamed to `images/<split>` and `labels/<split>`, with the Roboflow split `valid` called `val`.
- **The images were re-split by video clip** (11,612 train, 1,575 val, 1,487 test) because the original split put every clip into all
  three splits. The lists are in `common/splits/brackish/` (`original-<split>.txt` is the Roboflow assignment). Image pixels and
  label contents were not changed.

## No warranty

The data is provided as it is, without warranty of any kind. Use it for detection research; the licence does not restrict that, but you must
keep this attribution when you share it.
