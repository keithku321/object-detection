# Download the Brackish dataset, give it the layout that underwater_colab.ipynb and underwater/train.py expect, and check it.
#   python -m underwater.prepareBrackish --output underwater/dataset
#   python -m underwater.prepareBrackish --output underwater/dataset --zip      (also builds brackish.zip for Google Drive or Hugging Face)
# The data comes from the unofficial CC BY 4.0 copy named in datasets/manifest.json (unofficialMirror), which is the Roboflow export of
# the Aalborg University dataset (80/10/10 split kept). Attribute Pedersen et al., Aalborg University, when you use or show it.
# Layout produced:  <output>/brackish/images/<train|val|test>   <output>/brackish/labels/<train|val|test>   <output>/brackish/dataset.yaml
# Near-identical frames from one video can sit on both sides of the split: run findCrossSplitDuplicates (common/datasets.py) before
# trusting an accuracy number (plans/01-underwater.md, risks).
import argparse
import shutil
from pathlib import Path

import yaml

from common.datasets import (checkClassNumbers, checkSplit, computeSha256, convertRoboflowLayout, getManifestEntry, writeDatasetYaml)

# Name of the dataset in datasets/manifest.json, and of the folder that holds it inside the output folder
datasetName = "brackish"
# Name of the file in the Roboflow export that lists the class names
sourceYamlName = "data.yaml"


# Download the files of the unofficial copy into a folder and return the folder that holds the Roboflow export (its data/ sub-folder)
def downloadMirror(downloadFolder, mirrorRepo):
    from huggingface_hub import snapshot_download
    snapshot_download(repo_id=mirrorRepo, repo_type="dataset", local_dir=str(downloadFolder), allow_patterns=["data/**"], max_workers=8)
    return Path(downloadFolder) / "data"


# Compare the class names in the export with the manifest; a different order would silently mislabel every box, so it stops the script
def checkClassOrder(sourceYamlPath, classNames):
    sourceNames = yaml.safe_load(Path(sourceYamlPath).read_text(encoding="utf-8"))["names"]
    sourceNames = [sourceNames[key] for key in sorted(sourceNames)] if isinstance(sourceNames, dict) else list(sourceNames)
    if sourceNames != list(classNames): raise ValueError(f"Class order of the download {sourceNames} differs from the manifest {list(classNames)}")


# Move the Roboflow export (train/, valid/, test/) to <outputRoot>/brackish, rename it to images/<split> and labels/<split>, write dataset.yaml
# and run the label checks of common/datasets.py; returns the path of dataset.yaml
def arrangeDataset(exportFolder, outputRoot, classNames):
    datasetRoot = Path(outputRoot) / datasetName
    if datasetRoot.exists(): raise FileExistsError(f"{datasetRoot} already exists; delete it to prepare the dataset again")
    checkClassOrder(Path(exportFolder) / sourceYamlName, classNames)
    datasetRoot.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(exportFolder), str(datasetRoot))
    movedSplits = convertRoboflowLayout(datasetRoot)
    yamlPath = writeDatasetYaml(datasetRoot, classNames, testSplit="images/test" if "test" in movedSplits else None)
    for splitName in movedSplits:
        checkSplit(datasetRoot, splitName)
        checkClassNumbers(datasetRoot, splitName, len(classNames))
    return yamlPath


# Zip the prepared dataset with brackish/ as the single top-level folder, which is what the notebook extracts; returns (zip path, SHA-256)
def buildZip(outputRoot):
    zipBase = Path(outputRoot) / datasetName
    zipPath = Path(shutil.make_archive(str(zipBase), "zip", root_dir=str(outputRoot), base_dir=datasetName))
    return zipPath, computeSha256(zipPath)


# Command line entry point
def main():
    parser = argparse.ArgumentParser(description="Download and prepare the Brackish dataset")
    parser.add_argument("--output", required=True, help="folder that will contain brackish/ (for example underwater/dataset)")
    parser.add_argument("--zip", action="store_true", help="also build brackish.zip for Google Drive (the checksum is printed)")
    arguments = parser.parse_args()
    entry = getManifestEntry(datasetName)
    outputRoot = Path(arguments.output)
    exportFolder = downloadMirror(outputRoot / "downloads", entry["unofficialMirror"]["huggingFaceRepo"])
    yamlPath = arrangeDataset(exportFolder, outputRoot, entry["classNames"])
    print("Ready:", yamlPath)
    if arguments.zip:
        zipPath, checksum = buildZip(outputRoot)
        print(f"Built {zipPath} with SHA-256 {checksum}. Only add the checksum to datasets/manifest.json if the team hosts this zip.")


if __name__ == "__main__":
    main()
