# Download the Brackish dataset, give it the layout that underwater_colab.ipynb and underwater/train.py expect, and check it.
#   python -m underwater.prepareBrackish --output underwater/dataset
#   python -m underwater.prepareBrackish --output underwater/dataset --zip      (also builds brackish.zip for Google Drive or Hugging Face)
# The data comes from the unofficial CC BY 4.0 copy named in datasets/manifest.json (unofficialMirror), which is the Roboflow export of
# the Aalborg University dataset. Attribute Pedersen et al., Aalborg University, when you use or show it (see the NOTICE file).
# The original 80/10/10 split cuts the same video clips into all three splits, so a validation frame has near-identical neighbours in
# training (decision D13). By default the clips are therefore re-split as whole units, again about 80/10/10; the original assignment is
# saved as common/splits/brackish/original-<split>.txt and the new one as <split>.txt. --keep-original-split skips the re-split.
# The copy keeps the training split in shard folders (train/shard_000/images, ...); they are merged into one folder first.
# Layout produced:  <output>/brackish/images/<train|val|test>   <output>/brackish/labels/<train|val|test>   <output>/brackish/dataset.yaml
# Frames of one clip still look alike across clips of the same day (a fixed camera), so quote accuracy with that caveat.
import argparse
import collections
import random
import re
import shutil
from pathlib import Path

import yaml

from common.datasets import (checkClassNumbers, checkSplit, computeSha256, convertRoboflowLayout, getManifestEntry, imageSuffixes,
                             listImages, splitsFolder, writeDatasetYaml, writeSplitFile)

# Name of the dataset in datasets/manifest.json, and of the folder that holds it inside the output folder
datasetName = "brackish"
# Name of the file in the Roboflow export that lists the class names
sourceYamlName = "data.yaml"
# Share of the images that goes to the validation and to the test split when the clips are re-split; the rest is training data
holdOutFraction = 0.1
# Number of random clip orders that are tried; the one whose validation and test splits mirror the class mix best is kept
candidateSeedCount = 500
# Folder for the committed split lists
brackishSplitsFolder = splitsFolder / "brackish"
# File names look like <clip start>to<clip end>_<part>-<frame>_jpg.rf.<hash>.jpg; clip start, end and part identify one video clip
clipPattern = re.compile(r"^(?P<clip>.+?)_(?P<part>\d+)-(?P<frame>\d+)_jpg")


# Download the files of the unofficial copy into a folder and return the folder that holds the Roboflow export (its data/ sub-folder)
def downloadMirror(downloadFolder, mirrorRepo):
    from huggingface_hub import snapshot_download
    snapshot_download(repo_id=mirrorRepo, repo_type="dataset", local_dir=str(downloadFolder), allow_patterns=["data/**"], max_workers=8)
    return Path(downloadFolder) / "data"


# Merge the shard folders of one split (<split>/shard_000/images and labels, ...) into <split>/images and <split>/labels
# Only images and .txt label files are moved. Every shard also holds a metadata.jsonl of the Hugging Face loader, which is not data and is
# deleted with the empty shard; any other leftover file stops the script, as does a file name that occurs in two shards (it would overwrite
# an image). Returns the number of files moved
def mergeShards(splitFolder):
    movedCount = 0
    for shardFolder in sorted(Path(splitFolder).glob("shard_*")):
        for kind, suffixes in (("images", imageSuffixes), ("labels", {".txt"})):
            targetFolder = Path(splitFolder) / kind
            targetFolder.mkdir(exist_ok=True)
            for sourceFile in (shardFolder / kind).glob("*"):
                if sourceFile.suffix.lower() not in suffixes: continue
                if (targetFolder / sourceFile.name).exists(): raise FileExistsError(f"{sourceFile.name} occurs in two shards of {splitFolder}")
                sourceFile.rename(targetFolder / sourceFile.name)
                movedCount += 1
        leftovers = [file for file in shardFolder.rglob("*") if file.is_file() and file.suffix != ".jsonl"]
        if leftovers: raise ValueError(f"Unexpected files left in {shardFolder}, for example {leftovers[:3]}")
        shutil.rmtree(shardFolder)
    return movedCount


# Compare the class names in the export with the manifest; a different order would silently mislabel every box, so it stops the script
def checkClassOrder(sourceYamlPath, classNames):
    sourceNames = yaml.safe_load(Path(sourceYamlPath).read_text(encoding="utf-8"))["names"]
    sourceNames = [sourceNames[key] for key in sorted(sourceNames)] if isinstance(sourceNames, dict) else list(sourceNames)
    if sourceNames != list(classNames): raise ValueError(f"Class order of the download {sourceNames} differs from the manifest {list(classNames)}")


# Move the Roboflow export (train/, valid/, test/) to <outputRoot>/brackish, rename it to images/<split> and labels/<split>, write dataset.yaml
# and run the label checks of common/datasets.py; returns the path of dataset.yaml
def arrangeDataset(exportFolder, outputRoot, classNames, splitByClip=False, listFolder=brackishSplitsFolder):
    datasetRoot = Path(outputRoot) / datasetName
    if datasetRoot.exists(): raise FileExistsError(f"{datasetRoot} already exists; delete it to prepare the dataset again")
    checkClassOrder(Path(exportFolder) / sourceYamlName, classNames)
    datasetRoot.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(exportFolder), str(datasetRoot))
    mergeShards(datasetRoot / "train")
    movedSplits = convertRoboflowLayout(datasetRoot)
    for metadataFile in (datasetRoot / "images").rglob("metadata.jsonl"): metadataFile.unlink()
    for oldYamlName in (sourceYamlName, "detectionbench_data.yaml"): (datasetRoot / oldYamlName).unlink(missing_ok=True)
    for roboflowName in ("train", "valid", "test"):
        leftoverFolder = datasetRoot / roboflowName
        if leftoverFolder.exists() and not any(file.is_file() for file in leftoverFolder.rglob("*")): shutil.rmtree(leftoverFolder)
    if not {"train", "val"} <= set(movedSplits): raise ValueError(f"The download has no train and valid splits (found {movedSplits})")
    yamlPath = writeDatasetYaml(datasetRoot, classNames, testSplit="images/test" if "test" in movedSplits else None)
    for splitName in movedSplits:
        imageCount, labelCount = checkSplit(datasetRoot, splitName)
        if imageCount == 0: raise ValueError(f"The {splitName} split has no images")
        checkClassNumbers(datasetRoot, splitName, len(classNames))
    if splitByClip:
        summary = regroupByClip(datasetRoot, len(classNames), listFolder)
        print(f"Re-split {summary['clips']} clips with seed {summary['seed']}: images {summary['images']}")
        for splitName, boxCounts in summary["boxes"].items(): print(f"  {splitName} boxes per class {dict(zip(classNames, boxCounts))}")
    return yamlPath


# Name of the video clip that an image comes from; a name that does not follow the pattern stops the script, because it cannot be grouped
def getClipName(imageName):
    match = clipPattern.match(imageName)
    if not match: raise ValueError(f"Cannot read the video clip from the image name {imageName}")
    return f"{match['clip']}_{match['part']}"


# Per clip: its images (as paths) and how many boxes of each class its label files hold; also returns the current split of every image
def collectClipStatistics(datasetRoot, classCount):
    imagesByClip, boxesByClip, splitOfImage = collections.defaultdict(list), collections.defaultdict(lambda: [0] * classCount), {}
    for splitName in ("train", "val", "test"):
        for imagePath in listImages(Path(datasetRoot) / "images" / splitName):
            clipName = getClipName(imagePath.name)
            imagesByClip[clipName].append(imagePath)
            splitOfImage[imagePath] = splitName
            labelPath = Path(datasetRoot) / "labels" / splitName / f"{imagePath.stem}.txt"
            if labelPath.exists():
                for line in labelPath.read_text(encoding="utf-8").splitlines():
                    if line.strip(): boxesByClip[clipName][int(line.split()[0])] += 1
    return imagesByClip, boxesByClip, splitOfImage


# Assign every clip to a split for one random order: validation and test are filled from the front of the shuffled clips until each holds
# its share of the images, and the remaining clips are training data. Returns a dict from clip name to split name
def assignClipsToSplits(imageCountByClip, seed):
    clipNames = sorted(imageCountByClip)
    random.Random(seed).shuffle(clipNames)
    totalImages = sum(imageCountByClip.values())
    assignment, filledImages = {}, {"val": 0, "test": 0}
    for clipName in clipNames:
        openSplit = next((splitName for splitName in ("val", "test") if filledImages[splitName] < holdOutFraction * totalImages), None)
        assignment[clipName] = openSplit or "train"
        if openSplit: filledImages[openSplit] += imageCountByClip[clipName]
    return assignment


# How far the class mix of validation and test is from the overall class mix: the sum over classes of the difference between the
# share of that class's boxes held out and the wanted share, for both splits (smaller is better; a class missing from a split is a large error)
def scoreAssignment(assignment, boxesByClip, imageCountByClip, classCount):
    totalBoxes = [sum(boxesByClip[clipName][classId] for clipName in boxesByClip) for classId in range(classCount)]
    totalImages = sum(imageCountByClip.values())
    score = 0.0
    for splitName in ("val", "test"):
        wantedShare = sum(imageCountByClip[clipName] for clipName, assigned in assignment.items() if assigned == splitName) / totalImages
        for classId in range(classCount):
            splitBoxes = sum(boxesByClip[clipName][classId] for clipName, assigned in assignment.items() if assigned == splitName)
            score += abs(splitBoxes / max(totalBoxes[classId], 1) - wantedShare)
    return score


# Write the image lists of the current splits (relative to the dataset root) as <prefix><split>.txt in the split list folder
def writeSplitLists(datasetRoot, listFolder, prefix=""):
    for splitName in ("train", "val", "test"):
        writeSplitFile(Path(listFolder) / f"{prefix}{splitName}.txt", listImages(Path(datasetRoot) / "images" / splitName), datasetRoot)


# Re-split the dataset so that every video clip lies completely in one split. The original lists are saved first. Images and labels are
# moved between the split folders; returns a summary with the chosen seed, images per split and boxes per class and split
def regroupByClip(datasetRoot, classCount, listFolder=brackishSplitsFolder):
    imagesByClip, boxesByClip, splitOfImage = collectClipStatistics(datasetRoot, classCount)
    writeSplitLists(datasetRoot, listFolder, prefix="original-")
    imageCountByClip = {clipName: len(imagePaths) for clipName, imagePaths in imagesByClip.items()}
    bestSeed = min(range(candidateSeedCount), key=lambda seed: scoreAssignment(assignClipsToSplits(imageCountByClip, seed), boxesByClip, imageCountByClip, classCount))
    assignment = assignClipsToSplits(imageCountByClip, bestSeed)
    for clipName, imagePaths in imagesByClip.items():
        for imagePath in imagePaths:
            oldSplit, newSplit = splitOfImage[imagePath], assignment[clipName]
            if oldSplit == newSplit: continue
            for kind, fileName in (("images", imagePath.name), ("labels", f"{imagePath.stem}.txt")):
                sourcePath = Path(datasetRoot) / kind / oldSplit / fileName
                if sourcePath.exists(): sourcePath.rename(Path(datasetRoot) / kind / newSplit / fileName)
    writeSplitLists(datasetRoot, listFolder)
    summary = {"seed": bestSeed, "clips": len(imagesByClip), "images": {}, "boxes": {}}
    for splitName in ("train", "val", "test"):
        summary["images"][splitName] = sum(len(imagesByClip[clipName]) for clipName, assigned in assignment.items() if assigned == splitName)
        summary["boxes"][splitName] = [sum(boxesByClip[clipName][classId] for clipName, assigned in assignment.items() if assigned == splitName) for classId in range(classCount)]
    return summary


# Zip the prepared dataset with brackish/ as the single top-level folder, which is what the notebook extracts; returns (zip path, SHA-256)
def buildZip(outputRoot):
    zipBase = Path(outputRoot) / datasetName
    zipPath = Path(shutil.make_archive(str(zipBase), "zip", root_dir=str(outputRoot), base_dir=datasetName))
    return zipPath, computeSha256(zipPath)


# Command line entry point
def main():
    parser = argparse.ArgumentParser(description="Download and prepare the Brackish dataset")
    parser.add_argument("--output", required=True, help="folder that will contain brackish/ (for example underwater/dataset)")
    parser.add_argument("--keep-original-split", action="store_true", help="keep the Roboflow split, which shares video clips between train, val and test")
    parser.add_argument("--zip", action="store_true", help="also build brackish.zip for Google Drive (the checksum is printed)")
    arguments = parser.parse_args()
    entry = getManifestEntry(datasetName)
    outputRoot = Path(arguments.output).resolve()
    exportFolder = downloadMirror(outputRoot / "downloads", entry["unofficialMirror"]["huggingFaceRepo"])
    yamlPath = arrangeDataset(exportFolder, outputRoot, entry["classNames"], splitByClip=not arguments.keep_original_split)
    print("Ready:", yamlPath)
    if arguments.zip:
        zipPath, checksum = buildZip(outputRoot)
        print(f"Built {zipPath} with SHA-256 {checksum}. Only add the checksum to datasets/manifest.json if the team hosts this zip.")


if __name__ == "__main__":
    main()
