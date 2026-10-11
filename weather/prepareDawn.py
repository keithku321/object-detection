# Convert DAWN to a YOLO-format dataset and write stratified 5-fold lists (plans/02-weather-distill.md, "Splits").
# DAWN is educational-use only: keep the processed copy private and never publish it.
#   python -m weather.prepareDawn --source /content/DAWN_raw --output /content/datasets/dawn
# Layout produced:  <output>/images/<weather>/<image>   <output>/labels/<weather>/<image>.txt
# Fold lists are committed under common/splits/dawn/ (paths relative to the dataset root, so they work on every machine).
# Near-duplicate images (perceptual hash) are kept together in one fold, so a copy of a training image can never sit in validation.
import argparse
import json
import shutil
from pathlib import Path

from common.datasets import (convertVocFile, defaultHashDistance, findNearDuplicateGroups, getManifestEntry, imageSuffixes,
                             makeStratifiedFolds, readSplitFile, splitsFolder, writeAbsoluteImageList, writeDatasetYaml, writeSplitFile)

# Number of cross-validation folds
foldCount = 5
# Seed that fixes the fold assignment; change it only by team agreement, because every number depends on the folds
foldSeed = 0
# Folder for the committed fold lists
dawnSplitsFolder = splitsFolder / "dawn"


# Weather types and class names from the dataset manifest, so both stay defined in one place
def getDawnSettings():
    entry = getManifestEntry("dawn")
    return entry["weatherTypes"], entry["classNames"]


# Folders of one weather type below the raw DAWN folder; folder names are matched case-insensitively ("Fog", "fog", "Foggy")
def findWeatherFolders(sourceRoot, weatherType):
    weatherFolders = [folder for folder in sorted(Path(sourceRoot).iterdir()) if folder.is_dir() and folder.name.lower().startswith(weatherType)]
    if not weatherFolders: raise FileNotFoundError(f"No folder starting with '{weatherType}' in {sourceRoot}")
    return weatherFolders


# Find the images of each weather type below the raw DAWN folder; returns a dict from weather type to a list of image paths
def collectWeatherImages(sourceRoot, weatherTypes):
    return {weatherType: sorted(path for folder in findWeatherFolders(sourceRoot, weatherType) for path in folder.rglob("*") if path.suffix.lower() in imageSuffixes)
            for weatherType in weatherTypes}


# Map each file stem below some folders to its annotation file of one kind (".txt" for YOLO, ".xml" for Pascal VOC)
# The index is built per weather type, so equal file names in two weather folders cannot receive each other's labels
def indexAnnotations(folders, suffix):
    index = {}
    for folder in folders:
        for path in sorted(Path(folder).rglob(f"*{suffix}")): index.setdefault(path.stem, path)
    return index


# Copy images and labels of every weather type into <output>/images/<weather> and <output>/labels/<weather>
# YOLO txt files are copied as they are; if only a Pascal VOC XML exists for an image, it is converted. Images without any
# annotation file are reported and still copied with an empty label file (they count as background images).
def convertDawn(sourceRoot, outputRoot, weatherTypes, classNames):
    outputRoot = Path(outputRoot)
    imagesByWeather = collectWeatherImages(sourceRoot, weatherTypes)
    classNumbers = {className: number for number, className in enumerate(classNames)}
    unknownClasses, missingAnnotations = {}, []
    for weatherType, imagePaths in imagesByWeather.items():
        weatherFolders = findWeatherFolders(sourceRoot, weatherType)
        yoloIndex, xmlIndex = indexAnnotations(weatherFolders, ".txt"), indexAnnotations(weatherFolders, ".xml")
        if len({imagePath.stem for imagePath in imagePaths}) != len(imagePaths): print(f"WARNING: {weatherType} has images with the same file name in different sub-folders; later ones overwrite earlier ones")
        for imagePath in imagePaths:
            targetImage = outputRoot / "images" / weatherType / imagePath.name
            targetLabel = outputRoot / "labels" / weatherType / f"{imagePath.stem}.txt"
            targetImage.parent.mkdir(parents=True, exist_ok=True)
            targetLabel.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(imagePath, targetImage)
            if imagePath.stem in yoloIndex: shutil.copy2(yoloIndex[imagePath.stem], targetLabel)
            elif imagePath.stem in xmlIndex: convertVocFile(xmlIndex[imagePath.stem], targetLabel, classNumbers, unknownClasses)
            else:
                missingAnnotations.append(imagePath.name)
                targetLabel.write_text("", encoding="utf-8")
    if missingAnnotations: print(f"WARNING: {len(missingAnnotations)} images have no annotation file, for example {missingAnnotations[:3]}")
    if unknownClasses: print(f"WARNING: XML files contained classes that are not in classNames: {unknownClasses}")
    return {weatherType: sorted((outputRoot / "images" / weatherType).glob("*")) for weatherType in weatherTypes}


# Count boxes per class id over all label files; compare the result with what you expect to catch a wrong class order
def countClassIds(outputRoot):
    counts = {}
    for labelFile in Path(outputRoot).glob("labels/*/*.txt"):
        for line in labelFile.read_text(encoding="utf-8").splitlines():
            if line.strip(): counts[int(line.split()[0])] = counts.get(int(line.split()[0]), 0) + 1
    return dict(sorted(counts.items()))


# Assign images to folds, stratified by weather, keeping near-duplicate groups together
# Returns a list of foldCount lists of image paths
def assignFolds(imagesByWeather, count=foldCount, seed=foldSeed, hashDistance=defaultHashDistance):
    allImages = [path for paths in imagesByWeather.values() for path in paths]
    weatherOf = {path: weatherType for weatherType, paths in imagesByWeather.items() for path in paths}
    duplicateGroups = findNearDuplicateGroups(allImages, hashDistance)
    groupedPaths = {path for group in duplicateGroups for path in group}
    units = [[path] for path in allImages if path not in groupedPaths] + [list(group) for group in duplicateGroups]
    unitNames = [str(index) for index in range(len(units))]
    unitStrata = [weatherOf[unit[0]] for unit in units]
    foldsOfUnitNames = makeStratifiedFolds(unitNames, unitStrata, count, seed)
    folds = [[path for name in foldUnitNames for path in units[int(name)]] for foldUnitNames in foldsOfUnitNames]
    print(f"{len(allImages)} images in {len(units)} units ({len(duplicateGroups)} near-duplicate groups kept together)")
    return folds


# Write the committed fold lists: common/splits/dawn/fold<k>.txt holds the held-out images of fold k
def writeFoldLists(folds, datasetRoot, folder=dawnSplitsFolder):
    for foldIndex, foldImages in enumerate(folds):
        writeSplitFile(Path(folder) / f"fold{foldIndex}.txt", foldImages, datasetRoot)
    return Path(folder)


# Build, on this machine, the absolute image lists and dataset YAML of every fold from the committed lists
# Fold k validates on fold k and trains on the other folds. Returns a dict from fold index to YAML path.
def buildFoldYamls(datasetRoot, classNames, folder=dawnSplitsFolder, count=foldCount):
    datasetRoot = Path(datasetRoot)
    heldOut = [readSplitFile(Path(folder) / f"fold{foldIndex}.txt", datasetRoot) for foldIndex in range(count)]
    yamlByFold = {}
    for foldIndex in range(count):
        trainImages = [path for otherIndex in range(count) if otherIndex != foldIndex for path in heldOut[otherIndex]]
        trainList = writeAbsoluteImageList(datasetRoot / "folds" / f"fold{foldIndex}_train.txt", trainImages)
        valList = writeAbsoluteImageList(datasetRoot / "folds" / f"fold{foldIndex}_val.txt", heldOut[foldIndex])
        yamlByFold[foldIndex] = writeDatasetYaml(datasetRoot, classNames, trainSplit=f"folds/{trainList.name}", valSplit=f"folds/{valList.name}", yamlName=f"dawn-fold{foldIndex}.yaml")
    return yamlByFold


# Held-out images of one fold grouped by weather type, used by the per-weather evaluation
def heldOutImagesByWeather(datasetRoot, foldIndex, weatherTypes, folder=dawnSplitsFolder):
    imagesByWeather = {weatherType: [] for weatherType in weatherTypes}
    for path in readSplitFile(Path(folder) / f"fold{foldIndex}.txt", datasetRoot):
        imagesByWeather[path.parent.name].append(path)
    return imagesByWeather


# Command line entry point: convert the raw download, write fold lists and per-fold YAML files
def main():
    parser = argparse.ArgumentParser(description="Prepare DAWN: convert to YOLO format and write 5-fold lists")
    parser.add_argument("--source", required=True, help="folder with the raw DAWN weather folders")
    parser.add_argument("--output", required=True, help="folder for the YOLO-format copy (keep it private)")
    arguments = parser.parse_args()
    weatherTypes, classNames = getDawnSettings()
    imagesByWeather = convertDawn(arguments.source, arguments.output, weatherTypes, classNames)
    print("Boxes per class id (verify against the class order in plans/02-weather-distill.md):", json.dumps(countClassIds(arguments.output)))
    folds = assignFolds(imagesByWeather)
    writeFoldLists(folds, arguments.output)
    print(buildFoldYamls(arguments.output, classNames))
    for foldIndex, foldImages in enumerate(folds): print(f"fold {foldIndex}: {len(foldImages)} images")


if __name__ == "__main__":
    main()
