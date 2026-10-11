# Dataset helpers shared by every task: manifest lookup, download and checksum, YOLO label checks, VOC to YOLO
# conversion, near-duplicate detection and k-fold lists.
# Steward: plan A owner (plans/05-team-workflow.md). The repo ships scripts and checksums, never third-party images.
import hashlib
import json
import os
import random
import urllib.request
import zipfile
from pathlib import Path
from xml.etree import ElementTree

# Repository root, so paths work from any working directory
repositoryRoot = Path(__file__).resolve().parent.parent
# Folder for local dataset copies; ignored by git except for the manifest
datasetsRootFolder = repositoryRoot / "datasets"
# Source, licence and checksum of every dataset
manifestPath = datasetsRootFolder / "manifest.json"
# Folder for committed file lists of splits and folds
splitsFolder = Path(__file__).resolve().parent / "splits"
# File endings that count as images
imageSuffixes = {".jpg", ".jpeg", ".png", ".bmp"}
# Bytes read at a time when hashing or downloading
chunkSize = 1 << 20
# Largest Hamming distance between two 64-bit perceptual hashes that still counts as a near-duplicate
defaultHashDistance = 4
# Environment variable that holds the Hugging Face token (set it from Colab Secrets, never write it in a notebook)
huggingFaceTokenVariable = "HF_TOKEN"


# Read datasets/manifest.json and return its entries as a dict keyed by dataset name
def loadManifest(path=manifestPath):
    entries = json.loads(Path(path).read_text(encoding="utf-8"))["datasets"]
    return {entry["name"]: entry for entry in entries}


# Return the manifest entry of one dataset, with a helpful error for a misspelled name
def getManifestEntry(datasetName, path=manifestPath):
    manifest = loadManifest(path)
    if datasetName not in manifest: raise KeyError(f"Unknown dataset '{datasetName}'. Known datasets: {sorted(manifest)}")
    return manifest[datasetName]


# SHA-256 checksum of a file as a lowercase hex string
def computeSha256(filePath):
    digest = hashlib.sha256()
    with Path(filePath).open("rb") as file:
        for block in iter(lambda: file.read(chunkSize), b""):
            digest.update(block)
    return digest.hexdigest()


# Compare a file with its expected checksum; returns True or False, or None when no checksum is recorded yet
def verifySha256(filePath, expectedHash):
    if not expectedHash:
        print(f"No SHA-256 recorded for {Path(filePath).name}; its checksum is {computeSha256(filePath)}. Add it to datasets/manifest.json.")
        return None
    actualHash = computeSha256(filePath)
    if actualHash.lower() != expectedHash.lower():
        print(f"CHECKSUM MISMATCH for {Path(filePath).name}: expected {expectedHash}, got {actualHash}")
        return False
    return True


# Download one URL to a file, showing progress, and verify the checksum when one is given
def downloadFile(url, targetPath, expectedHash=None):
    targetPath = Path(targetPath)
    targetPath.parent.mkdir(parents=True, exist_ok=True)
    if targetPath.exists() and expectedHash and verifySha256(targetPath, expectedHash):
        print(f"{targetPath.name} is already downloaded and verified")
        return targetPath
    temporaryPath = targetPath.with_suffix(targetPath.suffix + ".part")
    with urllib.request.urlopen(url) as response, temporaryPath.open("wb") as file:
        totalBytes = int(response.headers.get("Content-Length", 0))
        receivedBytes = 0
        for block in iter(lambda: response.read(chunkSize), b""):
            file.write(block)
            receivedBytes += len(block)
            if totalBytes: print(f"\r{targetPath.name}: {receivedBytes / 1e6:.0f} / {totalBytes / 1e6:.0f} MB", end="")
    print()
    temporaryPath.replace(targetPath)
    if verifySha256(targetPath, expectedHash) is False: raise ValueError(f"Checksum mismatch for {targetPath}")
    return targetPath


# Download one file from a Hugging Face repo (datasets are private by default); the token comes from the argument or HF_TOKEN
def downloadFromHuggingFace(repoId, fileName, targetFolder, repoType="dataset", token=None):
    from huggingface_hub import hf_hub_download
    token = token or os.environ.get(huggingFaceTokenVariable)
    localPath = hf_hub_download(repo_id=repoId, filename=fileName, repo_type=repoType, token=token, local_dir=str(targetFolder))
    return Path(localPath)


# Unzip an archive into a folder, refusing any member that would be written outside that folder
def extractZip(zipPath, targetFolder):
    targetFolder = Path(targetFolder).resolve()
    targetFolder.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zipPath) as archive:
        for member in archive.namelist():
            if not (targetFolder / member).resolve().is_relative_to(targetFolder): raise ValueError(f"Unsafe path in zip: {member}")
        archive.extractall(targetFolder)
    return targetFolder


# Get a dataset as described by its manifest entry and return the local folder (or the Ultralytics YAML name)
# access "huggingface": download the zip from the Hugging Face dataset repo, verify it and unzip it
# access "url": download the zip from a direct link, verify it and unzip it
# access "ultralytics": Ultralytics downloads and converts the dataset itself on first use of the named YAML
# access "manual": the licence or the host needs a person to download the files; the entry's instructions are printed
def fetchDataset(datasetName, targetRoot=datasetsRootFolder, token=None, path=manifestPath):
    entry = getManifestEntry(datasetName, path)
    access = entry.get("access")
    targetRoot = Path(targetRoot)
    if access == "ultralytics":
        print(f"{datasetName}: Ultralytics downloads this dataset when you train with data='{entry['ultralyticsYaml']}'")
        return entry["ultralyticsYaml"]
    if access == "manual" or not (entry.get("huggingFaceRepo") or entry.get("downloadUrl")):
        raise RuntimeError(f"{datasetName} cannot be downloaded automatically yet.\n{entry.get('instructions', 'See the dataset page in datasets/manifest.json.')}")
    zipPath = targetRoot / "downloads" / entry["zipName"]
    if access == "huggingface":
        zipPath = downloadFromHuggingFace(entry["huggingFaceRepo"], entry["zipName"], zipPath.parent, token=token)
        if verifySha256(zipPath, entry.get("sha256")) is False: raise ValueError(f"Checksum mismatch for {zipPath}")
    else:
        downloadFile(entry["downloadUrl"], zipPath, entry.get("sha256"))
    return extractZip(zipPath, targetRoot)


# Write a dataset.yaml for a folder that contains images/<split> and labels/<split>
# Splits may also be .txt files that list image paths (used for k-fold lists); the test split is optional
def writeDatasetYaml(rootPath, classNames, trainSplit="images/train", valSplit="images/val", testSplit=None, yamlName="dataset.yaml"):
    rootPath = Path(rootPath)
    yamlPath = rootPath / yamlName
    lines = [f"path: {rootPath.as_posix()}", f"train: {Path(trainSplit).as_posix()}", f"val: {Path(valSplit).as_posix()}"]
    if testSplit: lines.append(f"test: {Path(testSplit).as_posix()}")
    lines.append("names:")
    for classNumber, className in enumerate(classNames):
        lines.append(f"  {classNumber}: {className}")
    yamlPath.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return yamlPath


# Read an image as a BGR array; works for Windows paths with spaces or non-ASCII characters, where cv2.imread can fail
def readImageBgr(imagePath):
    import cv2
    import numpy as np
    image = cv2.imdecode(np.fromfile(str(imagePath), dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None: raise ValueError(f"Could not read image {imagePath}")
    return image


# Write a BGR array as an image file; the file ending of the path picks the format
def writeImageBgr(imagePath, image):
    import cv2
    wasEncoded, encoded = cv2.imencode(Path(imagePath).suffix or ".png", image)
    if not wasEncoded: raise ValueError(f"Could not encode image for {imagePath}")
    Path(imagePath).parent.mkdir(parents=True, exist_ok=True)
    encoded.tofile(str(imagePath))
    return Path(imagePath)


# Roboflow exports use <split>/images and <split>/labels with the split names train, valid and test; this project uses
# images/<split> and labels/<split> with train, val and test. Move the folders into the project layout; returns the splits it moved.
def convertRoboflowLayout(rootPath):
    rootPath = Path(rootPath)
    movedSplits = []
    for roboflowName, splitName in (("train", "train"), ("valid", "val"), ("test", "test")):
        for kind in ("images", "labels"):
            sourceFolder = rootPath / roboflowName / kind
            if not sourceFolder.exists(): continue
            targetFolder = rootPath / kind / splitName
            targetFolder.parent.mkdir(parents=True, exist_ok=True)
            sourceFolder.rename(targetFolder)
            if splitName not in movedSplits: movedSplits.append(splitName)
    return movedSplits


# Image files directly inside a folder
def listImages(folderPath):
    return sorted(file for file in Path(folderPath).glob("*") if file.suffix.lower() in imageSuffixes)


# Count the images and label files of one split and warn about empty folders
# A missing label file is allowed because an image with no objects needs no label file
def checkSplit(rootPath, splitName):
    imageFolder = Path(rootPath) / "images" / splitName
    labelFolder = Path(rootPath) / "labels" / splitName
    imageFiles = listImages(imageFolder)
    labelFiles = list(labelFolder.glob("*.txt"))
    print(f"{splitName}: {len(imageFiles)} images, {len(labelFiles)} label files")
    if len(imageFiles) == 0: print(f"  WARNING: no images found in {imageFolder}")
    if len(labelFiles) == 0: print(f"  WARNING: no labels found in {labelFolder}")
    return len(imageFiles), len(labelFiles)


# Find problems in the label files of one split: wrong number of columns, class numbers out of range, values outside 0..1
# Returns a list of readable problem strings; an empty list means the labels look fine
def findLabelProblems(rootPath, splitName, classCount, maximumProblems=50):
    problems = []
    for labelFile in sorted((Path(rootPath) / "labels" / splitName).glob("*.txt")):
        for lineNumber, line in enumerate(labelFile.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip(): continue
            parts = line.split()
            if len(parts) != 5:
                problems.append(f"{labelFile.name}:{lineNumber} has {len(parts)} columns, expected 5 (class x y w h)")
            else:
                try:
                    classNumber, *coordinates = int(parts[0]), *map(float, parts[1:])
                except ValueError:
                    problems.append(f"{labelFile.name}:{lineNumber} has a non-numeric value")
                    continue
                if not 0 <= classNumber < classCount: problems.append(f"{labelFile.name}:{lineNumber} class {classNumber} is out of range 0..{classCount - 1}")
                if any(not 0.0 <= value <= 1.0 for value in coordinates): problems.append(f"{labelFile.name}:{lineNumber} has a coordinate outside 0..1")
            if len(problems) >= maximumProblems: return problems + [f"... stopped after {maximumProblems} problems"]
    return problems


# Print the label problems of one split; an out-of-range class number usually means classes were ordered differently
def checkClassNumbers(rootPath, splitName, classCount):
    problems = findLabelProblems(rootPath, splitName, classCount)
    for problem in problems: print(f"  WARNING: {problem}")
    if not problems: print(f"  {splitName}: labels look fine")
    return problems


# Convert one Pascal VOC XML file to a YOLO label file; boxes of classes missing from classNumbers are skipped and counted in unknownClasses
def convertVocFile(xmlFile, labelFile, classNumbers, unknownClasses):
    root = ElementTree.parse(xmlFile).getroot()
    width, height = float(root.findtext("size/width")), float(root.findtext("size/height"))
    lines = []
    for objectNode in root.iter("object"):
        className = objectNode.findtext("name", default="").strip()
        if className not in classNumbers:
            unknownClasses[className] = unknownClasses.get(className, 0) + 1
            continue
        xMin, yMin, xMax, yMax = (float(objectNode.findtext(f"bndbox/{side}")) for side in ("xmin", "ymin", "xmax", "ymax"))
        xCentre, yCentre = (xMin + xMax) / 2 / width, (yMin + yMax) / 2 / height
        boxWidth, boxHeight = (xMax - xMin) / width, (yMax - yMin) / height
        lines.append(f"{classNumbers[className]} {xCentre:.6f} {yCentre:.6f} {boxWidth:.6f} {boxHeight:.6f}")
    Path(labelFile).parent.mkdir(parents=True, exist_ok=True)
    Path(labelFile).write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


# Convert a folder of Pascal VOC XML annotations (URPC, DUO and similar) to YOLO label files and return the unknown class names with counts
# classNames fixes the class order; boxes of unknown classes are skipped and counted
def convertVocXmlToYolo(xmlFolder, labelFolder, classNames):
    classNumbers = {className: number for number, className in enumerate(classNames)}
    unknownClasses = {}
    for xmlFile in sorted(Path(xmlFolder).glob("*.xml")):
        convertVocFile(xmlFile, Path(labelFolder) / f"{xmlFile.stem}.txt", classNumbers, unknownClasses)
    return unknownClasses


# 64-bit perceptual hash of an image as an integer
def computePerceptualHash(imagePath):
    import imagehash
    from PIL import Image
    with Image.open(imagePath) as image:
        return int(str(imagehash.phash(image)), 16)


# Number of differing bits between one hash and many hashes (numpy array of uint64)
def hammingDistances(oneHash, manyHashes):
    import numpy as np
    differences = np.bitwise_xor(manyHashes, np.uint64(oneHash))
    if hasattr(np, "bitwise_count"): return np.bitwise_count(differences)
    return np.unpackbits(differences.view(np.uint8).reshape(len(differences), 8), axis=1).sum(axis=1)


# Group images that are near-duplicates of each other (neighbouring video frames, repeated web images)
# Returns a list of groups with at least two image paths; a path appears in at most one group
def findNearDuplicateGroups(imagePaths, maximumDistance=defaultHashDistance):
    import numpy as np
    imagePaths = list(imagePaths)
    hashes = np.array([computePerceptualHash(path) for path in imagePaths], dtype=np.uint64)
    parent = list(range(len(imagePaths)))

    # Find the group leader of an image index, flattening the chain as it goes
    def findLeader(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    for index in range(len(imagePaths) - 1):
        closeOnes = np.nonzero(hammingDistances(int(hashes[index]), hashes[index + 1:]) <= maximumDistance)[0] + index + 1
        for otherIndex in closeOnes:
            parent[findLeader(int(otherIndex))] = findLeader(index)
    groups = {}
    for index, path in enumerate(imagePaths):
        groups.setdefault(findLeader(index), []).append(path)
    return [group for group in groups.values() if len(group) > 1]


# Near-duplicate groups whose images sit in more than one split; these inflate scores and must be fixed before training
# imagePathsBySplit maps a split name to its image paths
def findCrossSplitDuplicates(imagePathsBySplit, maximumDistance=defaultHashDistance):
    splitOfPath = {path: splitName for splitName, paths in imagePathsBySplit.items() for path in paths}
    groups = findNearDuplicateGroups(list(splitOfPath), maximumDistance)
    return [group for group in groups if len({splitOfPath[path] for path in group}) > 1]


# Split items into stratified folds: every stratum (for example a weather type) is spread evenly over the folds
# Returns a list of foldCount lists of items; the same seed always gives the same folds
def makeStratifiedFolds(items, strata, foldCount=5, seed=0):
    if len(items) != len(strata): raise ValueError("items and strata must have the same length")
    randomGenerator = random.Random(seed)
    itemsByStratum = {}
    for item, stratum in zip(items, strata):
        itemsByStratum.setdefault(stratum, []).append(item)
    folds = [[] for _ in range(foldCount)]
    nextFold = 0
    for stratum in sorted(itemsByStratum):
        stratumItems = sorted(itemsByStratum[stratum])
        randomGenerator.shuffle(stratumItems)
        for item in stratumItems:
            folds[nextFold].append(item)
            nextFold = (nextFold + 1) % foldCount
    return [sorted(fold) for fold in folds]


# Write a list of image paths to a text file, relative to a root with forward slashes, so lists work on Windows and Linux
def writeSplitFile(listPath, imagePaths, rootPath):
    listPath = Path(listPath)
    listPath.parent.mkdir(parents=True, exist_ok=True)
    relativePaths = sorted(Path(path).resolve().relative_to(Path(rootPath).resolve()).as_posix() for path in imagePaths)
    listPath.write_text("\n".join(relativePaths) + "\n", encoding="utf-8")
    return listPath


# Read a split list written by writeSplitFile and return absolute image paths below the given root
def readSplitFile(listPath, rootPath):
    lines = Path(listPath).read_text(encoding="utf-8").splitlines()
    return [Path(rootPath) / line for line in lines if line.strip()]


# Write an absolute-path image list that Ultralytics accepts as a train or val entry; labels are found by swapping /images/ for /labels/
def writeAbsoluteImageList(listPath, imagePaths):
    listPath = Path(listPath)
    listPath.parent.mkdir(parents=True, exist_ok=True)
    listPath.write_text("\n".join(Path(path).resolve().as_posix() for path in sorted(imagePaths)) + "\n", encoding="utf-8")
    return listPath
