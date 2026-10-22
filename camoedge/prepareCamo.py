# Download COD10K and CAMO in the layout of the standard protocol (the release of the SINet authors), unzip them and check the counts.
#   python -m camoedge.prepareCamo --output camoedge/dataset
# Produces <output>/camo/TrainDataset/{Imgs,GT,Edge} (COD10K-Tr 3,040 + CAMO-Tr 1,000 = 4,040 pairs) and <output>/camo/TestDataset/<COD10K|CAMO|CHAMELEON>,
# which is what camoedge/train.py --dataset <output>/camo reads (CHAMELEON is unpacked but not used). The two zips (1.1 GB and 0.7 GB) come from the
# Google Drive links of the SINet repository and are checked against the SHA-256 values recorded in datasets/manifest.json (entry "camoedge").
# COD10K is for non-commercial research only: keep every copy private and never commit it (the folder is git-ignored).
# Google Drive refuses a file for about a day after many downloads; the script then stops with the manual links (datasets/README.md).
import argparse
from pathlib import Path

from camoedge.dataset import collectTestPairs, collectTrainingPairs
from common.datasets import extractZip, getManifestEntry, verifySha256

# Name of the dataset entry in datasets/manifest.json that lists the Google Drive files
datasetName = "camoedge"
# Name of the folder inside the output folder that holds TrainDataset/ and TestDataset/
datasetFolderName = "camo"
# Pairs that the standard protocol expects: training pairs and pairs of each test set
expectedCounts = {"train": 4040, "COD10K": 2026, "CAMO": 250}


# Download one Google Drive file with gdown (large Drive files need its confirmation handling); returns the path of the file
def downloadDriveFile(googleDriveId, targetPath):
    try:
        import gdown
    except ImportError as error:
        raise SystemExit("gdown is missing: pip install -r requirements.txt (it lists gdown), then run this again") from error
    targetPath.parent.mkdir(parents=True, exist_ok=True)
    downloadedPath = gdown.download(id=googleDriveId, output=str(targetPath), quiet=False)
    if not downloadedPath:
        raise SystemExit(f"Google Drive refused {targetPath.name} (it limits downloads of popular files for about a day). Download it by hand from "
                         f"datasets/README.md (COD10K section), put it at {targetPath} and run this again.")
    return Path(downloadedPath)


# Count training pairs and test pairs per test set and compare them with the protocol; returns the counts and raises on a mismatch
def checkCounts(datasetRoot):
    counts = {"train": len(collectTrainingPairs(datasetRoot)), **{name: len(pairs) for name, pairs in collectTestPairs(datasetRoot).items()}}
    wrongCounts = {name: (counts.get(name, 0), expected) for name, expected in expectedCounts.items() if counts.get(name, 0) != expected}
    if wrongCounts: raise ValueError(f"Unexpected numbers of image and mask pairs (found, expected): {wrongCounts}")
    return counts


# Download (or reuse) and verify every zip of the manifest entry, unzip them into <output>/camo and check the counts; returns the dataset folder
def prepareCamo(outputRoot):
    outputRoot = Path(outputRoot)
    datasetRoot = outputRoot / datasetFolderName
    for fileEntry in getManifestEntry(datasetName)["googleDriveFiles"]:
        zipPath = outputRoot / "downloads" / fileEntry["name"]
        if not (zipPath.exists() and verifySha256(zipPath, fileEntry["sha256"])):
            downloadDriveFile(fileEntry["googleDriveId"], zipPath)
            if verifySha256(zipPath, fileEntry["sha256"]) is False: raise ValueError(f"{zipPath} does not match the checksum in datasets/manifest.json")
        print(f"Unzipping {zipPath.name} into {datasetRoot}")
        extractZip(zipPath, datasetRoot)
    print("Pairs:", checkCounts(datasetRoot))
    return datasetRoot


# Command line entry point
def main():
    parser = argparse.ArgumentParser(description="Download and check COD10K and CAMO for Plan D")
    parser.add_argument("--output", default="camoedge/dataset", help="folder that will contain camo/ and downloads/ (git-ignored)")
    arguments = parser.parse_args()
    print("Ready:", prepareCamo(arguments.output))


if __name__ == "__main__":
    main()
