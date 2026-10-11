# Classical underwater image enhancement for experiment U1: grey-world white balance followed by CLAHE.
# The thesis check compares "classical enhancement + plain YOLO26" with "attention model, no enhancement".
import shutil
from pathlib import Path

import cv2
import numpy as np

from common.datasets import listImages, readImageBgr, writeImageBgr

# CLAHE contrast limit; higher values give stronger local contrast and more noise
claheClipLimit = 2.0
# CLAHE tile grid; the image is split into this many tiles per side and each tile is equalised separately
claheTileGridSize = (8, 8)


# Grey-world white balance: scale each colour channel so that all three channel means become equal
# Underwater images lose red first, so this lifts the red channel and removes the blue-green cast
def greyWorldWhiteBalance(imageBgr):
    image = imageBgr.astype(np.float32)
    channelMeans = image.reshape(-1, 3).mean(axis=0)
    gains = channelMeans.mean() / np.maximum(channelMeans, 1e-6)
    return np.clip(image * gains, 0, 255).astype(np.uint8)


# CLAHE on the lightness channel only, so colours are not shifted
def applyClahe(imageBgr, clipLimit=claheClipLimit, tileGridSize=claheTileGridSize):
    lightness, greenRed, blueYellow = cv2.split(cv2.cvtColor(imageBgr, cv2.COLOR_BGR2LAB))
    equaliser = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)
    return cv2.cvtColor(cv2.merge((equaliser.apply(lightness), greenRed, blueYellow)), cv2.COLOR_LAB2BGR)


# The full classical pipeline used in experiment U1
def enhanceImage(imageBgr):
    return applyClahe(greyWorldWhiteBalance(imageBgr))


# Write an enhanced copy of a YOLO dataset: images of the chosen splits are enhanced, label folders are copied unchanged
# Splits that do not exist in the source are skipped. Returns the target root folder.
def enhanceDataset(sourceRoot, targetRoot, splitNames=("train", "val", "test")):
    sourceRoot, targetRoot = Path(sourceRoot), Path(targetRoot)
    for splitName in splitNames:
        sourceImages = sourceRoot / "images" / splitName
        if not sourceImages.exists(): continue
        for imagePath in listImages(sourceImages):
            writeImageBgr(targetRoot / "images" / splitName / imagePath.name, enhanceImage(readImageBgr(imagePath)))
        sourceLabels = sourceRoot / "labels" / splitName
        if sourceLabels.exists(): shutil.copytree(sourceLabels, targetRoot / "labels" / splitName, dirs_exist_ok=True)
    return targetRoot
