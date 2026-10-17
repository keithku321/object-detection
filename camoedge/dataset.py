# Data loading for CamoEdge: image and mask pairs, joint augmentation, and edge ground truth derived from each mask.
# Standard protocol (plans/04-camoedge.md): train on COD10K-Tr (3,040) + CAMO-Tr (1,000) = 4,040 images; test on COD10K-Te and CAMO-Te
# (NC4K optional). The folder layout below is the packaging most camouflage repositories share; change datasetLayout if yours differs.
# Check each dataset's README for its licence before storing copies anywhere public.
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import ColorJitter
from torchvision.transforms import functional as transformFunctions

from common.datasets import imageSuffixes

# Side length in pixels that images and masks are resized to for training and inference
defaultImageSize = 352
# ImageNet mean and standard deviation, matching the pretrained encoders
imageNetMean, imageNetStd = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
# Thickness in pixels of the edge ground truth; edges are rare pixels, so a slightly thick band is easier to learn
defaultEdgeWidth = 3
# Largest rotation in degrees used for augmentation
maximumRotation = 15
# Smallest fraction of the image area kept by the random crop
minimumCropScale = 0.75
# Folders below the dataset root: training (image folder, mask folder) pairs and test sets by name
datasetLayout = {
    "train": [("TrainDataset/Imgs", "TrainDataset/GT")],
    "test": {"COD10K": ("TestDataset/COD10K/Imgs", "TestDataset/COD10K/GT"), "CAMO": ("TestDataset/CAMO/Imgs", "TestDataset/CAMO/GT"), "NC4K": ("TestDataset/NC4K/Imgs", "TestDataset/NC4K/GT")},
}


# Edge ground truth of a binary mask: Canny on the mask, then dilation to the wanted thickness; returns a uint8 array of 0 and 255
def makeEdgeMap(maskArray, edgeWidth=defaultEdgeWidth):
    binaryMask = (np.asarray(maskArray) > 127).astype(np.uint8) * 255
    edges = cv2.Canny(binaryMask, 100, 200)
    if edgeWidth > 1: edges = cv2.dilate(edges, np.ones((edgeWidth, edgeWidth), np.uint8))
    return edges


# Super-class encoded in a COD10K file name such as "COD10K-CAM-1-Aquatic-1-BatFish-1.jpg" ("Aquatic"); "unknown" for other names
# Confirm this pattern against your download before trusting the per-category table
def superClassOf(fileName):
    parts = Path(fileName).stem.split("-")
    return parts[3] if len(parts) >= 6 and parts[0] == "COD10K" else "unknown"


# Pair every image with the mask that has the same file stem; images without a mask are skipped and counted
def pairImagesWithMasks(imageFolder, maskFolder):
    masksByStem = {path.stem: path for path in Path(maskFolder).iterdir() if path.suffix.lower() in imageSuffixes}
    imagePaths = sorted(path for path in Path(imageFolder).iterdir() if path.suffix.lower() in imageSuffixes)
    pairs = [(path, masksByStem[path.stem]) for path in imagePaths if path.stem in masksByStem]
    if len(pairs) < len(imagePaths): print(f"WARNING: {len(imagePaths) - len(pairs)} images in {imageFolder} have no mask and were skipped")
    return pairs


# Dataset of (image, mask, edge) samples; augment=True applies the same random geometry to image and mask, and colour jitter to the image
class CamouflageDataset(Dataset):
    # pairs is a list of (image path, mask path); maximumCount keeps only the first images, for quick local trials
    def __init__(self, pairs, imageSize=defaultImageSize, augment=False, edgeWidth=defaultEdgeWidth, maximumCount=None):
        self.pairs = list(pairs)[:maximumCount] if maximumCount else list(pairs)
        self.imageSize, self.augment, self.edgeWidth = imageSize, augment, edgeWidth
        self.colourJitter = ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02)

    # Number of samples
    def __len__(self):
        return len(self.pairs)

    # Random flip, rotation and crop applied identically to image and mask, then colour jitter on the image only
    def augmentPair(self, image, mask):
        if random.random() < 0.5: image, mask = transformFunctions.hflip(image), transformFunctions.hflip(mask)
        angle = random.uniform(-maximumRotation, maximumRotation)
        image = transformFunctions.rotate(image, angle, interpolation=transformFunctions.InterpolationMode.BILINEAR)
        mask = transformFunctions.rotate(mask, angle, interpolation=transformFunctions.InterpolationMode.NEAREST)
        width, height = image.size
        scale = random.uniform(minimumCropScale, 1.0) ** 0.5
        cropWidth, cropHeight = int(width * scale), int(height * scale)
        left, top = random.randint(0, width - cropWidth), random.randint(0, height - cropHeight)
        image, mask = transformFunctions.crop(image, top, left, cropHeight, cropWidth), transformFunctions.crop(mask, top, left, cropHeight, cropWidth)
        return self.colourJitter(image), mask

    # Load one sample: image tensor (3, S, S) normalised for the encoder, mask and edge tensors (1, S, S) with values 0 or 1
    def __getitem__(self, index):
        imagePath, maskPath = self.pairs[index]
        image, mask = Image.open(imagePath).convert("RGB"), Image.open(maskPath).convert("L")
        originalSize = (mask.height, mask.width)
        if self.augment: image, mask = self.augmentPair(image, mask)
        size = [self.imageSize, self.imageSize]
        image = transformFunctions.resize(image, size, interpolation=transformFunctions.InterpolationMode.BILINEAR)
        mask = transformFunctions.resize(mask, size, interpolation=transformFunctions.InterpolationMode.NEAREST)
        maskArray = np.asarray(mask)
        imageTensor = transformFunctions.normalize(transformFunctions.to_tensor(image), imageNetMean, imageNetStd)
        maskTensor = torch.from_numpy((maskArray > 127).astype(np.float32)).unsqueeze(0)
        edgeTensor = torch.from_numpy((makeEdgeMap(maskArray, self.edgeWidth) > 0).astype(np.float32)).unsqueeze(0)
        return {"image": imageTensor, "mask": maskTensor, "edge": edgeTensor, "name": imagePath.name, "maskPath": str(maskPath), "originalSize": originalSize}


# Turn an RGB image array into the normalised input tensor (1, 3, S, S) that the model expects (used by the demo and by inference)
def preprocessImage(imageArray, imageSize=defaultImageSize):
    image = transformFunctions.resize(Image.fromarray(imageArray).convert("RGB"), [imageSize, imageSize], interpolation=transformFunctions.InterpolationMode.BILINEAR)
    return transformFunctions.normalize(transformFunctions.to_tensor(image), imageNetMean, imageNetStd).unsqueeze(0)


# All training pairs below a dataset root, following datasetLayout
def collectTrainingPairs(datasetRoot, layout=datasetLayout):
    return [pair for imageFolder, maskFolder in layout["train"] for pair in pairImagesWithMasks(Path(datasetRoot) / imageFolder, Path(datasetRoot) / maskFolder)]


# Test pairs by test-set name; test sets whose folders are missing are left out
def collectTestPairs(datasetRoot, layout=datasetLayout):
    pairsByName = {}
    for testName, (imageFolder, maskFolder) in layout["test"].items():
        if (Path(datasetRoot) / imageFolder).exists(): pairsByName[testName] = pairImagesWithMasks(Path(datasetRoot) / imageFolder, Path(datasetRoot) / maskFolder)
    return pairsByName


# Training data loader with augmentation and shuffling; drops the last incomplete batch so batch normalisation never sees one image
def buildTrainingLoader(datasetRoot, imageSize=defaultImageSize, batchSize=16, workers=2, maximumCount=None, layout=datasetLayout):
    dataset = CamouflageDataset(collectTrainingPairs(datasetRoot, layout), imageSize, augment=True, maximumCount=maximumCount)
    return DataLoader(dataset, batch_size=batchSize, shuffle=True, num_workers=workers, drop_last=len(dataset) > batchSize, pin_memory=torch.cuda.is_available())


# Test data loader for one list of pairs, without augmentation; batch size 1 because results are scored at each image's original size
def buildTestLoader(pairs, imageSize=defaultImageSize, workers=2, maximumCount=None):
    return DataLoader(CamouflageDataset(pairs, imageSize, augment=False, maximumCount=maximumCount), batch_size=1, shuffle=False, num_workers=workers)
