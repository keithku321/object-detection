# Synthetic turbidity for the robustness test (experiment U5): degraded copies of the TEST set at severity 0 to 4.
# Each severity combines red-channel attenuation (colour cast), haze I = J*t + A*(1 - t), and Gaussian blur. The recipe is
# fully deterministic (no random numbers), so every model sees exactly the same degraded images.
import shutil
from pathlib import Path

import cv2
import numpy as np

from common.datasets import listImages, readImageBgr, writeDatasetYaml, writeImageBgr

# Severity levels; 0 is the untouched image
severityLevels = (0, 1, 2, 3, 4)
# Haze transmission t per severity; lower means thicker haze
hazeTransmission = (1.0, 0.85, 0.70, 0.55, 0.40)
# Gain applied to the red channel per severity; lower means a stronger blue-green cast
redChannelGain = (1.0, 0.85, 0.70, 0.55, 0.40)
# Gaussian blur sigma in pixels per severity
blurSigma = (0.0, 0.8, 1.6, 2.4, 3.2)
# Veiling light A of the haze model as BGR fractions of 255; blue-green like water
ambientLightBgr = (0.60, 0.55, 0.20)


# Scale the red channel of a BGR image
def attenuateRed(imageBgr, gain):
    image = imageBgr.astype(np.float32)
    image[:, :, 2] *= gain
    return np.clip(image, 0, 255).astype(np.uint8)


# Haze model I = J * t + A * (1 - t) with one transmission value for the whole image
def addHaze(imageBgr, transmission, ambientLight=ambientLightBgr):
    veil = np.array(ambientLight, dtype=np.float32) * 255.0
    hazy = imageBgr.astype(np.float32) * transmission + veil * (1.0 - transmission)
    return np.clip(hazy, 0, 255).astype(np.uint8)


# Gaussian blur with the given sigma; sigma 0 returns the image unchanged
def blurImage(imageBgr, sigma):
    return imageBgr if sigma <= 0 else cv2.GaussianBlur(imageBgr, (0, 0), sigma)


# Apply all three degradations for one severity level (0 to 4)
def degradeImage(imageBgr, severity):
    if severity not in severityLevels: raise ValueError(f"severity must be one of {severityLevels}")
    degraded = attenuateRed(imageBgr, redChannelGain[severity])
    degraded = addHaze(degraded, hazeTransmission[severity])
    return blurImage(degraded, blurSigma[severity])


# Write one degraded copy of a split per severity level and a dataset YAML for each, so model.val can score every level
# afterDegrade is an optional function applied after the degradation (for example enhanceImage, to test the enhancement pipeline)
# Returns a dict from severity to YAML path
def buildDegradedTestSets(rootPath, outputRoot, classNames, splitName="test", severities=severityLevels, afterDegrade=None):
    rootPath, outputRoot = Path(rootPath), Path(outputRoot)
    yamlBySeverity = {}
    for severity in severities:
        severityRoot = outputRoot / f"severity{severity}"
        for imagePath in listImages(rootPath / "images" / splitName):
            degraded = degradeImage(readImageBgr(imagePath), severity)
            writeImageBgr(severityRoot / "images" / splitName / imagePath.name, afterDegrade(degraded) if afterDegrade else degraded)
        shutil.copytree(rootPath / "labels" / splitName, severityRoot / "labels" / splitName, dirs_exist_ok=True)
        yamlBySeverity[severity] = writeDatasetYaml(severityRoot, classNames, trainSplit=f"images/{splitName}", valSplit=f"images/{splitName}")
    return yamlBySeverity


# Score a model on every degraded set and return {severity: mAP@0.5}
def evaluateTurbidityRobustness(weightsPath, yamlBySeverity, imageSize=640, device=None):
    from common.training import evaluateDetector
    return {severity: float(evaluateDetector(weightsPath, yamlPath, "val", imageSize, device=device).box.map50) for severity, yamlPath in sorted(yamlBySeverity.items())}


# Plot mAP@0.5 against severity for several models and save the figure; curvesByModel maps a label to {severity: mAP}
def plotRobustnessCurves(curvesByModel, outputPath):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(figsize=(6, 4))
    for label, curve in curvesByModel.items():
        axes.plot(list(curve.keys()), list(curve.values()), marker="o", label=label)
    axes.set_xlabel("Turbidity severity")
    axes.set_ylabel("mAP@0.5")
    axes.set_xticks(list(severityLevels))
    axes.legend()
    axes.grid(alpha=0.3)
    figure.tight_layout()
    figure.savefig(outputPath, dpi=150)
    plt.close(figure)
    return Path(outputPath)
