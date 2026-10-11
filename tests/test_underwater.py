# Tests for the underwater image processing (underwater/enhance.py, underwater/degrade.py); they need OpenCV and NumPy only
import pytest

numpy = pytest.importorskip("numpy")
pytest.importorskip("cv2")

from underwater.degrade import buildDegradedTestSets, degradeImage, severityLevels  # noqa: E402
from underwater.enhance import enhanceDataset, enhanceImage, greyWorldWhiteBalance  # noqa: E402


# A blue-green image with a weak red channel, like an underwater photograph
def makeUnderwaterImage():
    randomGenerator = numpy.random.default_rng(0)
    image = randomGenerator.integers(0, 256, (64, 64, 3)).astype(numpy.float32)
    image[:, :, 2] *= 0.3
    return image.astype(numpy.uint8)


# Grey-world white balance makes the three channel means almost equal
def testGreyWorldEqualisesChannels():
    balanced = greyWorldWhiteBalance(makeUnderwaterImage())
    channelMeans = balanced.reshape(-1, 3).mean(axis=0)
    assert channelMeans.max() - channelMeans.min() < 12


# Enhancement keeps the shape and type of the image
def testEnhanceKeepsShape():
    image = makeUnderwaterImage()
    enhanced = enhanceImage(image)
    assert enhanced.shape == image.shape and enhanced.dtype == numpy.uint8


# Severity 0 leaves the image untouched, and each higher severity moves it further from the original
def testDegradationGrowsWithSeverity():
    image = makeUnderwaterImage()
    assert numpy.array_equal(degradeImage(image, 0), image)
    differences = [numpy.abs(degradeImage(image, severity).astype(int) - image.astype(int)).mean() for severity in severityLevels]
    assert differences == sorted(differences) and differences[-1] > differences[1] > 0
    with pytest.raises(ValueError):
        degradeImage(image, 9)


# The degraded test sets keep file names and labels, and each severity gets its own dataset YAML
def testDegradedSetsKeepLabels(tmp_path):
    from common.datasets import writeImageBgr
    writeImageBgr(tmp_path / "data" / "images" / "test" / "frame.png", makeUnderwaterImage())
    (tmp_path / "data" / "labels" / "test").mkdir(parents=True)
    (tmp_path / "data" / "labels" / "test" / "frame.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    yamlBySeverity = buildDegradedTestSets(tmp_path / "data", tmp_path / "degraded", ["fish"], severities=(0, 4))
    assert sorted(yamlBySeverity) == [0, 4]
    for severity in (0, 4):
        assert (tmp_path / "degraded" / f"severity{severity}" / "images" / "test" / "frame.png").exists()
        assert (tmp_path / "degraded" / f"severity{severity}" / "labels" / "test" / "frame.txt").read_text(encoding="utf-8").startswith("0 0.5")
    enhancedRoot = enhanceDataset(tmp_path / "data", tmp_path / "enhanced")
    assert (enhancedRoot / "images" / "test" / "frame.png").exists() and (enhancedRoot / "labels" / "test" / "frame.txt").exists()
