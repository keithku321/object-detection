# Tests for the underwater image processing (underwater/enhance.py, underwater/degrade.py); they need OpenCV and NumPy only
import pytest

numpy = pytest.importorskip("numpy")
cv2 = pytest.importorskip("cv2")

from underwater.degrade import buildDegradedTestSets, degradeImage, severityLevels  # noqa: E402
from underwater.enhance import enhanceDataset, enhanceImage, greyWorldWhiteBalance  # noqa: E402
from underwater.prepareBrackish import arrangeDataset  # noqa: E402


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


# Build a tiny Roboflow export (train/, valid/ and test/ with images/ and labels/, plus data.yaml) with the given class names
def makeRoboflowExport(exportFolder, classNames):
    for roboflowName, shardNames in (("train", ("shard_000", "shard_001")), ("valid", ("",)), ("test", ("",))):
        for shardName in shardNames:
            shardFolder = exportFolder / roboflowName / shardName
            (shardFolder / "images").mkdir(parents=True)
            (shardFolder / "labels").mkdir(parents=True)
            frameName = f"frame{shardName}"
            cv2.imwrite(str(shardFolder / "images" / f"{frameName}.jpg"), numpy.zeros((16, 16, 3), dtype=numpy.uint8))
            (shardFolder / "labels" / f"{frameName}.txt").write_text("1 0.5 0.5 0.2 0.2\n", encoding="utf-8")
            if shardName: (shardFolder / "images" / "metadata.jsonl").write_text("{}\n", encoding="utf-8")
    (exportFolder / "data.yaml").write_text(f"nc: {len(classNames)}\nnames: {list(classNames)}\n", encoding="utf-8")


# The Roboflow folders are renamed to images/<split> and labels/<split> and dataset.yaml lists the test split
def testBrackishExportIsArranged(tmp_path):
    classNames = ["crab", "fish", "jellyfish", "shrimp", "small_fish", "starfish"]
    makeRoboflowExport(tmp_path / "export", classNames)
    yamlPath = arrangeDataset(tmp_path / "export", tmp_path / "out", classNames)
    datasetRoot = tmp_path / "out" / "brackish"
    assert yamlPath == datasetRoot / "dataset.yaml" and "images/test" in yamlPath.read_text(encoding="utf-8")
    assert sorted(file.name for file in (datasetRoot / "images" / "train").glob("*.jpg")) == ["frameshard_000.jpg", "frameshard_001.jpg"]
    for splitName in ("val", "test"):
        assert (datasetRoot / "images" / splitName / "frame.jpg").exists() and (datasetRoot / "labels" / splitName / "frame.txt").exists()
    assert not list((datasetRoot / "train").glob("shard_*"))


# A different class order in the download is refused, because it would mislabel every box
def testBrackishClassOrderMismatchIsRefused(tmp_path):
    makeRoboflowExport(tmp_path / "export", ["fish", "crab", "jellyfish", "shrimp", "small_fish", "starfish"])
    with pytest.raises(ValueError):
        arrangeDataset(tmp_path / "export", tmp_path / "out", ["crab", "fish", "jellyfish", "shrimp", "small_fish", "starfish"])

