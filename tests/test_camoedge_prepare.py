# Tests for the COD10K and CAMO preparation (camoedge/prepareCamo.py) on a tiny made-up layout; nothing is downloaded
import pytest

image = pytest.importorskip("PIL.Image")

from camoedge import prepareCamo  # noqa: E402


# Write image and mask pairs with the given names into an Imgs and a GT folder
def makePairs(imageFolder, maskFolder, names):
    imageFolder.mkdir(parents=True, exist_ok=True)
    maskFolder.mkdir(parents=True, exist_ok=True)
    for name in names:
        image.new("RGB", (8, 8)).save(imageFolder / f"{name}.jpg")
        image.new("L", (8, 8)).save(maskFolder / f"{name}.png")


# The counts of the protocol pass, and a missing pair is reported with the found and expected numbers
def testCountsAreChecked(tmp_path, monkeypatch):
    monkeypatch.setattr(prepareCamo, "expectedCounts", {"train": 3, "COD10K": 2, "CAMO": 1})
    makePairs(tmp_path / "TrainDataset" / "Imgs", tmp_path / "TrainDataset" / "GT", ["a", "b", "c"])
    makePairs(tmp_path / "TestDataset" / "COD10K" / "Imgs", tmp_path / "TestDataset" / "COD10K" / "GT", ["d", "e"])
    makePairs(tmp_path / "TestDataset" / "CAMO" / "Imgs", tmp_path / "TestDataset" / "CAMO" / "GT", ["f"])
    assert prepareCamo.checkCounts(tmp_path) == {"train": 3, "COD10K": 2, "CAMO": 1}
    (tmp_path / "TrainDataset" / "GT" / "c.png").unlink()
    with pytest.raises(ValueError, match="train"):
        prepareCamo.checkCounts(tmp_path)


# The manifest lists both zips of the release with a Drive id and a 64-character checksum
def testManifestListsBothZips():
    from common.datasets import getManifestEntry
    driveFiles = getManifestEntry("camoedge")["googleDriveFiles"]
    assert [fileEntry["name"] for fileEntry in driveFiles] == ["TrainDataset.zip", "TestDataset.zip"]
    assert all(len(fileEntry["sha256"]) == 64 and fileEntry["googleDriveId"] for fileEntry in driveFiles)
