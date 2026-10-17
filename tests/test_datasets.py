# Tests for the dataset helpers (common/datasets.py): manifest, checksums, label checks, VOC conversion, folds and split lists
import hashlib
import zipfile

import pytest

from common import datasets


# Every manifest entry has the fields the download code relies on, and no entry ships a made-up checksum format
def testManifestEntriesAreComplete():
    manifest = datasets.loadManifest()
    assert {"brackish", "dawn", "visdrone", "cod10k", "camo"} <= set(manifest)
    for name, entry in manifest.items():
        assert entry["access"] in {"ultralytics", "huggingface", "url", "manual"}, name
        assert {"plan", "task", "license", "redistribution", "instructions"} <= set(entry), name
        assert entry["sha256"] is None or len(entry["sha256"]) == 64, name


# The checksum matches hashlib, a wrong checksum is reported, and a missing checksum is neither pass nor fail
def testChecksumVerification(tmp_path):
    filePath = tmp_path / "data.bin"
    filePath.write_bytes(b"robust detection")
    expected = hashlib.sha256(b"robust detection").hexdigest()
    assert datasets.computeSha256(filePath) == expected
    assert datasets.verifySha256(filePath, expected) is True
    assert datasets.verifySha256(filePath, "0" * 64) is False
    assert datasets.verifySha256(filePath, None) is None


# A zip member that points outside the target folder is refused
def testUnsafeZipIsRefused(tmp_path):
    zipPath = tmp_path / "bad.zip"
    with zipfile.ZipFile(zipPath, "w") as archive: archive.writestr("../escape.txt", "x")
    with pytest.raises(ValueError):
        datasets.extractZip(zipPath, tmp_path / "out")


# The dataset YAML lists the splits and the classes in order
def testWriteDatasetYaml(tmp_path):
    yamlText = datasets.writeDatasetYaml(tmp_path, ["crab", "fish"], testSplit="images/test").read_text(encoding="utf-8")
    assert "train: images/train" in yamlText and "test: images/test" in yamlText
    assert "  0: crab" in yamlText and "  1: fish" in yamlText


# Label problems are found: a class number out of range, a coordinate outside 0..1 and a line with the wrong number of columns
def testLabelProblemsAreFound(tmp_path):
    labelFolder = tmp_path / "labels" / "train"
    labelFolder.mkdir(parents=True)
    (labelFolder / "good.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    (labelFolder / "bad.txt").write_text("7 0.5 0.5 0.2 0.2\n0 1.5 0.5 0.2 0.2\n0 0.5 0.5\n", encoding="utf-8")
    problems = datasets.findLabelProblems(tmp_path, "train", classCount=2)
    assert len(problems) == 3 and all("bad.txt" in problem for problem in problems)


# A Pascal VOC box becomes the right normalised YOLO line, and an unknown class is counted instead of written
def testVocConversion(tmp_path):
    xmlText = ("<annotation><size><width>200</width><height>100</height></size>"
               "<object><name>scallop</name><bndbox><xmin>50</xmin><ymin>20</ymin><xmax>150</xmax><ymax>60</ymax></bndbox></object>"
               "<object><name>waterweeds</name><bndbox><xmin>0</xmin><ymin>0</ymin><xmax>10</xmax><ymax>10</ymax></bndbox></object></annotation>")
    (tmp_path / "frame.xml").write_text(xmlText, encoding="utf-8")
    unknownClasses = datasets.convertVocXmlToYolo(tmp_path, tmp_path / "labels", ["holothurian", "echinus", "scallop", "starfish"])
    assert unknownClasses == {"waterweeds": 1}
    assert (tmp_path / "labels" / "frame.txt").read_text(encoding="utf-8").strip() == "2 0.500000 0.400000 0.500000 0.400000"


# Folds are disjoint, cover every item, keep each stratum balanced and are the same for the same seed
def testStratifiedFolds():
    items = [f"image{index}" for index in range(100)]
    strata = ["fog"] * 40 + ["rain"] * 30 + ["snow"] * 20 + ["sand"] * 10
    folds = datasets.makeStratifiedFolds(items, strata, foldCount=5, seed=0)
    assert sorted(item for fold in folds for item in fold) == sorted(items)
    stratumOf = dict(zip(items, strata))
    for fold in folds:
        assert [sum(stratumOf[item] == stratum for item in fold) for stratum in ("fog", "rain", "snow", "sand")] == [8, 6, 4, 2]
    assert folds == datasets.makeStratifiedFolds(items, strata, foldCount=5, seed=0)
    assert folds != datasets.makeStratifiedFolds(items, strata, foldCount=5, seed=1)


# Split lists are written relative to the dataset root with forward slashes and read back as paths below a root
def testSplitFileRoundTrip(tmp_path):
    imagePaths = [tmp_path / "images" / "fog" / "b.jpg", tmp_path / "images" / "fog" / "a.jpg"]
    listPath = datasets.writeSplitFile(tmp_path / "splits" / "fold0.txt", imagePaths, tmp_path)
    assert listPath.read_text(encoding="utf-8").splitlines() == ["images/fog/a.jpg", "images/fog/b.jpg"]
    assert datasets.readSplitFile(listPath, tmp_path) == sorted(imagePaths)


# Hamming distances count differing bits
def testHammingDistances():
    numpy = pytest.importorskip("numpy")
    distances = datasets.hammingDistances(0b1111, numpy.array([0b1111, 0b0000, 0b1010], dtype=numpy.uint64))
    assert distances.tolist() == [0, 4, 2]


# A copy of an image with slightly different brightness is grouped with the original; an unrelated image is not
def testNearDuplicateGroups(tmp_path):
    numpy = pytest.importorskip("numpy")
    pytest.importorskip("imagehash")
    image = pytest.importorskip("PIL.Image")
    randomGenerator = numpy.random.default_rng(0)
    first = numpy.kron(randomGenerator.integers(0, 256, (8, 8), dtype=numpy.uint8), numpy.ones((16, 16), dtype=numpy.uint8))
    other = numpy.kron(randomGenerator.integers(0, 256, (8, 8), dtype=numpy.uint8), numpy.ones((16, 16), dtype=numpy.uint8))
    paths = {"first": tmp_path / "first.png", "copy": tmp_path / "copy.png", "other": tmp_path / "other.png"}
    image.fromarray(first).save(paths["first"])
    image.fromarray(numpy.clip(first.astype(int) + 3, 0, 255).astype(numpy.uint8)).save(paths["copy"])
    image.fromarray(other).save(paths["other"])
    groups = datasets.findNearDuplicateGroups(list(paths.values()))
    assert len(groups) == 1 and sorted(groups[0]) == sorted([paths["first"], paths["copy"]])
    crossSplit = datasets.findCrossSplitDuplicates({"train": [paths["first"], paths["other"]], "val": [paths["copy"]]})
    assert len(crossSplit) == 1


# A dataset delivered as several zips is downloaded, verified against each checksum and unzipped into the entry's extract folder
def testFetchDatasetWithSeveralZips(tmp_path):
    import json
    import zipfile
    fileEntries = []
    for weatherName in ("Fog", "Rain"):
        zipPath = tmp_path / "source" / f"{weatherName}.zip"
        zipPath.parent.mkdir(exist_ok=True)
        with zipfile.ZipFile(zipPath, "w") as archive: archive.writestr(f"{weatherName}/{weatherName}-1.jpg", b"image")
        fileEntries.append({"name": zipPath.name, "url": zipPath.as_uri(), "sha256": datasets.computeSha256(zipPath)})
    manifestFile = tmp_path / "manifest.json"
    manifestFile.write_text(json.dumps({"datasets": [{"name": "demo", "access": "url", "downloadFiles": fileEntries, "extractFolder": "demo_raw"}]}), encoding="utf-8")
    extractedFolder = datasets.fetchDataset("demo", tmp_path / "target", path=manifestFile)
    assert extractedFolder == tmp_path / "target" / "demo_raw"
    assert (extractedFolder / "Fog" / "Fog-1.jpg").read_bytes() == b"image" and (extractedFolder / "Rain" / "Rain-1.jpg").exists()
    fileEntries[0]["sha256"] = "0" * 64
    manifestFile.write_text(json.dumps({"datasets": [{"name": "demo", "access": "url", "downloadFiles": fileEntries, "extractFolder": "demo_raw"}]}), encoding="utf-8")
    (tmp_path / "target" / "downloads" / "Fog.zip").unlink()
    with pytest.raises(ValueError): datasets.fetchDataset("demo", tmp_path / "target", path=manifestFile)

