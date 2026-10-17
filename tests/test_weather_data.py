# Tests for the DAWN preparation (weather/prepareDawn.py) on a small made-up download; no real DAWN images are needed
import pytest

numpy = pytest.importorskip("numpy")
pytest.importorskip("imagehash")
image = pytest.importorskip("PIL.Image")

from weather import prepareDawn  # noqa: E402

# Weather types and class names used by the made-up download
weatherTypes = ["fog", "rain", "snow", "sand"]
classNames = ["car", "bus", "truck", "person", "motorcycle", "bicycle"]


# Save a random blocky image, different for every seed, so the perceptual hash does not see duplicates
def saveRandomImage(path, seed):
    blocks = numpy.random.default_rng(seed).integers(0, 256, (8, 8), dtype=numpy.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.fromarray(numpy.kron(blocks, numpy.ones((16, 16), dtype=numpy.uint8))).save(path)


# Build a raw download: every weather folder has images 0.jpg to 4.jpg (the same names in every folder) with YOLO labels whose
# class number is the index of the weather type, plus one XML-only image and one image without any annotation in the fog folder
def makeRawDownload(rawRoot):
    for weatherIndex, weatherType in enumerate(weatherTypes):
        folder = rawRoot / weatherType.capitalize()
        for imageIndex in range(5):
            saveRandomImage(folder / "images" / f"{imageIndex}.jpg", seed=weatherIndex * 100 + imageIndex)
            (folder / "labels").mkdir(parents=True, exist_ok=True)
            (folder / "labels" / f"{imageIndex}.txt").write_text(f"{weatherIndex} 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    saveRandomImage(rawRoot / "Fog" / "images" / "xmlOnly.jpg", seed=900)
    xmlText = "<annotation><size><width>128</width><height>128</height></size><object><name>bus</name><bndbox><xmin>32</xmin><ymin>32</ymin><xmax>96</xmax><ymax>96</ymax></bndbox></object></annotation>"
    (rawRoot / "Fog" / "labels" / "xmlOnly.xml").write_text(xmlText, encoding="utf-8")
    saveRandomImage(rawRoot / "Fog" / "images" / "unlabelled.jpg", seed=901)


# Images with the same file name in different weather folders keep their own labels; XML is converted; a missing label becomes empty
# The made-up labels already follow the project class order, so their numbers are copied unchanged (sourceClassIds=None)
def testConversionKeepsLabelsPerWeather(tmp_path):
    makeRawDownload(tmp_path / "raw")
    imagesByWeather = prepareDawn.convertDawn(tmp_path / "raw", tmp_path / "dawn", weatherTypes, classNames, sourceClassIds=None)
    assert [len(imagesByWeather[weatherType]) for weatherType in weatherTypes] == [7, 5, 5, 5]
    for weatherIndex, weatherType in enumerate(weatherTypes):
        assert (tmp_path / "dawn" / "labels" / weatherType / "0.txt").read_text(encoding="utf-8").startswith(f"{weatherIndex} ")
    assert (tmp_path / "dawn" / "labels" / "fog" / "xmlOnly.txt").read_text(encoding="utf-8").strip() == "1 0.500000 0.500000 0.500000 0.500000"
    assert (tmp_path / "dawn" / "labels" / "fog" / "unlabelled.txt").read_text(encoding="utf-8") == ""
    assert prepareDawn.countClassIds(tmp_path / "dawn") == {0: 5, 1: 6, 2: 5, 3: 5}


# Folds are stratified by weather, written as relative lists, and rebuilt into per-fold YAML files with disjoint train and val
def testFoldsAndYamls(tmp_path):
    makeRawDownload(tmp_path / "raw")
    imagesByWeather = prepareDawn.convertDawn(tmp_path / "raw", tmp_path / "dawn", weatherTypes, classNames, sourceClassIds=None)
    folds = prepareDawn.assignFolds(imagesByWeather, count=5, seed=0)
    assert sorted(len(fold) for fold in folds) == [4, 4, 4, 5, 5] and sum(len(fold) for fold in folds) == 22
    prepareDawn.writeFoldLists(folds, tmp_path / "dawn", tmp_path / "splits")
    assert (tmp_path / "splits" / "fold0.txt").read_text(encoding="utf-8").splitlines()[0].startswith("images/")
    yamlByFold = prepareDawn.buildFoldYamls(tmp_path / "dawn", classNames, tmp_path / "splits", count=5)
    assert sorted(yamlByFold) == [0, 1, 2, 3, 4] and "val: folds/fold2_val.txt" in yamlByFold[2].read_text(encoding="utf-8")
    trainLines = set((tmp_path / "dawn" / "folds" / "fold2_train.txt").read_text(encoding="utf-8").splitlines())
    valLines = set((tmp_path / "dawn" / "folds" / "fold2_val.txt").read_text(encoding="utf-8").splitlines())
    assert not trainLines & valLines and len(trainLines) + len(valLines) == 22
    heldOut = prepareDawn.heldOutImagesByWeather(tmp_path / "dawn", 2, weatherTypes, tmp_path / "splits")
    assert sum(len(paths) for paths in heldOut.values()) == len(valLines) and all(len(paths) >= 1 for paths in heldOut.values())


# The settings come from the dataset manifest, so the class order is defined in one place
def testSettingsComeFromManifest():
    manifestWeatherTypes, manifestClassNames = prepareDawn.getDawnSettings()
    assert manifestWeatherTypes == weatherTypes and manifestClassNames == classNames


# The real download numbers its classes like COCO counting from 1; the numbers are renumbered by class name and the unused class is dropped
def testRealDownloadClassNumbersAreRemapped(tmp_path):
    labelFolder = tmp_path / "raw" / "Fog" / "Fog_YOLO_darknet"
    labelFolder.mkdir(parents=True)
    saveRandomImage(tmp_path / "raw" / "Fog" / "fog-1.jpg", seed=1)
    (labelFolder / "fog-1.txt").write_text("3 0.5 0.5 0.2 0.2\n1 0.1 0.1 0.1 0.1\n7 0.3 0.3 0.1 0.1\n8 0.7 0.7 0.2 0.2\n6 0.4 0.4 0.1 0.1\n", encoding="utf-8")
    prepareDawn.convertDawn(tmp_path / "raw", tmp_path / "dawn", ["fog"], classNames)
    convertedLines = (tmp_path / "dawn" / "labels" / "fog" / "fog-1.txt").read_text(encoding="utf-8").splitlines()
    assert [line.split()[0] for line in convertedLines] == ["0", "3", "2", "1"]
    assert convertedLines[0] == "0 0.5 0.5 0.2 0.2"


# The source class table in the manifest names only classes that the project uses, plus the one dropped class
def testSourceClassTableMatchesProjectClasses():
    sourceClassIds = prepareDawn.getDawnSourceClassIds()
    assert set(sourceClassIds.values()) - set(classNames) == {"train"}
    assert set(classNames) <= set(sourceClassIds.values())

