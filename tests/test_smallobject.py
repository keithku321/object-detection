# Tests for the small-object evaluation and the NWD distance (smallobject/evalCoco.py, smallobject/nwdLoss.py)
import pytest

numpy = pytest.importorskip("numpy")

from smallobject.evalCoco import averagePrecision, evaluateCoco, loadPredictions  # noqa: E402


# A ground-truth dict with one image that holds boxCount boxes of the given side length in a grid, plus matching perfect predictions
def makeGridScene(boxCount, side, imageId=1, firstAnnotationId=1):
    columns = 20
    boxes = [[(index % columns) * (side + 4.0), (index // columns) * (side + 4.0), float(side), float(side)] for index in range(boxCount)]
    annotations = [{"id": firstAnnotationId + index, "image_id": imageId, "category_id": 1, "bbox": box, "area": box[2] * box[3], "iscrowd": 0} for index, box in enumerate(boxes)]
    predictions = [{"image_id": imageId, "category_id": 1, "bbox": box, "score": 0.9 - index * 1e-4} for index, box in enumerate(boxes)]
    return {"id": imageId, "file_name": f"image{imageId}.jpg", "width": 2000, "height": 1500}, annotations, predictions


# Average precision skips the -1 entries that mean "no ground truth of this size" and returns 0 when nothing is left
def testAveragePrecisionSkipsMissing():
    precision = numpy.full((10, 101, 1, 4, 3), -1.0)
    precision[:, :, :, 0, -1] = 0.5
    precision[0, :, :, 1, -1] = 1.0
    assert averagePrecision(precision) == 0.5
    assert averagePrecision(precision, "small") == 1.0 and averagePrecision(precision, "small", iouIndex=0) == 1.0
    assert averagePrecision(precision, "large") == 0.0


# Perfect predictions score 1 for the sizes that exist; small means an area below 32 x 32 pixels in the original image
def testPerfectPredictionsBySize():
    pytest.importorskip("pycocotools")
    smallImage, smallAnnotations, smallPredictions = makeGridScene(10, side=10, imageId=1)
    largeImage, largeAnnotations, largePredictions = makeGridScene(5, side=120, imageId=2, firstAnnotationId=100)
    groundTruth = {"images": [smallImage, largeImage], "annotations": smallAnnotations + largeAnnotations, "categories": [{"id": 1, "name": "car"}]}
    metrics = evaluateCoco(groundTruth, smallPredictions + largePredictions)
    assert metrics["apSmall"] == pytest.approx(1.0) and metrics["apLarge"] == pytest.approx(1.0)
    assert metrics["apMedium"] == 0.0 and metrics["apCoco"] == pytest.approx(1.0)


# With 150 objects in one image, COCO's default limit of 100 detections caps the score; the raised limit does not
def testRaisedDetectionLimitMatters():
    pytest.importorskip("pycocotools")
    image, annotations, predictions = makeGridScene(150, side=10)
    groundTruth = {"images": [image], "annotations": annotations, "categories": [{"id": 1, "name": "pedestrian"}]}
    assert evaluateCoco(groundTruth, predictions, maximumDetections=500)["apSmall"] == pytest.approx(1.0)
    assert evaluateCoco(groundTruth, predictions, maximumDetections=100)["apSmall"] < 0.75


# Predictions are matched to ground-truth images by file name, and predictions for unknown files are dropped
def testPredictionsMatchedByFileName(tmp_path):
    import json
    groundTruth = {"images": [{"id": 7, "file_name": "a.jpg"}], "annotations": [], "categories": []}
    saved = [{"image_id": "a", "file_name": "a.jpg", "category_id": 1, "bbox": [0, 0, 5, 5], "score": 0.5}, {"image_id": "z", "file_name": "z.jpg", "category_id": 1, "bbox": [0, 0, 5, 5], "score": 0.5}]
    (tmp_path / "predictions.json").write_text(json.dumps(saved), encoding="utf-8")
    assert loadPredictions(tmp_path / "predictions.json", groundTruth) == [{"image_id": 7, "category_id": 1, "bbox": [0, 0, 5, 5], "score": 0.5}]


# NWD is 1 for identical boxes, falls smoothly with distance, and stays informative where IoU is already zero
def testNormalisedWassersteinDistance():
    torch = pytest.importorskip("torch")
    pytest.importorskip("ultralytics")
    from smallobject.nwdLoss import normalisedWassersteinDistance
    box = torch.tensor([[100.0, 100.0, 108.0, 108.0]])
    shifts = [0.0, 4.0, 8.0, 16.0]
    distances = [float(normalisedWassersteinDistance(box, box + shift, constant=12.8)) for shift in shifts]
    assert distances[0] == pytest.approx(1.0, abs=1e-3)
    assert distances == sorted(distances, reverse=True) and distances[2] > 0.3  # shift 8 has no overlap (IoU 0) but NWD is still clearly above 0
    assert float(normalisedWassersteinDistance(box, box + 4.0, constant=12.8)) == pytest.approx(float(normalisedWassersteinDistance(box + 4.0, box, constant=12.8)))


# Variant names are checked: unknown features, two attention blocks and "baseline plus something" are rejected
def testVariantNamesAreChecked():
    pytest.importorskip("ultralytics")
    from smallobject.train import parseVariant
    assert parseVariant("p2+eca+nwd") == {"p2", "eca", "nwd"}
    for badName in ("p3", "eca+cbam", "baseline+p2"):
        with pytest.raises(ValueError):
            parseVariant(badName)


# The command line computes the NWD constant from the training labels for a variant with nwd, unless --nwd-constant gives one
def testCommandLineComputesNwdConstant(monkeypatch):
    import sys
    from smallobject import train
    receivedConstants = []
    monkeypatch.setattr(train, "resolveSplitImages", lambda dataYaml, split: (["image.jpg"], ["car"]))
    monkeypatch.setattr(train, "computeNwdConstant", lambda imagePaths, imageSize: 9.5)
    monkeypatch.setattr(train, "runVariant", lambda *arguments, **keywords: receivedConstants.append(arguments[12]))
    for commandLine in (["--variant", "p2+nwd"], ["--variant", "p2+nwd", "--nwd-constant", "7"], ["--variant", "p2"]):
        monkeypatch.setattr(sys, "argv", ["train", "--project", "runs", *commandLine])
        train.main()
    assert receivedConstants == [9.5, 7.0, train.defaultNwdConstant]
