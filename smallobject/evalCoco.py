# COCO-style AP by object size (AP_small, AP_medium, AP_large) for Plan C, with the detection limit raised above COCO's default of 100.
#   python -m smallobject.evalCoco --weights best.pt --data VisDrone.yaml --imgsz 640
# Steps (plans/03-small-object.md, "Evaluation: AP_s"):
#   1. Ultralytics validates with a very low confidence threshold and saves predictions.json (boxes in original image pixels).
#   2. The YOLO labels are converted to a COCO ground-truth dict, with box areas in ORIGINAL image pixels.
#   3. pycocotools COCOeval runs with a raised maxDets; AP is read from its precision array for small (< 32^2 px), medium and large.
#   4. The overall AP is compared with Ultralytics' own mAP@0.5:0.95; a large gap means the conversion is wrong.
# Note: Ultralytics 8.4.175 also prints its own mAP_small when it validates with save_json=True (it runs faster-coco-eval, which it
# installs on first use), but with COCO's default limit of 100 detections per image, which is too low for dense VisDrone scenes.
# Use the numbers of this file in the report and treat the printed ones as a rough cross-check.
import argparse
import json
import statistics
import tempfile
from pathlib import Path

from common.datasets import imageSuffixes
from common.training import evaluateDetector, protocolImageSize

# Detections per image that COCOeval may use; COCO's default of 100 is too low for images with several hundred objects
cocoMaximumDetections = 500
# Confidence threshold for saving detections; very low so the precision-recall curve is complete
lowConfidence = 0.001
# Largest accepted gap between our COCO AP and Ultralytics' mAP@0.5:0.95 (they interpolate the curve slightly differently)
agreementTolerance = 0.02
# Index of each object-size range in COCOeval's precision array
areaIndices = {"all": 0, "small": 1, "medium": 2, "large": 3}
# The NMS-free head returns at most this many detections per image unless the limit is raised
endToEndDetectionCap = 300


# Image paths and class names of one split of a dataset YAML; Ultralytics resolves (and if needed downloads) the dataset
def resolveSplitImages(dataYaml, split="val"):
    from ultralytics.data.utils import check_det_dataset
    data = check_det_dataset(str(dataYaml))
    entries = data[split] if isinstance(data[split], (list, tuple)) else [data[split]]
    imagePaths = []
    for entry in entries:
        entryPath = Path(entry)
        if entryPath.is_dir(): imagePaths.extend(path for path in entryPath.rglob("*") if path.suffix.lower() in imageSuffixes)
        else: imagePaths.extend(Path(line) for line in entryPath.read_text(encoding="utf-8").splitlines() if line.strip())
    return sorted(imagePaths), [data["names"][index] for index in sorted(data["names"])]


# Read one YOLO label file as a list of (classNumber, xCentre, yCentre, width, height) with normalised coordinates
def readYoloLabels(labelPath):
    labelPath = Path(labelPath)
    if not labelPath.exists(): return []
    rows = [line.split() for line in labelPath.read_text(encoding="utf-8").splitlines() if line.strip()]
    return [(int(row[0]), *map(float, row[1:5])) for row in rows]


# Convert YOLO labels to a COCO ground-truth dict; box coordinates and areas are in original image pixels
# Image ids count from 1 in the order of imagePaths, and category ids are class number + 1 (the convention Ultralytics uses)
def buildCocoGroundTruth(imagePaths, classNames):
    from PIL import Image
    from ultralytics.data.utils import exif_size, img2label_paths
    labelPaths = img2label_paths([str(path) for path in imagePaths])
    images, annotations = [], []
    for imageId, (imagePath, labelPath) in enumerate(zip(imagePaths, labelPaths), start=1):
        with Image.open(imagePath) as image:
            width, height = exif_size(image)
        images.append({"id": imageId, "file_name": Path(imagePath).name, "width": width, "height": height})
        for classNumber, xCentre, yCentre, boxWidth, boxHeight in readYoloLabels(labelPath):
            pixelWidth, pixelHeight = boxWidth * width, boxHeight * height
            annotations.append({
                "id": len(annotations) + 1, "image_id": imageId, "category_id": classNumber + 1,
                "bbox": [xCentre * width - pixelWidth / 2, yCentre * height - pixelHeight / 2, pixelWidth, pixelHeight],
                "area": pixelWidth * pixelHeight, "iscrowd": 0,
            })
    categories = [{"id": number + 1, "name": name} for number, name in enumerate(classNames)]
    return {"images": images, "annotations": annotations, "categories": categories}


# Load the predictions that Ultralytics saved and give each one the image id used in our ground truth (matched by file name)
def loadPredictions(predictionsPath, groundTruth):
    imageIdByName = {image["file_name"]: image["id"] for image in groundTruth["images"]}
    if len(imageIdByName) != len(groundTruth["images"]): raise ValueError("Two images share a file name; predictions cannot be matched by name")
    predictions = json.loads(Path(predictionsPath).read_text(encoding="utf-8"))
    return [{"image_id": imageIdByName[item["file_name"]], "category_id": item["category_id"], "bbox": item["bbox"], "score": item["score"]}
            for item in predictions if item["file_name"] in imageIdByName]


# Average precision from COCOeval's precision array [iou, recall, class, area, maxDets]; -1 entries mean "no ground truth" and are skipped
# iouIndex None averages over IoU 0.50:0.95; 0 is IoU 0.50 and 5 is IoU 0.75. Returns 0.0 when nothing can be scored.
def averagePrecision(precision, areaName="all", iouIndex=None):
    values = precision[:, :, :, areaIndices[areaName], -1] if iouIndex is None else precision[iouIndex, :, :, areaIndices[areaName], -1]
    valid = values[values > -1]
    return float(valid.mean()) if valid.size else 0.0


# Run COCOeval on a ground-truth dict and a prediction list; returns AP overall, at IoU 0.5 and 0.75, and by object size
def evaluateCoco(groundTruth, predictions, maximumDetections=cocoMaximumDetections):
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    emptyResult = {"apCoco": 0.0, "ap50Coco": 0.0, "ap75Coco": 0.0, "apSmall": 0.0, "apMedium": 0.0, "apLarge": 0.0}
    if not predictions or not groundTruth["annotations"]: return emptyResult
    cocoTruth = COCO()
    cocoTruth.dataset = groundTruth
    cocoTruth.createIndex()
    evaluator = COCOeval(cocoTruth, cocoTruth.loadRes(predictions), "bbox")
    evaluator.params.maxDets = [1, 10, maximumDetections]
    evaluator.evaluate()
    evaluator.accumulate()
    precision = evaluator.eval["precision"]
    return {
        "apCoco": averagePrecision(precision), "ap50Coco": averagePrecision(precision, iouIndex=0), "ap75Coco": averagePrecision(precision, iouIndex=5),
        "apSmall": averagePrecision(precision, "small"), "apMedium": averagePrecision(precision, "medium"), "apLarge": averagePrecision(precision, "large"),
    }


# Objects-per-image statistics of a split, to judge whether the 300-detection cap of the NMS-free head matters (Phase 1 check)
def countObjectsPerImage(imagePaths):
    from ultralytics.data.utils import img2label_paths
    counts = sorted(len(readYoloLabels(labelPath)) for labelPath in img2label_paths([str(path) for path in imagePaths]))
    if not counts: return {}
    return {
        "images": len(counts), "meanObjects": round(statistics.mean(counts), 1), "medianObjects": statistics.median(counts),
        "p95Objects": counts[int(0.95 * (len(counts) - 1))], "maxObjects": counts[-1],
        "imagesOverCap": sum(count > endToEndDetectionCap for count in counts),
        "objectsLostToCap": sum(max(count - endToEndDetectionCap, 0) for count in counts),
    }


# Full size-aware evaluation of one model: Ultralytics validation with saved predictions, then COCOeval by object size
# Returns one flat metric dict (Ultralytics mAP, COCO AP by size, agreement check) ready for a results row
def evaluateSmallObjects(weightsPath, dataYaml, split="val", imageSize=protocolImageSize, batch=16, device=None,
                         maximumDetections=cocoMaximumDetections, extraArguments=None):
    # The run folder is set explicitly because the metrics object of Ultralytics 8.4.175 does not say where predictions.json was saved
    outputFolder = Path(tempfile.mkdtemp(prefix="smallobject_val_"))
    arguments = {"save_json": True, "conf": lowConfidence, "max_det": maximumDetections, "project": str(outputFolder), "name": "val", "exist_ok": True}
    arguments.update(extraArguments or {})
    validationMetrics = evaluateDetector(weightsPath, dataYaml, split, imageSize, batch, device, arguments)
    imagePaths, classNames = resolveSplitImages(dataYaml, split)
    groundTruth = buildCocoGroundTruth(imagePaths, classNames)
    predictionsPath = outputFolder / "val" / "predictions.json"
    predictions = loadPredictions(predictionsPath, groundTruth) if predictionsPath.exists() else []
    metrics = {"map50": float(validationMetrics.box.map50), "map50_95": float(validationMetrics.box.map),
               "precision": float(validationMetrics.box.mp), "recall": float(validationMetrics.box.mr)}
    metrics.update(evaluateCoco(groundTruth, predictions, maximumDetections))
    metrics["cocoAgreesWithUltralytics"] = int(abs(metrics["apCoco"] - metrics["map50_95"]) <= agreementTolerance)
    if not metrics["cocoAgreesWithUltralytics"]:
        print(f"WARNING: COCO AP {metrics['apCoco']:.4f} and Ultralytics mAP50-95 {metrics['map50_95']:.4f} differ by more than {agreementTolerance}; check the label conversion.")
    return metrics


# Command line entry point
def main():
    parser = argparse.ArgumentParser(description="COCO-style AP by object size")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--data", default="VisDrone.yaml")
    parser.add_argument("--split", default="val")
    parser.add_argument("--imgsz", type=int, default=protocolImageSize)
    parser.add_argument("--max-det", type=int, default=cocoMaximumDetections)
    parser.add_argument("--stats-only", action="store_true", help="only print objects-per-image statistics of the split")
    arguments = parser.parse_args()
    if arguments.stats_only:
        print(countObjectsPerImage(resolveSplitImages(arguments.data, arguments.split)[0]))
        return
    for metricName, metricValue in evaluateSmallObjects(arguments.weights, arguments.data, arguments.split, arguments.imgsz, maximumDetections=arguments.max_det).items():
        print(f"{metricName}: {metricValue}")


if __name__ == "__main__":
    main()
