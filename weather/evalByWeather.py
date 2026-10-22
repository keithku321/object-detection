# Per-weather evaluation for Plan B: score a model separately on the fog, rain, snow and sand images of a held-out fold.
#   python -m weather.evalByWeather --weights last.pt --dataset /content/datasets/dawn --fold 0
from pathlib import Path

from common.datasets import writeAbsoluteImageList, writeDatasetYaml
from common.results import appendResultRow, buildResultRow, extractDetectionMetrics, findResultRow
from common.training import evaluateDetector, protocolImageSize
from weather.prepareDawn import getDawnSettings, heldOutImagesByWeather

# Name of this task in the results log
taskName = "weather"


# Write a dataset YAML whose validation set is the images of one weather type in one held-out fold
def writeWeatherYaml(datasetRoot, foldIndex, weatherType, imagePaths, classNames):
    datasetRoot = Path(datasetRoot)
    imageList = writeAbsoluteImageList(datasetRoot / "folds" / f"fold{foldIndex}_val_{weatherType}.txt", imagePaths)
    return writeDatasetYaml(datasetRoot, classNames, trainSplit=f"folds/{imageList.name}", valSplit=f"folds/{imageList.name}", yamlName=f"dawn-fold{foldIndex}-{weatherType}.yaml")


# Evaluate one model on every weather subset of a held-out fold; returns {weatherType: metric dict with an image count}
def evaluateByWeather(weightsPath, datasetRoot, foldIndex, imageSize=protocolImageSize, device=None):
    weatherTypes, classNames = getDawnSettings()
    metricsByWeather = {}
    for weatherType, imagePaths in heldOutImagesByWeather(datasetRoot, foldIndex, weatherTypes).items():
        if not imagePaths:
            print(f"fold {foldIndex} has no {weatherType} images; skipped")
            continue
        yamlPath = writeWeatherYaml(datasetRoot, foldIndex, weatherType, imagePaths, classNames)
        metrics = extractDetectionMetrics(evaluateDetector(weightsPath, yamlPath, "val", imageSize, device=device))
        metrics["images"] = len(imagePaths)
        metricsByWeather[weatherType] = metrics
    return metricsByWeather


# Evaluate by weather and append one results row per weather type; the variant column becomes "<variant>@<weather>"
# A fold that already has per-weather rows for this variant and seed is skipped, so cells can be re-run after a Colab restart
def logByWeather(modelName, variant, seed, weightsPath, datasetRoot, foldIndex, imageSize=protocolImageSize, device=None, notes=""):
    weatherTypes, _ = getDawnSettings()
    if any(findResultRow(taskName, modelName, f"{variant}@{weatherType}", seed, [f"fold={foldIndex}", notes]) for weatherType in weatherTypes):
        print(f"{variant} seed {seed} fold {foldIndex} already has per-weather rows; skipping")
        return {}
    metricsByWeather = evaluateByWeather(weightsPath, datasetRoot, foldIndex, imageSize, device)
    for weatherType, metrics in metricsByWeather.items():
        appendResultRow(buildResultRow(taskName, modelName, f"{variant}@{weatherType}", seed, metrics, notes=f"fold={foldIndex}; {notes}".strip("; ")))
    return metricsByWeather


# Command line entry point
def main():
    import argparse
    parser = argparse.ArgumentParser(description="Per-weather evaluation on a held-out DAWN fold")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--dataset", required=True, help="root of the prepared DAWN dataset")
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--imgsz", type=int, default=protocolImageSize)
    arguments = parser.parse_args()
    for weatherType, metrics in evaluateByWeather(arguments.weights, arguments.dataset, arguments.fold, arguments.imgsz).items():
        print(weatherType, metrics)


if __name__ == "__main__":
    main()
