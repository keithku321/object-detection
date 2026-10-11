# Slow integration tests against the pinned Ultralytics: custom layers in a real YOLO26 model, pretrained transfer through the layer
# map, one-epoch training of the Plan A and Plan C variants, the speed protocol, and which head validation uses by default.
# Run with:   pytest -m slow          (downloads yolo26n.pt and the 8-image coco8 dataset on first use; runs on CPU)
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("ultralytics")

pytestmark = pytest.mark.slow


# Arguments that keep a training run tiny enough for a CPU
def tinyTraining(tmp_path, runName):
    return dict(dataYaml="coco8.yaml", seed=0, epochs=1, project=tmp_path, runName=runName, imageSize=64, batch=4, device="cpu", workers=0)


# Every attention variant of Plan A builds, runs forward, and receives all pretrained tensors of the layers it shares with yolo26n
@pytest.mark.parametrize("variantName", ["cbam-a", "eca-c", "simam-b", "ca-a"])
def testAttentionVariantTransfersPretrainedWeights(variantName, tmp_path, monkeypatch):
    import underwater.train as underwaterTrain
    from common.registerModules import loadPretrainedWithLayerMap, registerCustomModules
    from ultralytics import YOLO
    from ultralytics.nn.tasks import DetectionModel
    registerCustomModules()
    monkeypatch.setattr(underwaterTrain, "configFolder", tmp_path)
    yamlPath, layerMap = underwaterTrain.buildAttentionVariant(variantName, "n")
    variant = DetectionModel(str(yamlPath), nc=80, verbose=False)
    pretrained = YOLO("yolo26n.pt").model.float()
    loadPretrainedWithLayerMap(variant, pretrained, layerMap)
    for baseIndex, variantIndex in layerMap.items():
        sourceState, targetState = pretrained.model[baseIndex].state_dict(), variant.model[variantIndex].state_dict()
        assert sourceState.keys() == targetState.keys(), f"layer {baseIndex}"
        assert all(torch.equal(sourceState[key].float(), targetState[key].float()) for key in sourceState), f"layer {baseIndex} was not transferred"
    variant.eval()
    with torch.no_grad():
        assert variant(torch.zeros(1, 3, 64, 64))[0].shape[0] == 1


# Without the layer map, name-based loading pairs shifted layers wrongly or not at all; this is the failure the map prevents
def testNameBasedLoadingMissesShiftedLayers(tmp_path, monkeypatch):
    import underwater.train as underwaterTrain
    from common.registerModules import registerCustomModules
    from ultralytics import YOLO
    from ultralytics.nn.tasks import DetectionModel
    registerCustomModules()
    monkeypatch.setattr(underwaterTrain, "configFolder", tmp_path)
    yamlPath, layerMap = underwaterTrain.buildAttentionVariant("eca-a", "n")
    variant = DetectionModel(str(yamlPath), nc=80, verbose=False)
    pretrained = YOLO("yolo26n.pt").model.float()
    variant.load(pretrained)
    lastBaseLayerWithWeights = 22
    sourceState, targetState = pretrained.model[lastBaseLayerWithWeights].state_dict(), variant.model[layerMap[lastBaseLayerWithWeights]].state_dict()
    assert not all(torch.equal(sourceState[key].float(), targetState[key].float()) for key in sourceState)


# One epoch of Plan A training with an inserted attention layer runs end to end and logs a results row
def testUnderwaterVariantTrainsAndLogs(tmp_path, monkeypatch):
    import common.results as results
    import underwater.train as underwaterTrain
    monkeypatch.setattr(underwaterTrain, "configFolder", tmp_path / "configs")
    monkeypatch.setattr(results, "resultsFolder", tmp_path / "results")
    monkeypatch.setattr("common.training.appendResultRow", lambda row: results.appendResultRow(row, tmp_path / "results"))
    weightsPath = underwaterTrain.runVariant("eca-a", seed=0, epochs=1, dataYaml="coco8.yaml", project=tmp_path, imageSize=64, batch=4, device="cpu", workers=0)
    rows = results.readResultRows(tmp_path / "results" / "underwater.csv")
    assert weightsPath.name == "underwater-eca-a-seed0.pt" and weightsPath.exists()
    assert len(rows) == 1 and rows[0]["variant"] == "eca-a" and float(rows[0]["parameters"]) > 0 and "head=one2one" in rows[0]["notes"]


# The combined Plan C model (P2 head, attention, DySample, weighted fusion, NWD loss) builds through its trainer and trains one epoch
def testCombinedSmallObjectVariantTrains(tmp_path, monkeypatch):
    import smallobject.train as smallObjectTrain
    from common.modules import DySample, WeightedConcat
    from common.registerModules import registerCustomModules
    from common.training import trainDetector
    from smallobject.nwdLoss import NwdDetectionModel
    from ultralytics import YOLO
    registerCustomModules()
    monkeypatch.setattr(smallObjectTrain, "configFolder", tmp_path / "configs")
    variant = smallObjectTrain.resolveVariant("p2+eca+dysample+wfusion+nwd", "n", nwdWeight=0.5, nwdConstant=12.8)
    bestWeights = trainDetector(variant["modelSource"], trainerClass=variant["trainerClass"], pretrained=variant["pretrained"], **tinyTraining(tmp_path, "combined"))
    network = YOLO(str(bestWeights)).model
    assert isinstance(network, NwdDetectionModel) and network.model[-1].nl == 4
    assert any(isinstance(layer, DySample) for layer in network.model) and any(isinstance(layer, WeightedConcat) for layer in network.model)


# The NWD loss alone on the baseline architecture trains one epoch from the pretrained checkpoint
def testNwdBaselineTrains(tmp_path):
    from common.training import trainDetector
    from smallobject.train import resolveVariant
    variant = resolveVariant("nwd", "n")
    assert trainDetector(variant["modelSource"], trainerClass=variant["trainerClass"], pretrained=variant["pretrained"], **tinyTraining(tmp_path, "nwd")).exists()


# A run that already finished is skipped, and an interrupted run is found as resumable
def testFinishedRunIsSkipped(tmp_path):
    from common.training import findResumableCheckpoint, trainDetector
    arguments = tinyTraining(tmp_path, "baseline")
    bestWeights = trainDetector("yolo26n.pt", **arguments)
    modifiedTime = bestWeights.stat().st_mtime
    assert findResumableCheckpoint(tmp_path / "baseline") is None
    assert trainDetector("yolo26n.pt", **arguments).stat().st_mtime == modifiedTime


# The speed protocol runs in both modes and counts parameters of the fused inference network
def testBenchmarkModes():
    from common.benchmark import countParametersAndGflops, measureLatency
    for mode in ("end2end", "nms"):
        metrics = measureLatency("yolo26n.pt", mode=mode, device="cpu", imageSize=64, warmupIterations=1, timedIterations=3)
        assert metrics["mode"] == mode and metrics["fps"] > 0 and metrics["precision"] == "fp32"
    complexity = countParametersAndGflops("yolo26n.pt", 640)
    assert 2_000_000 < complexity["parameters"] < 3_000_000 and complexity["gflops"] > 0


# In the pinned Ultralytics, prediction uses the one-to-many head with NMS unless nms=False is passed; common/training.py relies on this
def testDefaultHeadIsNotEndToEnd():
    import numpy
    from ultralytics import YOLO
    image = numpy.zeros((64, 64, 3), dtype=numpy.uint8)
    defaultModel = YOLO("yolo26n.pt")
    defaultModel.predict(image, imgsz=64, device="cpu", verbose=False)
    endToEndModel = YOLO("yolo26n.pt")
    endToEndModel.predict(image, imgsz=64, device="cpu", verbose=False, nms=False)
    assert bool(defaultModel.predictor.model.end2end) is False
    assert bool(endToEndModel.predictor.model.end2end) is True
