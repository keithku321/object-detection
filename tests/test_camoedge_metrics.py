# Tests for the CamoEdge metrics, losses, edge ground truth and network (plans/04-camoedge.md asks for hand-made mask tests)
import pytest

numpy = pytest.importorskip("numpy")
pytest.importorskip("cv2")

from camoedge.metrics import boundaryFScore, extractBoundary, meanAbsoluteError  # noqa: E402


# A square object in the middle of an empty mask
def makeSquareMask(size=64, low=16, high=48):
    mask = numpy.zeros((size, size), dtype=bool)
    mask[low:high, low:high] = True
    return mask


# The boundary of a square is its one-pixel outline
def testBoundaryIsOutline():
    boundary = extractBoundary(makeSquareMask())
    assert boundary.sum() == 4 * 32 - 4 and boundary[16, 16] and not boundary[30, 30]


# A perfect prediction scores 1, a small shift inside the tolerance still scores 1, and a far shift scores 0
def testBoundaryFScore():
    trueMask = makeSquareMask()
    assert boundaryFScore(trueMask, trueMask) == 1.0
    assert boundaryFScore(makeSquareMask(low=17, high=49), trueMask, tolerance=2) == 1.0
    assert boundaryFScore(makeSquareMask(low=2, high=10), trueMask, tolerance=2) == 0.0
    assert boundaryFScore(numpy.zeros_like(trueMask), trueMask) == 0.0
    assert boundaryFScore(numpy.zeros_like(trueMask), numpy.zeros_like(trueMask)) == 1.0


# MAE is 0 for a perfect prediction and 1 for an inverted one
def testMeanAbsoluteError():
    trueMask = makeSquareMask()
    assert meanAbsoluteError(trueMask.astype(float), trueMask) == 0.0
    assert meanAbsoluteError((~trueMask).astype(float), trueMask) == 1.0


# With PySODMetrics: a perfect prediction gives S = 1 and MAE = 0, and an inverted prediction gives MAE = 1
def testSodMetricsOnHandMadeMasks():
    pytest.importorskip("py_sod_metrics")
    from camoedge.metrics import CamoMetrics
    trueMask = makeSquareMask()
    perfect, inverted = CamoMetrics(), CamoMetrics()
    perfect.update(trueMask.astype(float), trueMask)
    inverted.update((~trueMask).astype(float), trueMask)
    perfectScores, invertedScores = perfect.compute(), inverted.compute()
    assert perfectScores["sAlpha"] == pytest.approx(1.0, abs=1e-6) and perfectScores["mae"] == pytest.approx(0.0, abs=1e-6)
    assert perfectScores["fWeighted"] == pytest.approx(1.0, abs=1e-3) and perfectScores["boundaryF"] == 1.0
    assert invertedScores["mae"] == pytest.approx(1.0, abs=1e-6) and invertedScores["sAlpha"] < 0.1


# The edge ground truth is a thin band on the mask boundary and is empty for an empty mask
def testEdgeMap():
    pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    from camoedge.dataset import makeEdgeMap, superClassOf
    maskArray = makeSquareMask().astype(numpy.uint8) * 255
    edges = makeEdgeMap(maskArray, edgeWidth=3) > 0
    assert 0 < edges.mean() < 0.2 and edges[16, 30] and not edges[32, 32]
    assert makeEdgeMap(numpy.zeros((32, 32), dtype=numpy.uint8)).sum() == 0
    assert superClassOf("COD10K-CAM-1-Aquatic-1-BatFish-1.jpg") == "Aquatic" and superClassOf("camourflage_00012.jpg") == "unknown"


# The structure loss is lower for a good prediction than for a bad one, and the total loss has a gradient for every output
def testLossesPreferGoodPredictions():
    torch = pytest.importorskip("torch")
    from camoedge.losses import camoEdgeLoss, edgeLoss, structureLoss
    mask = torch.from_numpy(makeSquareMask()).float()[None, None]
    goodLogits, badLogits = (mask * 2 - 1) * 6, (1 - mask * 2) * 6
    assert structureLoss(goodLogits, mask) < 0.05 < structureLoss(badLogits, mask)
    edge = torch.zeros_like(mask)
    edge[..., 16, 16:48] = 1
    assert edgeLoss((edge * 2 - 1) * 6, edge) < edgeLoss(torch.zeros_like(edge), edge)
    outputs = {"masks": [torch.zeros_like(mask, requires_grad=True), torch.zeros_like(mask, requires_grad=True)], "edge": torch.zeros_like(mask, requires_grad=True)}
    total, parts = camoEdgeLoss(outputs, mask, edge, maskLoss="structure", edgeLossWeight=1.0)
    total.backward()
    assert all(tensor.grad is not None for tensor in outputs["masks"]) and outputs["edge"].grad is not None
    assert parts["loss"] == pytest.approx(parts["maskLoss"] + parts["edgeLoss"], rel=1e-5)


# Every experiment builds, returns four mask outputs at the input size, and has an edge output only when it has an edge branch
@pytest.mark.parametrize("experimentName,encoderName", [("E1", "resnet50"), ("E3", "resnet50"), ("E5", "resnet50"), ("E5", "pvt_v2_b2")])
def testModelOutputs(experimentName, encoderName):
    torch = pytest.importorskip("torch")
    pytest.importorskip("timm")
    from camoedge.model import buildExperimentModel
    model = buildExperimentModel(experimentName, encoderName, pretrained=False).eval()
    with torch.no_grad():
        outputs = model(torch.randn(1, 3, 96, 96))
    assert len(outputs["masks"]) == 4 and all(logits.shape == (1, 1, 96, 96) for logits in outputs["masks"])
    assert (outputs["edge"] is None) == (experimentName == "E1")
    if outputs["edge"] is not None: assert outputs["edge"].shape == (1, 1, 96, 96)
