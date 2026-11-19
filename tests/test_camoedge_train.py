# Tests for the CamoEdge training helpers that need no dataset (camoedge/train.py)
import pytest

torch = pytest.importorskip("torch")

from camoedge.train import exportFinalWeights, makeVariantName  # noqa: E402


# The final weights are a slim copy of last.pt named <task>-<variant>-seed<k>.pt: model and settings are kept, the optimiser state is dropped,
# floating-point weights are stored in half precision and integer buffers keep their type
def testExportFinalWeightsKeepsModelAndSettingsOnly(tmp_path):
    settings = {"variantName": makeVariantName("E1"), "seed": 2, "imageSize": 352, "epochs": 1}
    modelState = {"layer.weight": torch.full((2, 2), 0.25), "norm.num_batches_tracked": torch.tensor(7)}
    checkpointPath = tmp_path / "E1-seed2" / "last.pt"
    checkpointPath.parent.mkdir()
    torch.save({"model": modelState, "optimiser": {"state": {}}, "scheduler": {}, "scaler": {}, "epoch": 1, "history": [], "settings": settings}, checkpointPath)
    weightsPath = exportFinalWeights(checkpointPath, tmp_path / "weights")
    exported = torch.load(weightsPath, map_location="cpu", weights_only=False)
    assert weightsPath == tmp_path / "weights" / "camoedge-E1-seed2.pt"
    assert set(exported) == {"model", "settings"} and exported["settings"] == settings
    assert exported["model"]["layer.weight"].dtype == torch.float16 and torch.equal(exported["model"]["layer.weight"].float(), modelState["layer.weight"])
    assert exported["model"]["norm.num_batches_tracked"].dtype == torch.int64


# Half-precision final weights load into a float32 model, as loadCheckpointModel does for the demo and camoedge.evaluate
def testHalfPrecisionWeightsLoadIntoFloatModel():
    model = torch.nn.Sequential(torch.nn.Linear(3, 2), torch.nn.BatchNorm1d(2))
    halfState = {name: tensor.half() if tensor.is_floating_point() else tensor for name, tensor in model.state_dict().items()}
    reloaded = torch.nn.Sequential(torch.nn.Linear(3, 2), torch.nn.BatchNorm1d(2))
    reloaded.load_state_dict(halfState)
    assert all(parameter.dtype == torch.float32 for parameter in reloaded.parameters())
    assert torch.allclose(reloaded[0].weight, model[0].weight, atol=1e-3)
