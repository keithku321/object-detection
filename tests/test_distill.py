# Tests for WeatherDistill (weather/distillTrainer.py). The fast tests check the loss terms; the slow tests are the correctness
# checks the plan asks for before any real run: the identity check, the lambda = 0 check and a one-epoch smoke test on 8 images.
import copy

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("ultralytics")

from weather.distillTrainer import (WeatherDistillationModel, binaryKlDivergence, defaultDistillSettings, distillPresets, makeForegroundMask,  # noqa: E402
                                     splitDistillSettings, useWeatherDistillation)


# The binary KL divergence is zero for equal logits, positive otherwise, and scaled by the temperature squared
def testBinaryKlDivergence():
    logits = torch.tensor([[[2.0, -1.0, 0.5]]])
    assert torch.allclose(binaryKlDivergence(logits, logits, 2.0), torch.zeros_like(logits), atol=1e-7)
    different = binaryKlDivergence(torch.zeros_like(logits), logits, 1.0)
    assert (different > 0).all()
    probability = torch.sigmoid(logits)
    expected = probability * torch.log(probability / 0.5) + (1 - probability) * torch.log((1 - probability) / 0.5)
    assert torch.allclose(different, expected, atol=1e-6)
    assert torch.allclose(binaryKlDivergence(logits * 2, logits * 2, 2.0), torch.zeros_like(logits), atol=1e-7)


# The foreground mask marks the cells inside the enlarged ground-truth box of the right image and nothing else
def testForegroundMask():
    batch = {"img": torch.zeros(2, 3, 64, 64), "bboxes": torch.tensor([[0.5, 0.5, 0.25, 0.25]]), "batch_idx": torch.tensor([1.0])}
    mask = makeForegroundMask(batch, (8, 8), dilation=1.0)
    assert mask.shape == (2, 1, 8, 8) and mask[0].sum() == 0
    assert mask[1, 0, 3:5, 3:5].all() and mask[1].sum() == 4
    assert makeForegroundMask(batch, (8, 8), dilation=2.0)[1].sum() > 4
    emptyBatch = {"img": torch.zeros(1, 3, 64, 64), "bboxes": torch.zeros(0, 4), "batch_idx": torch.zeros(0)}
    assert makeForegroundMask(emptyBatch, (4, 4)).sum() == 0


# Presets split into the feature weight (Ultralytics' dis argument) and the settings of the extra terms; overrides win
def testPresetSettings():
    lambdaFeature, settings = splitDistillSettings("full", {"temperature": 4.0})
    assert lambdaFeature == 6.0 and settings == {"lambdaClass": 1.0, "lambdaBox": 1.0, "temperature": 4.0}
    assert splitDistillSettings("lambda0") == (0.0, {})
    assert set(distillPresets) >= {"lambda0", "feat", "cls", "box", "full"}


# The context manager swaps the model class in the Ultralytics trainer and always puts the original back
def testContextManagerRestoresClass():
    import ultralytics.engine.trainer as trainerModule
    original = trainerModule.DistillationModel
    with useWeatherDistillation({"lambdaClass": 2.0}):
        assert trainerModule.DistillationModel is WeatherDistillationModel
        assert WeatherDistillationModel.settings == {**defaultDistillSettings, "lambdaClass": 2.0}
    assert trainerModule.DistillationModel is original


# Build a distillation wrapper around two copies of one pretrained network, with BatchNorm frozen so both see the same statistics
def makeSelfDistillationModel(settings):
    from ultralytics import YOLO
    from ultralytics.cfg import get_cfg
    teacher = YOLO("yolo26n.pt").model.float()
    student = copy.deepcopy(teacher)
    for parameter in student.parameters(): parameter.requires_grad_(True)
    student.args = get_cfg(overrides={"imgsz": 64, "dis": 6.0, "epochs": 2})
    WeatherDistillationModel.settings = {**defaultDistillSettings, **settings}
    wrapper = WeatherDistillationModel(teacher_model=teacher, student_model=student)
    wrapper.train()
    for module in wrapper.student_model.modules():
        if isinstance(module, torch.nn.BatchNorm2d): module.eval()
    return wrapper


# A small batch with two boxes, in the format the Ultralytics dataloader produces
def makeBatch():
    torch.manual_seed(0)
    return {"img": torch.rand(2, 3, 64, 64), "cls": torch.tensor([[0.0], [2.0]]), "bboxes": torch.tensor([[0.5, 0.5, 0.4, 0.4], [0.3, 0.6, 0.3, 0.5]]),
            "batch_idx": torch.tensor([0.0, 1.0])}


# Identity check: a teacher used as its own student gives class and box distillation losses near zero
@pytest.mark.slow
def testIdentityGivesZeroHeadTerms():
    wrapper = makeSelfDistillationModel({"lambdaClass": 1.0, "lambdaBox": 1.0, "head": "both"})
    loss, lossItems = wrapper.loss(makeBatch())
    assert loss.shape == (6,) and torch.isfinite(loss).all()
    assert float(lossItems["cls_kd_loss"]) < 1e-4 and float(lossItems["box_kd_loss"]) < 1e-4


# A student that differs from the teacher gets positive class and box terms, and they have gradients back to the student
@pytest.mark.slow
def testDifferentStudentGetsPositiveTerms():
    wrapper = makeSelfDistillationModel({"lambdaClass": 1.0, "lambdaBox": 1.0, "foregroundMask": True, "normaliseFeatures": True, "useGiou": True})
    with torch.no_grad():
        for parameter in wrapper.student_model.model[-1].parameters(): parameter.add_(0.05 * torch.randn_like(parameter))
    loss, lossItems = wrapper.loss(makeBatch())
    assert float(lossItems["cls_kd_loss"]) > 0 and float(lossItems["box_kd_loss"]) >= 0
    loss.sum().backward()
    assert any(parameter.grad is not None and parameter.grad.abs().sum() > 0 for parameter in wrapper.student_model.model[-1].parameters())
    wrapper.eval()
    validationLoss, validationItems = wrapper.loss(makeBatch())
    assert validationLoss.shape == (6,) and float(validationItems["cls_kd_loss"]) == 0.0


# lambda = 0: with every weight at zero the extra terms are exactly zero, so only the plain detection loss trains the student
@pytest.mark.slow
def testLambdaZeroAddsNothing():
    wrapper = makeSelfDistillationModel({"lambdaClass": 0.0, "lambdaBox": 0.0})
    wrapper.dis = 0.0
    loss, lossItems = wrapper.loss(makeBatch())
    assert float(loss[3]) == 0.0 and float(loss[4]) == 0.0 and float(loss[5]) == 0.0
    assert float(lossItems["dis_loss"]) == 0.0


# One-epoch smoke test on the 8 images of coco8: training with every term runs, and the final weights are a plain student
@pytest.mark.slow
def testOneEpochSmokeAndStudentSize(tmp_path):
    from common.benchmark import countParametersAndGflops
    from weather.distillTrainer import trainDistilledStudent
    bestWeights = trainDistilledStudent("yolo26n.pt", "yolo26s.pt", "coco8.yaml", seed=0, epochs=1, project=tmp_path, runName="smoke", presetName="full",
                                        imageSize=64, batch=4, device="cpu", workers=0)
    assert bestWeights.exists()
    assert countParametersAndGflops(bestWeights)["parameters"] == countParametersAndGflops("yolo26n.pt")["parameters"]
