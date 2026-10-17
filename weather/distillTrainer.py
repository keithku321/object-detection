# WeatherDistill training objectives (plans/02-weather-distill.md), built on the knowledge distillation that Ultralytics 8.4.175 already ships.
#
# What Ultralytics provides (model.train(distill_model="teacher.pt", dis=6.0)): a DistillationModel that runs a frozen teacher on the same
# augmented batch, hooks the three neck feature maps and the Detect head, passes the student features through a small projector
# (1x1 conv, ReLU, 1x1 conv) and adds a teacher-confidence-weighted L2 loss. The projector and teacher are stripped from the final best.pt.
# That is the feature term (L_feat) of the plan, so "feature distillation only" (W3) needs no custom code.
#
# What this file adds:
#   L_cls  temperature-softened binary KL between teacher and student class logits (class scores are independent sigmoids, not a softmax)
#   L_box  L1 (optionally GIoU) between student and teacher boxes on the anchors the student's assigner marks positive, weighted by
#          teacher confidence; this replaces the DFL-KL of the proposal, because YOLO26 has no DFL (reg_max = 1)
#   options: restrict L_feat to ground-truth regions (W6), normalise features, apply the terms to the one-to-many head, the
#          one-to-one head or both
# It works by subclassing DistillationModel and swapping it in for the duration of training (useWeatherDistillation).
from contextlib import contextmanager

import torch
import torch.nn.functional as F

from common.training import protocolImageSize, trainDetector
from ultralytics.nn.distill_model import DistillationModel
from ultralytics.utils.metrics import bbox_iou
from ultralytics.utils.tal import dist2bbox

# Floor for teacher-confidence weights, so anchors where the teacher sees only background still get a small share of the class loss
classWeightFloor = 0.02
# Small number that protects divisions by a weight sum
weightEpsilon = 1e-9
# Settings of a plain run with all extra terms off, which behaves like Ultralytics' built-in feature distillation
defaultDistillSettings = {
    "lambdaClass": 0.0,          # weight of the classification KL term
    "lambdaBox": 0.0,            # weight of the box regression term
    "temperature": 2.0,          # softening of the class logits before the KL
    "head": "one2many",          # which head gets the class and box terms: "one2many", "one2one" or "both"
    "useGiou": False,            # add a GIoU term to the box L1 distance
    "foregroundMask": False,     # restrict the feature loss to ground-truth boxes
    "foregroundDilation": 1.2,   # ground-truth boxes are enlarged by this factor for the mask
    "normaliseFeatures": False,  # normalise each feature channel (instance norm) before the feature loss
}
# Named experiment settings; "lambdaFeature" is Ultralytics' dis argument, the weight of the feature term
distillPresets = {
    "lambda0": {"lambdaFeature": 0.0},
    "feat": {"lambdaFeature": 6.0},
    "cls": {"lambdaFeature": 0.0, "lambdaClass": 1.0},
    "box": {"lambdaFeature": 0.0, "lambdaBox": 1.0},
    "full": {"lambdaFeature": 6.0, "lambdaClass": 1.0, "lambdaBox": 1.0},
    "full-fg": {"lambdaFeature": 6.0, "lambdaClass": 1.0, "lambdaBox": 1.0, "foregroundMask": True},
    "full-both-heads": {"lambdaFeature": 6.0, "lambdaClass": 1.0, "lambdaBox": 1.0, "head": "both"},
}


# Binary mask (batch, 1, height, width) that marks feature cells inside ground-truth boxes enlarged by the dilation factor
def makeForegroundMask(batch, featureSize, dilation=1.2):
    height, width = featureSize
    mask = torch.zeros(batch["img"].shape[0], 1, height, width, device=batch["img"].device)
    boxes = batch["bboxes"].float()
    if boxes.numel() == 0: return mask
    left = ((boxes[:, 0] - boxes[:, 2] * dilation / 2) * width).floor().clamp(0, width - 1).long()
    right = torch.maximum(((boxes[:, 0] + boxes[:, 2] * dilation / 2) * width).ceil().clamp(1, width).long(), left + 1)
    top = ((boxes[:, 1] - boxes[:, 3] * dilation / 2) * height).floor().clamp(0, height - 1).long()
    bottom = torch.maximum(((boxes[:, 1] + boxes[:, 3] * dilation / 2) * height).ceil().clamp(1, height).long(), top + 1)
    for imageIndex, x1, x2, y1, y2 in zip(batch["batch_idx"].long().tolist(), left.tolist(), right.tolist(), top.tolist(), bottom.tolist()):
        mask[imageIndex, 0, y1:y2, x1:x2] = 1.0
    return mask


# Binary KL divergence KL(teacher || student) per class and anchor for independent sigmoid outputs, softened by a temperature
# Inputs are raw logits of shape (batch, classes, anchors); the result has the same shape and is scaled by temperature squared
def binaryKlDivergence(studentLogits, teacherLogits, temperature):
    student, teacher = studentLogits.float() / temperature, teacherLogits.float() / temperature
    teacherProbability = torch.sigmoid(teacher)
    divergence = teacherProbability * (F.logsigmoid(teacher) - F.logsigmoid(student)) + (1 - teacherProbability) * (F.logsigmoid(-teacher) - F.logsigmoid(-student))
    return divergence * temperature ** 2


# Extends Ultralytics' DistillationModel with the classification and box distillation terms and optional feature-loss variants
class WeatherDistillationModel(DistillationModel):
    # Settings are class attributes because Ultralytics builds the instance itself; useWeatherDistillation sets them before training
    settings = dict(defaultDistillSettings)

    # Drop the per-step batch reference before pickling so checkpoints do not carry image tensors
    def __getstate__(self):
        state = super().__getstate__()
        state["_currentBatch"] = None
        return state

    # Feature loss of the parent class, with optional normalisation and ground-truth masking
    def loss_sl2(self, student_feat, teacher_feat, feat_idx, teacher_scores):
        settings = type(self).settings
        if settings["normaliseFeatures"]:
            student_feat, teacher_feat = F.instance_norm(student_feat), F.instance_norm(teacher_feat)
        teacherScore = teacher_scores[feat_idx]
        currentBatch = getattr(self, "_currentBatch", None)
        if settings["foregroundMask"] and currentBatch is not None:
            teacherScore = teacherScore * makeForegroundMask(currentBatch, student_feat.shape[-2:], settings["foregroundDilation"]).flatten(2)
        batchSize, channels = student_feat.shape[:2]
        squaredError = F.mse_loss(student_feat.reshape(batchSize, channels, -1), teacher_feat.reshape(batchSize, channels, -1), reduction="none")
        return (squaredError * teacherScore).sum() / (teacherScore.sum() * channels + weightEpsilon)

    # Class and box terms for one head; returns two scalar tensors (already weighted by the lambdas)
    def computeHeadTerms(self, batch, headName):
        settings = type(self).settings
        levelCount = len(self.feats_idx) - 1
        studentHead = self.decouple_outputs(self._student_feats[levelCount], branch=headName)
        teacherHead = self.decouple_outputs(self._teacher_feats[levelCount], branch=headName)
        zero = studentHead["scores"].sum() * 0.0
        teacherScores = teacherHead["scores"].detach()
        teacherConfidence = teacherScores.sigmoid().max(dim=1).values  # batch, anchors
        classTerm, boxTerm = zero, zero
        if settings["lambdaClass"] > 0:
            weights = teacherConfidence.clamp_min(classWeightFloor)
            divergence = binaryKlDivergence(studentHead["scores"], teacherScores, settings["temperature"]).sum(dim=1)
            classTerm = settings["lambdaClass"] * (divergence * weights).sum() / (weights.sum() + weightEpsilon)
        if settings["lambdaBox"] > 0:
            boxTerm = settings["lambdaBox"] * self.computeBoxTerm(batch, headName, studentHead, teacherHead, teacherConfidence, zero)
        return classTerm, boxTerm

    # Box distillation on the anchors that the student's assigner marks positive; boxes are compared in image-normalised coordinates
    def computeBoxTerm(self, batch, headName, studentHead, teacherHead, teacherConfidence, zero):
        detectHead = self.student_model.model[-1]
        if detectHead.reg_max != 1: raise ValueError("Box distillation needs a head without DFL (reg_max = 1, as in YOLO26)")
        criterion = self.student_model.criterion
        detectionLoss = getattr(criterion, headName, criterion)  # E2ELoss keeps one loss object per head
        with torch.no_grad():
            (foregroundMask, _, _, anchorPoints, strideTensor), _, _ = detectionLoss.get_assigned_targets_and_loss(studentHead, batch)
        if not foregroundMask.any(): return zero
        imageSize = torch.tensor(studentHead["feats"][0].shape[2:], device=anchorPoints.device, dtype=torch.float32) * detectHead.stride[0]
        normaliser = imageSize[[1, 0, 1, 0]]
        studentBoxes = dist2bbox(studentHead["boxes"].float().permute(0, 2, 1), anchorPoints, xywh=False) * strideTensor / normaliser
        teacherBoxes = dist2bbox(teacherHead["boxes"].float().detach().permute(0, 2, 1), anchorPoints, xywh=False) * strideTensor / normaliser
        weights = teacherConfidence[foregroundMask]
        distance = (studentBoxes[foregroundMask] - teacherBoxes[foregroundMask]).abs().mean(-1)
        if type(self).settings["useGiou"]:
            distance = distance + (1.0 - bbox_iou(studentBoxes[foregroundMask], teacherBoxes[foregroundMask], xywh=False, GIoU=True).squeeze(-1))
        return (distance * weights).sum() / (weights.sum() + weightEpsilon)

    # Detection loss and built-in feature loss from the parent, plus the class and box terms; loss and loss_items gain two entries
    def loss(self, batch, preds=None):
        settings = type(self).settings
        self._currentBatch = batch if self.training else None
        try:
            totalLoss, lossItems = super().loss(batch, preds)
            zero = torch.zeros((), device=batch["img"].device)
            classTerm, boxTerm = zero, zero
            wantsHeadTerms = settings["lambdaClass"] > 0 or settings["lambdaBox"] > 0
            if self.training and wantsHeadTerms:
                headNames = ("one2many", "one2one") if settings["head"] == "both" else (settings["head"],)
                for headName in headNames:
                    headClass, headBox = self.computeHeadTerms(batch, headName)
                    classTerm, boxTerm = classTerm + headClass, boxTerm + headBox
            lossItems["cls_kd_loss"], lossItems["box_kd_loss"] = classTerm.detach(), boxTerm.detach()
            batchSize = batch["img"].shape[0]
            return torch.cat([totalLoss, (classTerm * batchSize).reshape(1), (boxTerm * batchSize).reshape(1)]), lossItems
        finally:
            self._currentBatch = None


# Swap Ultralytics' DistillationModel for WeatherDistillationModel while training runs, with the given settings
# The swap must stay active for the whole call, including a resume, because the trainer rebuilds the model from the checkpoint
@contextmanager
def useWeatherDistillation(settings=None):
    import ultralytics.engine.trainer as trainerModule
    originalClass = trainerModule.DistillationModel
    WeatherDistillationModel.settings = {**defaultDistillSettings, **(settings or {})}
    trainerModule.DistillationModel = WeatherDistillationModel
    try:
        yield
    finally:
        trainerModule.DistillationModel = originalClass


# Split a preset or custom setting dict into (lambdaFeature, settings for WeatherDistillationModel)
def splitDistillSettings(presetName=None, overrides=None):
    merged = {**(distillPresets[presetName] if presetName else {}), **(overrides or {})}
    lambdaFeature = merged.pop("lambdaFeature", 6.0)
    return lambdaFeature, merged


# Train a student with distillation from a teacher checkpoint; resumable like every other run (see common/training.py)
# presetName picks a row of distillPresets and overrides changes single settings (for the lambda and temperature sweeps)
def trainDistilledStudent(studentSource, teacherWeights, dataYaml, seed, epochs, project, runName, presetName="full", overrides=None,
                          imageSize=protocolImageSize, batch=16, device=None, workers=None, pretrained=True):
    lambdaFeature, settings = splitDistillSettings(presetName, overrides)
    with useWeatherDistillation(settings):
        return trainDetector(studentSource, dataYaml, seed, epochs, project, runName, imageSize, batch, device, workers,
                             pretrained=pretrained, trainArguments={"distill_model": str(teacherWeights), "dis": lambdaFeature})
