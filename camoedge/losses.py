# Losses for CamoEdge (plans/04-camoedge.md, "Loss"):  total = sum over mask outputs [ weighted BCE + weighted IoU ] + lambda_e * edge loss
import torch
import torch.nn.functional as F

# Window of the average pool that finds boundary and hard pixels for the structure-loss weights
structureWindow = 31
# Extra weight given to boundary pixels by the structure loss: weight = 1 + boundaryEmphasis * |avgpool(mask) - mask|
boundaryEmphasis = 5.0
# Default weight lambda_e of the edge loss; sweep it in experiment E6
defaultEdgeLossWeight = 1.0
# Upper limit of the positive-class weight of the edge BCE, so a batch with almost no edge pixels cannot explode the loss
maximumEdgePositiveWeight = 50.0
# Loss settings of the experiment matrix: which mask loss and how much edge loss
experimentLossSettings = {
    "E1": {"maskLoss": "bce", "edgeLossWeight": 0.0},
    "E2": {"maskLoss": "structure", "edgeLossWeight": 0.0},
    "E3": {"maskLoss": "structure", "edgeLossWeight": 0.0},
    "E4": {"maskLoss": "structure", "edgeLossWeight": defaultEdgeLossWeight},
    "E5": {"maskLoss": "structure", "edgeLossWeight": defaultEdgeLossWeight},
}


# Plain binary cross-entropy on mask logits (the E1 baseline)
def plainBceLoss(logits, mask):
    return F.binary_cross_entropy_with_logits(logits, mask)


# Structure loss: weighted BCE + weighted IoU, where pixels near the mask boundary get up to six times the weight
# This is the loss used in earlier camouflaged-object and polyp segmentation work (F3Net, PraNet, SINet-V2)
def structureLoss(logits, mask):
    weight = 1.0 + boundaryEmphasis * torch.abs(F.avg_pool2d(mask, kernel_size=structureWindow, stride=1, padding=structureWindow // 2) - mask)
    weightedBce = (weight * F.binary_cross_entropy_with_logits(logits, mask, reduction="none")).sum(dim=(2, 3)) / weight.sum(dim=(2, 3))
    probability = torch.sigmoid(logits)
    intersection = (probability * mask * weight).sum(dim=(2, 3))
    union = ((probability + mask) * weight).sum(dim=(2, 3))
    weightedIou = 1.0 - (intersection + 1.0) / (union - intersection + 1.0)
    return (weightedBce + weightedIou).mean()


# Edge loss: BCE with a positive-class weight (edge pixels are under 5% of the image) plus a Dice term
def edgeLoss(edgeLogits, edgeTarget, useDice=True):
    positiveCount = edgeTarget.sum().clamp_min(1.0)
    positiveWeight = ((edgeTarget.numel() - positiveCount) / positiveCount).clamp(max=maximumEdgePositiveWeight)
    loss = F.binary_cross_entropy_with_logits(edgeLogits, edgeTarget, pos_weight=positiveWeight)
    if useDice:
        probability = torch.sigmoid(edgeLogits)
        intersection = (probability * edgeTarget).sum(dim=(2, 3))
        dice = 1.0 - (2.0 * intersection + 1.0) / (probability.sum(dim=(2, 3)) + edgeTarget.sum(dim=(2, 3)) + 1.0)
        loss = loss + dice.mean()
    return loss


# Total CamoEdge loss for one batch; outputs is the dict returned by CamoEdgeNet
# Returns the total loss and a dict of detached parts for logging
def camoEdgeLoss(outputs, mask, edge, maskLoss="structure", edgeLossWeight=defaultEdgeLossWeight):
    maskLossFunction = structureLoss if maskLoss == "structure" else plainBceLoss
    maskTerm = sum(maskLossFunction(logits.float(), mask) for logits in outputs["masks"])
    edgeTerm = edgeLoss(outputs["edge"].float(), edge) if outputs["edge"] is not None and edgeLossWeight > 0 else mask.new_zeros(())
    total = maskTerm + edgeLossWeight * edgeTerm
    return total, {"loss": float(total.detach()), "maskLoss": float(maskTerm.detach()), "edgeLoss": float(edgeTerm.detach())}
