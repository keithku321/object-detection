# Normalised Wasserstein Distance (NWD) box loss for small objects (experiment S5), after Wang et al., "A Normalized Gaussian
# Wasserstein Distance for Tiny Object Detection". IoU changes abruptly for tiny boxes (a shift of a few pixels can take it from
# high to zero), so each box is modelled as a 2D Gaussian and the distance between Gaussians is used as a smoother signal.
#
#   box (cx, cy, w, h)  ->  Gaussian with mean (cx, cy) and covariance diag(w^2 / 4, h^2 / 4)
#   W2^2 = (cx_a - cx_b)^2 + (cy_a - cy_b)^2 + ((w_a - w_b) / 2)^2 + ((h_a - h_b) / 2)^2
#   NWD  = exp(-sqrt(W2^2) / C)          C is a constant close to the typical object size in pixels
#   box loss = (1 - alpha) * CIoU loss + alpha * (1 - NWD)
#
# How it plugs into Ultralytics 8.4.175 without editing the library: E2ELoss accepts a loss class (loss_fn) and builds one copy for
# each YOLO26 head, so NwdDetectionLoss only has to swap the box-loss object, and NwdDetectionModel only has to override init_criterion.
# YOLO26 already has small-target-aware assignment (STAL), so expect a smaller gain than older papers report, and say so.
import torch
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils import RANK
from ultralytics.utils.loss import BboxLoss, E2ELoss, v8DetectionLoss

from common.registerModules import makeLayerMapTrainer

# Weight alpha of the NWD term in the box loss; the plan tries 0.3 and 0.5
defaultNwdWeight = 0.5
# Constant C in pixels; 12.8 is the value of the original paper, replace it with computeNwdConstant for your data and image size
defaultNwdConstant = 12.8
# Small number that keeps the square root differentiable at zero distance
distanceEpsilon = 1e-7


# NWD between matching rows of two (n, 4) tensors of boxes in xyxy pixel coordinates; returns a tensor of shape (n,) in (0, 1]
def normalisedWassersteinDistance(boxesA, boxesB, constant=defaultNwdConstant):
    centreA, centreB = (boxesA[:, :2] + boxesA[:, 2:]) / 2, (boxesB[:, :2] + boxesB[:, 2:]) / 2
    halfSizeA, halfSizeB = (boxesA[:, 2:] - boxesA[:, :2]) / 2, (boxesB[:, 2:] - boxesB[:, :2]) / 2
    squaredDistance = (centreA - centreB).pow(2).sum(-1) + (halfSizeA - halfSizeB).pow(2).sum(-1)
    return torch.exp(-torch.sqrt(squaredDistance + distanceEpsilon) / constant)


# Typical object size of a dataset in pixels at the training image size: the mean of sqrt(width * height) over all boxes
# imagePaths are the training images; each image is assumed to be resized so that its longer side equals imageSize
def computeNwdConstant(imagePaths, imageSize):
    from PIL import Image
    from ultralytics.data.utils import exif_size, img2label_paths
    from smallobject.evalCoco import readYoloLabels
    sizes = []
    for imagePath, labelPath in zip(imagePaths, img2label_paths([str(path) for path in imagePaths])):
        with Image.open(imagePath) as image:
            width, height = exif_size(image)
        scale = imageSize / max(width, height)
        sizes.extend(((boxWidth * width * scale) * (boxHeight * height * scale)) ** 0.5 for _, _, _, boxWidth, boxHeight in readYoloLabels(labelPath))
    return sum(sizes) / len(sizes) if sizes else defaultNwdConstant


# Box loss that mixes the CIoU loss of Ultralytics with the NWD loss; the L1 distance loss of YOLO26 is left unchanged
class NwdBboxLoss(BboxLoss):
    # Weight and constant are class attributes because Ultralytics builds the loss objects itself; makeNwdTrainer sets them
    nwdWeight = defaultNwdWeight
    nwdConstant = defaultNwdConstant

    # Same arguments as BboxLoss.forward; boxes arrive in grid units, so they are multiplied by the stride to get pixels for NWD
    def forward(self, pred_dist, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask, imgsz, stride):
        lossIou, lossDistance = super().forward(pred_dist, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask, imgsz, stride)
        positive = fg_mask.nonzero(as_tuple=True)
        if positive[0].numel() == 0: return lossIou, lossDistance
        weight = target_scores[positive].sum(-1)
        strideOfAnchor = stride[positive[1]]
        distance = normalisedWassersteinDistance(pred_bboxes[positive].float() * strideOfAnchor, target_bboxes[positive].float() * strideOfAnchor, type(self).nwdConstant)
        lossNwd = ((1.0 - distance) * weight).sum() / target_scores_sum
        return (1.0 - type(self).nwdWeight) * lossIou + type(self).nwdWeight * lossNwd, lossDistance


# Detection loss of Ultralytics with the box-loss object replaced by NwdBboxLoss; assignment and class loss are unchanged
class NwdDetectionLoss(v8DetectionLoss):
    # Build the normal loss, then swap in the NWD box loss on the same device
    def __init__(self, model, tal_topk=10, tal_topk2=None):
        super().__init__(model, tal_topk=tal_topk, tal_topk2=tal_topk2)
        self.bbox_loss = NwdBboxLoss(model.model[-1].reg_max).to(self.device)


# Detection model whose criterion uses the NWD loss for both YOLO26 heads (or for the single head of older model families)
class NwdDetectionModel(DetectionModel):
    # Called by Ultralytics on the first loss computation and again when a run is resumed
    def init_criterion(self):
        hasOneToOneHead = getattr(self.model[-1], "one2one_cv2", None) is not None
        return E2ELoss(self, NwdDetectionLoss) if hasOneToOneHead else NwdDetectionLoss(self)


# Build a trainer class that trains with the NWD loss; it also supports the layer map and weighted Concat options of
# common/registerModules.py makeLayerMapTrainer, so NWD can be combined with the architecture changes (experiment S6)
def makeNwdTrainer(nwdWeight=defaultNwdWeight, nwdConstant=defaultNwdConstant, layerMap=None, weightedConcatLayers=()):
    baseTrainer = makeLayerMapTrainer(layerMap, weightedConcatLayers)

    # Layer map trainer that builds NwdDetectionModel instead of the plain detection model
    class NwdTrainer(baseTrainer):
        # Set the NWD settings and build the model class that owns the NWD criterion
        def createModel(self, cfg, verbose):
            NwdBboxLoss.nwdWeight, NwdBboxLoss.nwdConstant = nwdWeight, nwdConstant
            return self.set_model_names_for_load(NwdDetectionModel(cfg, nc=self.data["nc"], ch=self.data["channels"], verbose=verbose and RANK == -1))

    return NwdTrainer
