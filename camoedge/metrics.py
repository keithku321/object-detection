# Segmentation metrics for CamoEdge: S-measure, E-measure, weighted F-measure and MAE from the PySODMetrics package, plus a
# boundary F-score written here. Global scores hide boundary gains because edge pixels are under 5% of a mask, and boundary quality
# is exactly what CamoEdge claims to improve, so the boundary F-score is the metric that tests the thesis directly.
# PySODMetrics: pip install pysodmetrics, import py_sod_metrics. Note its version in the report (see common/results.py getLibraryVersion).
import cv2
import numpy as np

# Distance in pixels within which a predicted boundary pixel counts as matching a true boundary pixel
defaultBoundaryTolerance = 2
# Threshold that turns a probability map into a binary mask for the boundary F-score
defaultBinaryThreshold = 0.5


# Boundary pixels of a binary mask: the mask minus its erosion; returns a boolean array
def extractBoundary(binaryMask):
    binaryMask = np.asarray(binaryMask).astype(np.uint8)
    eroded = cv2.erode(binaryMask, np.ones((3, 3), np.uint8), borderType=cv2.BORDER_CONSTANT, borderValue=0)
    return (binaryMask - eroded).astype(bool)


# Boundary F-score of one prediction: precision and recall of predicted boundary pixels against true boundary pixels within a tolerance
# Both inputs are binary masks of the same shape. Two empty masks score 1; exactly one empty mask scores 0.
def boundaryFScore(predictedMask, trueMask, tolerance=defaultBoundaryTolerance):
    predictedBoundary, trueBoundary = extractBoundary(predictedMask), extractBoundary(trueMask)
    if not predictedBoundary.any() and not trueBoundary.any(): return 1.0
    if not predictedBoundary.any() or not trueBoundary.any(): return 0.0
    distanceToTrue = cv2.distanceTransform((~trueBoundary).astype(np.uint8), cv2.DIST_L2, 3)
    distanceToPredicted = cv2.distanceTransform((~predictedBoundary).astype(np.uint8), cv2.DIST_L2, 3)
    precision = float((distanceToTrue[predictedBoundary] <= tolerance).mean())
    recall = float((distanceToPredicted[trueBoundary] <= tolerance).mean())
    return 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)


# Mean absolute error between a probability map in 0..1 and a binary mask; needs no extra package
def meanAbsoluteError(probabilityMap, trueMask):
    return float(np.abs(np.asarray(probabilityMap, dtype=np.float64) - (np.asarray(trueMask) > 0)).mean())


# Import PySODMetrics with a helpful message when it is missing (it has no Python 3.14 wheels; see requirements-local.txt)
def importSodMetrics():
    try:
        import py_sod_metrics
    except ImportError as error:
        raise ImportError("PySODMetrics is not installed. Run 'pip install pysodmetrics' (Python 3.14: see requirements-local.txt).") from error
    return py_sod_metrics


# Accumulates the metrics over a test set; call update for every image, then compute
class CamoMetrics:
    # Create the PySODMetrics accumulators and the list for the boundary scores
    def __init__(self, boundaryTolerance=defaultBoundaryTolerance, binaryThreshold=defaultBinaryThreshold):
        sodMetrics = importSodMetrics()
        self.structure, self.enhanced = sodMetrics.Smeasure(), sodMetrics.Emeasure()
        self.weightedF, self.absoluteError = sodMetrics.WeightedFmeasure(), sodMetrics.MAE()
        self.boundaryTolerance, self.binaryThreshold = boundaryTolerance, binaryThreshold
        self.boundaryScores = []

    # Add one image: probabilityMap is a float array in 0..1 and trueMask a binary array, both at the original image size
    def update(self, probabilityMap, trueMask):
        prediction = (np.clip(probabilityMap, 0.0, 1.0) * 255).astype(np.uint8)
        truth = (np.asarray(trueMask) > 0).astype(np.uint8) * 255
        for metric in (self.structure, self.enhanced, self.weightedF, self.absoluteError): metric.step(pred=prediction, gt=truth)
        self.boundaryScores.append(boundaryFScore(np.asarray(probabilityMap) >= self.binaryThreshold, truth > 0, self.boundaryTolerance))

    # Final scores: S_alpha, mean and maximum E_phi, weighted F_beta, MAE and the mean boundary F-score
    def compute(self):
        enhancedCurve = self.enhanced.get_results()["em"]["curve"]
        return {
            "sAlpha": float(self.structure.get_results()["sm"]), "eMean": float(enhancedCurve.mean()), "eMax": float(enhancedCurve.max()),
            "fWeighted": float(self.weightedF.get_results()["wfm"]), "mae": float(self.absoluteError.get_results()["mae"]),
            "boundaryF": float(np.mean(self.boundaryScores)) if self.boundaryScores else 0.0, "images": len(self.boundaryScores),
        }
