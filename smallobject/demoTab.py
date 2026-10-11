# Demo tab for Plan C: small-object detection on drone images, comparing the baseline with the combined model.
# Edit demoWeights after the final runs, using local paths or release URLs. Small objects need the full image: use large inputs.
from common.demoSupport import buildDetectionTab

# Weights shown in the model dropdown; names follow <task>-<variant>-seed<k>.pt (plans/05-team-workflow.md)
demoWeights = {
    "Baseline (YOLO26n)": "weights/smallobject-baseline-seed0.pt",
    "Proposed (P2 head and extras)": "weights/smallobject-p2-seed0.pt",
}


# Build the Gradio tab that demo/app.py mounts
def buildTab():
    return buildDetectionTab(
        "Small objects (VisDrone)",
        "Pedestrians, vehicles and other small targets seen from a drone. Compare the baseline with the model that adds a stride-4 detection head.",
        demoWeights,
    )
