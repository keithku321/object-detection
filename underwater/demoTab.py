# Demo tab for Plan A: image, video and webcam detection of underwater objects, with the baseline and the attention model to compare.
# Edit demoWeights after the final runs: point "Attention model" at the best variant, using a local path or a release URL.
from common.demoSupport import buildDetectionTab

# Weights shown in the model dropdown; names follow <task>-<variant>-seed<k>.pt (plans/05-team-workflow.md)
demoWeights = {
    "Baseline (YOLO26n)": "weights/underwater-baseline-seed0.pt",
    "Attention model": "weights/underwater-eca-a-seed0.pt",
}


# Build the Gradio tab that demo/app.py mounts
def buildTab():
    return buildDetectionTab(
        "Underwater detection",
        "Attention-enhanced lightweight YOLO26 for crabs, fish, jellyfish, shrimp and starfish. Pick the baseline or the attention model to compare.",
        demoWeights,
    )
