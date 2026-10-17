# Demo tab for Plan D: upload an image and see the predicted camouflage mask, the edge map and an overlay, with a threshold slider.
# Edit demoWeights after the final runs; the files come from camoedge/train.py exportFinalWeights (local paths or release URLs).
import functools

import numpy as np
import torch
import torch.nn.functional as F

from camoedge.dataset import preprocessImage
from common.demoSupport import resolveWeights
from common.environment import getDefaultDevice

# Weights shown in the model dropdown; names follow <task>-<variant>-seed<k>.pt (plans/05-team-workflow.md)
demoWeights = {
    "Baseline without edge guidance (E2)": "weights/camoedge-E2-seed0.pt",
    "CamoEdge (E5)": "weights/camoedge-E5-seed0.pt",
}
# Colour of the mask in the overlay, as RGB
overlayColour = (255, 64, 64)
# How strongly the overlay colour is mixed into masked pixels
overlayStrength = 0.5


# Load a CamoEdge model once per weights file and reuse it
@functools.lru_cache(maxsize=2)
def loadSegmenter(weightsPath):
    from camoedge.evaluate import loadCheckpointModel
    return loadCheckpointModel(weightsPath, getDefaultDevice())


# Predict the mask and edge probability maps of one RGB image array at its original size; the edge map is None without an edge branch
@torch.inference_mode()
def segmentImage(weightsPath, image):
    model, settings = loadSegmenter(str(weightsPath))
    device = next(model.parameters()).device
    outputs = model(preprocessImage(image, settings["imageSize"]).to(device))
    size = image.shape[:2]
    maskProbability = torch.sigmoid(F.interpolate(outputs["masks"][0].float(), size=size, mode="bilinear", align_corners=False))[0, 0].cpu().numpy()
    edgeProbability = None
    if outputs["edge"] is not None:
        edgeProbability = torch.sigmoid(F.interpolate(outputs["edge"].float(), size=size, mode="bilinear", align_corners=False))[0, 0].cpu().numpy()
    return maskProbability, edgeProbability


# Blend the overlay colour into the pixels of an RGB image where the binary mask is set
def makeOverlay(image, binaryMask):
    overlay = image.astype(np.float32)
    overlay[binaryMask] = (1.0 - overlayStrength) * overlay[binaryMask] + overlayStrength * np.array(overlayColour, dtype=np.float32)
    return overlay.astype(np.uint8)


# Build the Gradio tab that demo/app.py mounts
def buildTab():
    import gradio as gr

    # Handler: run the chosen model and return the mask, the edge map, the overlay and a status line
    def onImage(image, optionLabel, threshold):
        if image is None: return None, None, None, "Upload an image first."
        weightsPath = resolveWeights(demoWeights.get(optionLabel))
        if weightsPath is None: return None, None, None, f"Weights for '{optionLabel}' were not found. Train the model, then export it with camoedge/train.py exportFinalWeights."
        maskProbability, edgeProbability = segmentImage(weightsPath, image)
        binaryMask = maskProbability >= threshold
        edgeImage = (edgeProbability * 255).astype(np.uint8) if edgeProbability is not None else None
        status = f"{binaryMask.mean() * 100:.1f}% of the image is marked as camouflaged object" + ("" if edgeProbability is not None else " (this model has no edge branch)")
        return (binaryMask * 255).astype(np.uint8), edgeImage, makeOverlay(image, binaryMask), status

    optionLabels = list(demoWeights)
    with gr.Blocks() as tab:
        gr.Markdown("## Camouflaged objects (CamoEdge)\nBoundary-guided segmentation: the edge branch finds object boundaries and gates the decoder features.")
        with gr.Row():
            optionDropdown = gr.Dropdown(optionLabels, value=optionLabels[-1], label="Model")
            thresholdSlider = gr.Slider(0.05, 0.95, value=0.5, step=0.05, label="Mask threshold")
        imageInput = gr.Image(type="numpy", label="Input image")
        segmentButton = gr.Button("Segment")
        with gr.Row():
            maskOutput = gr.Image(label="Predicted mask")
            edgeOutput = gr.Image(label="Edge map")
            overlayOutput = gr.Image(label="Overlay")
        statusOutput = gr.Markdown()
        segmentButton.click(onImage, [imageInput, optionDropdown, thresholdSlider], [maskOutput, edgeOutput, overlayOutput, statusOutput])
    return tab
