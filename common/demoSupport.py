# Shared building blocks for the demo tabs: weights lookup, image and video detection, and a ready-made Gradio detection tab.
# Each task's demoTab.py calls buildDetectionTab with its own weights; demo/app.py mounts the tabs (plans/05-team-workflow.md).
import functools
import statistics
import time
from pathlib import Path

# Repository root, so the weights folder is found from any working directory
repositoryRoot = Path(__file__).resolve().parent.parent
# Folder for downloaded and trained weights; *.pt is ignored by git
weightsFolder = repositoryRoot / "weights"
# Default confidence threshold of the demo slider
defaultConfidence = 0.4
# Video frames are processed at this image size
demoImageSize = 640


# Turn a weights setting into a local path: a local path is used as is, a URL is downloaded once into the weights folder
# Returns None when the file does not exist yet, so a tab can show a friendly message instead of crashing
def resolveWeights(weightsSource):
    if not weightsSource: return None
    weightsSource = str(weightsSource)
    if weightsSource.startswith(("http://", "https://")):
        from common.datasets import downloadFile
        targetPath = weightsFolder / weightsSource.rsplit("/", 1)[-1]
        return targetPath if targetPath.exists() else downloadFile(weightsSource, targetPath)
    localPath = Path(weightsSource)
    localPath = localPath if localPath.is_absolute() else repositoryRoot / localPath
    return localPath if localPath.exists() else None


# Load a YOLO model once per weights path and reuse it; registers the custom layers so attention variants can be unpickled
@functools.lru_cache(maxsize=4)
def loadDetector(weightsPath):
    from common.registerModules import registerCustomModules
    from ultralytics import YOLO
    registerCustomModules()
    return YOLO(str(weightsPath))


# Table rows [class name, confidence, x1, y1, x2, y2] for one Ultralytics result
def formatDetectionRows(result):
    rows = []
    for box in result.boxes:
        x1, y1, x2, y2 = (round(value) for value in box.xyxy[0].tolist())
        rows.append([result.names[int(box.cls)], round(float(box.conf), 3), x1, y1, x2, y2])
    return rows


# Detect objects in one RGB image array; returns the annotated RGB image, the detection rows and the model time in milliseconds
def detectImage(weightsPath, image, confidenceThreshold=defaultConfidence, imageSize=demoImageSize):
    from common.training import endToEndArguments
    model = loadDetector(str(weightsPath))
    result = model.predict(image[:, :, ::-1], conf=confidenceThreshold, imgsz=imageSize, verbose=False, **endToEndArguments)[0]
    return result.plot()[:, :, ::-1], formatDetectionRows(result), sum(result.speed.values())


# Detect objects in every frame of a video file, write an annotated mp4 and return its path and the average frames per second
def detectVideo(weightsPath, videoPath, outputPath, confidenceThreshold=defaultConfidence, imageSize=demoImageSize):
    import cv2
    from common.training import endToEndArguments
    model = loadDetector(str(weightsPath))
    capture = cv2.VideoCapture(str(videoPath))
    width, height = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(str(outputPath), cv2.VideoWriter_fourcc(*"mp4v"), capture.get(cv2.CAP_PROP_FPS) or 25.0, (width, height))
    frameTimes = []
    while True:
        frameWasRead, frame = capture.read()
        if not frameWasRead: break
        startTime = time.perf_counter()
        annotatedFrame = model.predict(frame, conf=confidenceThreshold, imgsz=imageSize, verbose=False, **endToEndArguments)[0].plot()
        frameTimes.append(time.perf_counter() - startTime)
        writer.write(annotatedFrame)
    capture.release()
    writer.release()
    return str(outputPath), (1.0 / statistics.mean(frameTimes) if frameTimes else 0.0)


# Build a Gradio tab with image upload, video upload and webcam for a detection task
# weightsOptions maps a label such as "Baseline" or "Proposed" to a weights path or URL; the user picks one in a dropdown
def buildDetectionTab(title, description, weightsOptions):
    import gradio as gr
    import tempfile

    # Look up the chosen weights, or return None with a readable explanation when they are missing
    def pickWeights(optionLabel):
        weightsPath = resolveWeights(weightsOptions.get(optionLabel))
        message = None if weightsPath else f"Weights for '{optionLabel}' were not found. Train the model, then copy the .pt file to the path given in {title}'s demoTab.py."
        return weightsPath, message

    # Handler for the image tab
    def onImage(image, optionLabel, confidenceThreshold):
        if image is None: return None, [], "Upload an image first."
        weightsPath, message = pickWeights(optionLabel)
        if message: return None, [], message
        annotatedImage, rows, milliseconds = detectImage(weightsPath, image, confidenceThreshold)
        return annotatedImage, rows, f"{len(rows)} detections, model time {milliseconds:.1f} ms"

    # Handler for the video tab
    def onVideo(videoPath, optionLabel, confidenceThreshold):
        if videoPath is None: return None, "Upload a video first."
        weightsPath, message = pickWeights(optionLabel)
        if message: return None, message
        outputPath = Path(tempfile.mkdtemp()) / "annotated.mp4"
        outputFile, framesPerSecond = detectVideo(weightsPath, videoPath, outputPath, confidenceThreshold)
        return outputFile, f"Average speed: {framesPerSecond:.1f} FPS (including drawing and video decoding)"

    # Handler for the webcam stream: one frame in, one annotated frame out
    def onWebcamFrame(frame, optionLabel, confidenceThreshold):
        weightsPath, message = pickWeights(optionLabel)
        if frame is None or message: return frame
        return detectImage(weightsPath, frame, confidenceThreshold)[0]

    optionLabels = list(weightsOptions)
    with gr.Blocks() as tab:
        gr.Markdown(f"## {title}\n{description}")
        with gr.Row():
            optionDropdown = gr.Dropdown(optionLabels, value=optionLabels[0], label="Model")
            confidenceSlider = gr.Slider(0.05, 0.95, value=defaultConfidence, step=0.05, label="Confidence threshold")
        with gr.Tab("Image"):
            imageInput = gr.Image(type="numpy", label="Input image")
            imageButton = gr.Button("Detect")
            imageOutput = gr.Image(label="Detections")
            imageTable = gr.Dataframe(headers=["class", "confidence", "x1", "y1", "x2", "y2"], label="Detections")
            imageStatus = gr.Markdown()
            imageButton.click(onImage, [imageInput, optionDropdown, confidenceSlider], [imageOutput, imageTable, imageStatus])
        with gr.Tab("Video"):
            videoInput = gr.Video(label="Input video")
            videoButton = gr.Button("Detect")
            videoOutput = gr.Video(label="Annotated video")
            videoStatus = gr.Markdown()
            videoButton.click(onVideo, [videoInput, optionDropdown, confidenceSlider], [videoOutput, videoStatus])
        with gr.Tab("Webcam"):
            webcamInput = gr.Image(sources=["webcam"], streaming=True, type="numpy", label="Webcam")
            webcamOutput = gr.Image(label="Live detections")
            webcamInput.stream(onWebcamFrame, [webcamInput, optionDropdown, confidenceSlider], webcamOutput)
    return tab
