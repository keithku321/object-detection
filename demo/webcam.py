# Run a trained YOLO26 model on the live webcam of your own computer
# Install the requirements first with: pip install ultralytics
# Press q while the video window is focused to quit
import cv2
from ultralytics import YOLO
# Path to the best.pt file that you downloaded from Colab
weightsPath = "best.pt"
# Which camera to use; 0 is normally the built-in webcam
cameraIndex = 0
# Detections below this confidence are hidden
confidenceThreshold = 0.4

# Open the webcam, run detection on every frame and show the annotated result
# The loop ends when the camera stops delivering frames or when q is pressed
def runWebcamDemo(weightsPath, cameraIndex, confidenceThreshold):
    model = YOLO(weightsPath)
    camera = cv2.VideoCapture(cameraIndex)
    while camera.isOpened():
        frameWasRead, frame = camera.read()
        if not frameWasRead: break
        detectionResults = model.predict(frame, conf=confidenceThreshold, verbose=False)
        annotatedFrame = detectionResults[0].plot()
        cv2.imshow("YOLO26 occlusion demo", annotatedFrame)
        if cv2.waitKey(1) & 0xFF == ord("q"): break
    camera.release()
    cv2.destroyAllWindows()

if __name__ == "__main__": runWebcamDemo(weightsPath, cameraIndex, confidenceThreshold)