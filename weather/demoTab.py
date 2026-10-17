# Demo tab for Plan B: adverse-weather detection, comparing the plain student with the distilled student (same size and speed).
# Edit demoWeights after the final runs, using local paths or release URLs.
from common.demoSupport import buildDetectionTab

# Weights shown in the model dropdown; names follow <task>-<variant>-seed<k>.pt (plans/05-team-workflow.md)
demoWeights = {
    "Plain student (YOLO26n)": "weights/weather-student-fold0-seed0.pt",
    "Distilled student (WeatherDistill)": "weights/weather-distill-full-fold0-seed0.pt",
    "Teacher (YOLO26m)": "weights/weather-teacher-fold0-seed0.pt",
}


# Build the Gradio tab that demo/app.py mounts
def buildTab():
    return buildDetectionTab(
        "Adverse weather (WeatherDistill)",
        "Vehicles and people in fog, rain, snow and sand. The distilled student has the same size and speed as the plain student.",
        demoWeights,
    )
