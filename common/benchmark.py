# Speed protocol (plans/00-main-plan.md section 6): batch 1, FP16, 640 px, 50 warm-up and 300 timed iterations, GPU name printed.
# All speed numbers in the report must come from this file. Steward: plan D owner (plans/05-team-workflow.md).
#   python -m common.benchmark --weights best.pt --mode end2end
#   python -m common.benchmark --weights best.pt --mode nms --image sample.jpg
# YOLO26 is NMS-free by default, so "end2end" times the one-to-one head with no NMS and "nms" times the one-to-many head plus NMS.
import argparse
import copy
import statistics
import time
from pathlib import Path

# Image side length in pixels used by the protocol
protocolImageSize = 640
# Iterations that are run and thrown away before timing starts
protocolWarmupIterations = 50
# Iterations that are timed
protocolTimedIterations = 300
# Confidence threshold used for the NMS-inclusive path; low on purpose, like validation, so NMS sees realistic candidate counts
nmsConfidenceThreshold = 0.001
# IoU threshold for the NMS-inclusive path
nmsIouThreshold = 0.7
# Detections kept per image after NMS
nmsMaximumDetections = 300


# Name of the GPU, or "cpu"
def getDeviceName(device):
    import torch
    return torch.cuda.get_device_name(device) if str(device).startswith("cuda") else "cpu"


# Load a model from a weights path or accept an already loaded Ultralytics YOLO object; returns a private copy of the network
def loadNetworkCopy(modelSource):
    from ultralytics import YOLO
    yoloModel = modelSource if isinstance(modelSource, YOLO) else YOLO(str(modelSource))
    return copy.deepcopy(yoloModel.model)


# Fuse Conv and BatchNorm, keep only the head branch that the chosen mode uses, and move the network to the device
# mode "end2end" keeps the one-to-one head (NMS-free); mode "nms" keeps the one-to-many head
def prepareNetwork(network, device, halfPrecision, mode):
    network.eval()
    network.end2end = mode == "end2end"
    network.fuse(verbose=False)
    network = network.to(device)
    return network.half() if halfPrecision else network.float()


# Build the model input: a real image resized to a square when a path is given, otherwise random noise
def buildInputTensor(imageSize, device, halfPrecision, imagePath=None):
    import torch
    if imagePath:
        import cv2
        image = cv2.cvtColor(cv2.imread(str(imagePath)), cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (imageSize, imageSize))
        tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    else:
        tensor = torch.rand(1, 3, imageSize, imageSize)
    tensor = tensor.to(device)
    return tensor.half() if halfPrecision else tensor


# Run the network once and, in nms mode, the NMS step too, so both modes include everything after the input tensor
def runOnce(network, inputTensor, mode):
    output = network(inputTensor)
    if mode != "nms": return output
    from ultralytics.utils.nms import non_max_suppression
    predictions = output[0] if isinstance(output, (tuple, list)) else output
    return non_max_suppression(predictions, nmsConfidenceThreshold, nmsIouThreshold, max_det=nmsMaximumDetections)


# Number of parameters and GFLOPs of the fused inference network (what the README of a model family reports)
def countParametersAndGflops(modelSource, imageSize=protocolImageSize):
    from ultralytics.utils.torch_utils import get_flops, get_num_params
    network = prepareNetwork(loadNetworkCopy(modelSource), "cpu", False, "end2end")
    return {"parameters": get_num_params(network), "gflops": round(get_flops(network, imageSize), 3)}


# Time the network following the protocol and return latency statistics in milliseconds plus frames per second
def measureLatency(modelSource, mode="end2end", device=None, halfPrecision=True, imageSize=protocolImageSize,
                   warmupIterations=protocolWarmupIterations, timedIterations=protocolTimedIterations, imagePath=None):
    import torch
    if mode not in ("end2end", "nms"): raise ValueError("mode must be 'end2end' or 'nms'")
    device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    onGpu = str(device).startswith("cuda")
    if halfPrecision and not onGpu:
        print("FP16 needs a GPU; measuring in FP32 on the CPU instead. These numbers do not follow the protocol.")
        halfPrecision = False
    network = prepareNetwork(loadNetworkCopy(modelSource), device, halfPrecision, mode)
    inputTensor = buildInputTensor(imageSize, device, halfPrecision, imagePath)
    gpuName = getDeviceName(device)
    print(f"Benchmark on {gpuName}: mode={mode}, {'FP16' if halfPrecision else 'FP32'}, {imageSize}px, batch 1, {warmupIterations} warm-up + {timedIterations} timed")
    timingsMs = []
    with torch.inference_mode():
        for _ in range(warmupIterations): runOnce(network, inputTensor, mode)
        for _ in range(timedIterations):
            if onGpu: torch.cuda.synchronize()
            startTime = time.perf_counter()
            runOnce(network, inputTensor, mode)
            if onGpu: torch.cuda.synchronize()
            timingsMs.append((time.perf_counter() - startTime) * 1000.0)
    meanMs = statistics.mean(timingsMs)
    sortedTimings = sorted(timingsMs)
    return {
        "mode": mode, "precision": "fp16" if halfPrecision else "fp32", "imageSize": imageSize, "gpu": gpuName,
        "latencyMeanMs": round(meanMs, 3), "latencyMedianMs": round(statistics.median(timingsMs), 3),
        "latencyP95Ms": round(sortedTimings[int(0.95 * len(sortedTimings)) - 1], 3), "latencyStdMs": round(statistics.pstdev(timingsMs), 3),
        "fps": round(1000.0 / meanMs, 1),
    }


# Time an ONNX model with ONNX Runtime (CPU latency for the compression experiments, or GPU with onnxruntime-gpu)
def measureOnnxLatency(onnxPath, imageSize=protocolImageSize, providers=("CPUExecutionProvider",),
                       warmupIterations=protocolWarmupIterations, timedIterations=protocolTimedIterations):
    import numpy as np
    import onnxruntime
    session = onnxruntime.InferenceSession(str(onnxPath), providers=list(providers))
    inputName = session.get_inputs()[0].name
    inputType = np.float16 if "float16" in session.get_inputs()[0].type else np.float32
    inputArray = np.random.rand(1, 3, imageSize, imageSize).astype(inputType)
    for _ in range(warmupIterations): session.run(None, {inputName: inputArray})
    timingsMs = []
    for _ in range(timedIterations):
        startTime = time.perf_counter()
        session.run(None, {inputName: inputArray})
        timingsMs.append((time.perf_counter() - startTime) * 1000.0)
    meanMs = statistics.mean(timingsMs)
    return {"provider": session.get_providers()[0], "latencyMeanMs": round(meanMs, 3), "latencyMedianMs": round(statistics.median(timingsMs), 3), "fps": round(1000.0 / meanMs, 1)}


# Time any exported format (.pt, .onnx, .engine) through the Ultralytics predictor and return its own per-stage speed in milliseconds
# quantize=16 asks for FP16 (the old half=True argument is deprecated in 8.4.175); nms=False selects the NMS-free head of a .pt model
def measurePredictSpeed(weightsPath, imagePath, imageSize=protocolImageSize, halfPrecision=True, iterations=100):
    from ultralytics import YOLO
    model = YOLO(str(weightsPath))
    predictArguments = dict(imgsz=imageSize, quantize=16 if halfPrecision else None, nms=False, verbose=False)
    for _ in range(10): model.predict(str(imagePath), **predictArguments)
    stageTimes = {"preprocess": [], "inference": [], "postprocess": []}
    for _ in range(iterations):
        speed = model.predict(str(imagePath), **predictArguments)[0].speed
        for stageName in stageTimes: stageTimes[stageName].append(speed[stageName])
    return {f"{stageName}Ms": round(statistics.mean(values), 3) for stageName, values in stageTimes.items()}


# Measure speed and size together and return one flat dict ready to be used as metrics of a results row
def benchmarkModel(modelSource, mode="end2end", **measureArguments):
    metrics = measureLatency(modelSource, mode=mode, **measureArguments)
    metrics.update(countParametersAndGflops(modelSource, metrics["imageSize"]))
    return metrics


# Command line entry point
def main():
    parser = argparse.ArgumentParser(description="Measure latency with the shared speed protocol")
    parser.add_argument("--weights", required=True, help="path to a .pt weights file")
    parser.add_argument("--mode", choices=["end2end", "nms"], default="end2end")
    parser.add_argument("--image", default=None, help="optional real image, recommended for --mode nms")
    parser.add_argument("--imgsz", type=int, default=protocolImageSize)
    parser.add_argument("--fp32", action="store_true", help="measure in FP32 (not the protocol)")
    arguments = parser.parse_args()
    metrics = benchmarkModel(Path(arguments.weights), mode=arguments.mode, imagePath=arguments.image, imageSize=arguments.imgsz, halfPrecision=not arguments.fp32)
    for metricName, metricValue in metrics.items(): print(f"{metricName}: {metricValue}")


if __name__ == "__main__":
    main()
