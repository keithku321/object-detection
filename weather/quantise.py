# Static post-training INT8 quantisation for the compression extension (experiment W8, Phase 6, optional).
# The proposal says "dynamic quantization", but PyTorch dynamic quantisation only covers Linear and recurrent layers, so it barely
# changes a convolutional detector. Static INT8 with calibration images is used instead.
#   python -m weather.quantise --weights last.pt --data /content/datasets/dawn/dawn-fold0.yaml
#   python -m weather.quantise --weights last.pt --calibration /content/datasets/dawn/folds/fold0_train.txt --output student_int8.onnx
# Two ways are provided:
#   exportInt8WithUltralytics  one call to Ultralytics 8.4.175 (export with quantize=8); the simplest route and the default
#   quantiseStaticInt8         ONNX Runtime directly, with an explicit list of calibration images
# Calibration images must come from the TRAINING images only: Ultralytics calibrates on the val split unless split="train" is passed,
# and in 5-fold cross-validation the val split is the held-out fold.
# Only convolutions are quantised. Quantising every operator was tried on the 8-image coco8 set with yolo26n and gave mAP 0, because
# the box coordinates (0 to 640) and the class scores (0 to 1) leave the network in one tensor and cannot share one 8-bit range.
# With convolutions only, both routes kept mAP50-95 within about 0.02 of the FP32 ONNX model on that set (4 validation images, so
# this only shows that the routes work) at about a third of the file size. Measure accuracy, size and latency on your own model.
# Both routes export Ultralytics' default head (one-to-many with external NMS); compare the INT8 model with an FP32 ONNX model
# evaluated the same way (YOLO("model.onnx").val(data=...)), not with the NMS-free rows of the results file.
# Library names keep their own spelling: onnxruntime.quantization, quantize_static, quantize=8.
import argparse
from pathlib import Path

import numpy as np

from common.datasets import readImageBgr
from common.training import protocolImageSize

# Number of calibration images; a few hundred is the usual range
defaultCalibrationCount = 200
# Operator types that are quantised; everything else (the box decoding and the final concatenation) stays in floating point
quantisedOperatorTypes = ["Conv"]


# Export a trained detector to a floating-point ONNX file and return its path
def exportOnnx(weightsPath, imageSize=protocolImageSize):
    from ultralytics import YOLO
    return Path(YOLO(str(weightsPath)).export(format="onnx", imgsz=imageSize, simplify=True))


# Export a trained detector straight to an INT8 ONNX file with Ultralytics, calibrated on the training split of a dataset YAML
def exportInt8WithUltralytics(weightsPath, dataYaml, imageSize=protocolImageSize):
    from ultralytics import YOLO
    return Path(YOLO(str(weightsPath)).export(format="onnx", imgsz=imageSize, quantize=8, data=str(dataYaml), split="train"))


# Resize an image to a padded square exactly as Ultralytics does at inference, and return a float32 array (1, 3, size, size) in 0..1
def preprocessForOnnx(imagePath, imageSize):
    from ultralytics.data.augment import LetterBox
    letterboxed = LetterBox(new_shape=(imageSize, imageSize), auto=False)(image=readImageBgr(imagePath))
    return np.ascontiguousarray(letterboxed[:, :, ::-1].transpose(2, 0, 1), dtype=np.float32)[None] / 255.0


# Quantise the convolutions of an ONNX model to INT8 with the given calibration images and return the output path
def quantiseStaticInt8(onnxPath, calibrationImagePaths, outputPath, imageSize=protocolImageSize):
    import onnxruntime
    from onnxruntime.quantization import CalibrationDataReader, QuantFormat, QuantType, quantize_static
    inputName = onnxruntime.InferenceSession(str(onnxPath), providers=["CPUExecutionProvider"]).get_inputs()[0].name

    # Feeds the calibration images to ONNX Runtime one at a time
    class ImageCalibrationReader(CalibrationDataReader):
        # Keep an iterator over the calibration images
        def __init__(self):
            self.imageIterator = iter(calibrationImagePaths)

        # Return the next model input, or None when the images are used up
        def get_next(self):
            imagePath = next(self.imageIterator, None)
            return None if imagePath is None else {inputName: preprocessForOnnx(imagePath, imageSize)}

    quantize_static(str(onnxPath), str(outputPath), ImageCalibrationReader(), quant_format=QuantFormat.QDQ, per_channel=True,
                    activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8, op_types_to_quantize=quantisedOperatorTypes)
    return Path(outputPath)


# Size of a file in megabytes, for the accuracy-size-speed table
def fileSizeMb(path):
    return round(Path(path).stat().st_size / 1e6, 2)


# Command line entry point: pass --data for the Ultralytics route, or --calibration and --output for the ONNX Runtime route
def main():
    parser = argparse.ArgumentParser(description="Static INT8 quantisation of a trained detector")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--data", default=None, help="dataset YAML; its training split is used for calibration (Ultralytics route)")
    parser.add_argument("--calibration", default=None, help="text file with one training image path per line (ONNX Runtime route)")
    parser.add_argument("--output", default=None, help="output .onnx path for the ONNX Runtime route")
    parser.add_argument("--imgsz", type=int, default=protocolImageSize)
    parser.add_argument("--count", type=int, default=defaultCalibrationCount)
    arguments = parser.parse_args()
    if arguments.data:
        outputPath = exportInt8WithUltralytics(arguments.weights, arguments.data, arguments.imgsz)
        print(f"INT8 ONNX: {outputPath} ({fileSizeMb(outputPath)} MB)")
        return
    if not arguments.calibration or not arguments.output: parser.error("pass --data, or both --calibration and --output")
    calibrationImages = [line for line in Path(arguments.calibration).read_text(encoding="utf-8").splitlines() if line.strip()][:arguments.count]
    onnxPath = exportOnnx(arguments.weights, arguments.imgsz)
    outputPath = quantiseStaticInt8(onnxPath, calibrationImages, arguments.output, arguments.imgsz)
    print(f"FP32 ONNX: {fileSizeMb(onnxPath)} MB, INT8 ONNX: {fileSizeMb(outputPath)} MB")


if __name__ == "__main__":
    main()
