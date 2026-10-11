# Structured channel pruning for the compression extension (experiment W7, Phase 6, optional). NOT WORKING YET: a starting point only.
# The plan warns that pruning is the most likely place to lose time: prove it first on the plain student with a short fine-tune,
# then add distillation. Uses the torch-pruning library, which follows channel dependencies through concatenations and splits.
#   python -m weather.prune --weights last.pt --output pruned.pt --ratio 0.4
#
# Status on 2026-10-11 (torch-pruning 1.6.1, torch 2.11.0 on CPU, Python 3.14, yolo26n):
#   - pruneDetector does NOT finish. Counting operations works and torch_pruning.DependencyGraph().build_dependency() completes, but
#     constructing MagnitudePruner did not return within several minutes, so no pruned model has been produced by this file.
#   - makeKeepArchitectureTrainer and finetunePruned were checked with an unpruned network only: one epoch trains without the
#     network being rebuilt from its YAML. They have never been run on a pruned network.
# Where to look next: the C3k2 blocks of YOLO26 split their channels with chunk(), which is what torch-pruning's own YOLOv8 example
# works around by replacing those blocks with a split-free version before pruning; the C2PSA attention block and the two detection
# heads are further suspects. Try that replacement, a smaller pruning scope (backbone only), or another torch-pruning version.
#
# A pruned network no longer matches its model YAML, and Ultralytics rebuilds a model from the YAML when training starts. Fine-tuning
# must therefore use makeKeepArchitectureTrainer(), which hands the pruned network to the trainer as it is.
import argparse
import copy
from pathlib import Path

import torch

from common.training import protocolImageSize, trainDetector

# Fraction of channels to remove; the proposal asks for about 40%
defaultPruningRatio = 0.4
# Channel counts are kept at multiples of this number, which keeps the layers friendly to GPU kernels and exporters
channelRounding = 8


# Prune a trained detector and save it as an Ultralytics-style checkpoint that YOLO(path) can load
# The Detect head and the C2PSA attention block are not pruned: the head must keep its output layout and the attention block
# reshapes its channels into heads, which the dependency tracer cannot follow safely. Returns size statistics before and after.
def pruneDetector(weightsPath, outputPath, pruningRatio=defaultPruningRatio, imageSize=protocolImageSize):
    import torch_pruning
    import ultralytics
    from ultralytics import YOLO
    from ultralytics.nn.modules import C2PSA, Detect
    yoloModel = YOLO(str(weightsPath))
    network = copy.deepcopy(yoloModel.model).float()
    for parameter in network.parameters(): parameter.requires_grad_(True)
    network.eval()
    network.model[-1].training = True  # raw head outputs give the tracer a simple graph; BatchNorm stays in eval mode
    exampleInput = torch.randn(1, 3, imageSize, imageSize)
    ignoredLayers = [module for module in network.modules() if isinstance(module, (Detect, C2PSA))]
    operationsBefore, parametersBefore = torch_pruning.utils.count_ops_and_params(network, exampleInput)
    pruner = torch_pruning.pruner.MagnitudePruner(
        network, exampleInput, importance=torch_pruning.importance.GroupMagnitudeImportance(p=2),
        pruning_ratio=pruningRatio, ignored_layers=ignoredLayers, round_to=channelRounding,
    )
    pruner.step()
    operationsAfter, parametersAfter = torch_pruning.utils.count_ops_and_params(network, exampleInput)
    network.model[-1].training = False
    checkpoint = {"model": network.half(), "ema": None, "optimizer": None, "epoch": -1, "train_args": dict(yoloModel.ckpt.get("train_args", {})),
                  "version": ultralytics.__version__, "pruningRatio": pruningRatio}
    Path(outputPath).parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, outputPath)
    statistics = {"parametersBefore": parametersBefore, "parametersAfter": parametersAfter, "gflopsBefore": operationsBefore / 1e9, "gflopsAfter": operationsAfter / 1e9}
    print(f"Pruned {weightsPath}: parameters {parametersBefore:,} -> {parametersAfter:,}, operations {operationsBefore / 1e9:.2f}G -> {operationsAfter / 1e9:.2f}G")
    return statistics


# Trainer class that trains the loaded network as it is instead of rebuilding it from the model YAML (needed after pruning)
def makeKeepArchitectureTrainer():
    from ultralytics.models.yolo.detect import DetectionTrainer

    # Detection trainer that keeps the architecture of the weights it is given
    class KeepArchitectureTrainer(DetectionTrainer):
        # Return a trainable copy of the loaded network; fall back to the normal build when no network was loaded
        def get_model(self, cfg=None, weights=None, verbose=True):
            if weights is None: return super().get_model(cfg=cfg, weights=weights, verbose=verbose)
            model = copy.deepcopy(weights).float()
            for parameter in model.parameters(): parameter.requires_grad_(True)
            return model

    return KeepArchitectureTrainer


# Fine-tune a pruned checkpoint, optionally with distillation from a teacher (pass teacherWeights and a preset name)
def finetunePruned(prunedWeights, dataYaml, seed, epochs, project, runName, teacherWeights=None, presetName="full", overrides=None,
                   imageSize=protocolImageSize, batch=16, device=None, workers=None):
    trainerClass = makeKeepArchitectureTrainer()
    if teacherWeights is None:
        return trainDetector(prunedWeights, dataYaml, seed, epochs, project, runName, imageSize, batch, device, workers, trainerClass)
    from weather.distillTrainer import splitDistillSettings, useWeatherDistillation
    lambdaFeature, settings = splitDistillSettings(presetName, overrides)
    with useWeatherDistillation(settings):
        return trainDetector(prunedWeights, dataYaml, seed, epochs, project, runName, imageSize, batch, device, workers, trainerClass,
                             trainArguments={"distill_model": str(teacherWeights), "dis": lambdaFeature})


# Command line entry point
def main():
    parser = argparse.ArgumentParser(description="Prune a trained detector with torch-pruning")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--ratio", type=float, default=defaultPruningRatio)
    parser.add_argument("--imgsz", type=int, default=protocolImageSize)
    arguments = parser.parse_args()
    pruneDetector(arguments.weights, arguments.output, arguments.ratio, arguments.imgsz)


if __name__ == "__main__":
    main()
