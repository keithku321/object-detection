# Training for CamoEdge: one experiment (E1 to E5) and one seed per call, a checkpoint after every epoch, and resume.
#   python -m camoedge.train --experiment E1 --dataset /content/datasets/camo --project /content/drive/MyDrive/yoloProjectResults --epochs 30
#   python -m camoedge.train --experiment E5 --subset 100 --epochs 2 ...        (quick local trial on 100 images)
# The standard protocol has no validation split, so the reported model is the one after the final epoch (last.pt). Choosing the best
# epoch on the test sets would be model selection on test data.
# Build in the order of the experiment matrix: E1 gives a working model within days, and every later step is a small change.
import argparse
import random
import time
from pathlib import Path

import numpy as np
import torch

from camoedge.dataset import buildTrainingLoader, defaultImageSize
from camoedge.losses import camoEdgeLoss, experimentLossSettings
from camoedge.model import CamoEdgeNet, countParameters, experimentModelSettings
from common.environment import getDefaultDevice, getDefaultWorkerCount

# Name of this task in the results log and in run folders
taskName = "camoedge"
# Starting learning rate of Adam
defaultLearningRate = 1e-4
# Power of the polynomial learning-rate decay
decayPower = 0.9
# Largest gradient norm; larger gradients are scaled down to keep mixed-precision training stable
gradientClipNorm = 1.0


# Seed Python, NumPy and PyTorch so a run can be repeated
def setSeed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# Name of a run folder, for example "E5-seed0" or "E5-concat-seed0"
def makeVariantName(experimentName, gatingMode="multiply", encoderName="pvt_v2_b2", edgeLossWeight=None):
    parts = [experimentName]
    if gatingMode != "multiply": parts.append(gatingMode)
    if encoderName != "pvt_v2_b2": parts.append(encoderName)
    if edgeLossWeight is not None: parts.append(f"edge{edgeLossWeight:g}")
    return "-".join(parts)


# Save a checkpoint through a temporary file, so a session that ends during the write cannot leave a broken last.pt on Drive
def saveCheckpoint(checkpointPath, state):
    temporaryPath = Path(checkpointPath).with_suffix(".tmp")
    torch.save(state, temporaryPath)
    temporaryPath.replace(checkpointPath)


# Train one experiment. Returns the path of last.pt. Restarting the same call continues from the last finished epoch.
# maximumCount limits the training images (use about 100 for a local trial); edgeLossWeight and gatingMode override the experiment
# defaults for the sweeps of experiment E6; encoderName "resnet50" gives the encoder comparison of E7.
def trainCamoEdge(experimentName, datasetRoot, project, seed=0, epochs=30, batchSize=16, imageSize=defaultImageSize, learningRate=defaultLearningRate,
                  encoderName="pvt_v2_b2", gatingMode="multiply", edgeLossWeight=None, device=None, workers=None, maximumCount=None, pretrained=True):
    device = device or getDefaultDevice()
    useAmp = str(device).startswith("cuda")
    modelSettings = {**experimentModelSettings[experimentName], "gatingMode": gatingMode}
    lossSettings = dict(experimentLossSettings[experimentName])
    if edgeLossWeight is not None: lossSettings["edgeLossWeight"] = edgeLossWeight
    variantName = makeVariantName(experimentName, gatingMode, encoderName, edgeLossWeight)
    runFolder = Path(project) / taskName / f"{variantName}-seed{seed}"
    runFolder.mkdir(parents=True, exist_ok=True)
    checkpointPath = runFolder / "last.pt"
    setSeed(seed)
    model = CamoEdgeNet(encoderName=encoderName, pretrained=pretrained, **modelSettings).to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=learningRate)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimiser, lambda epoch: (1.0 - epoch / epochs) ** decayPower)
    scaler = torch.amp.GradScaler("cuda", enabled=useAmp)
    startEpoch, history = 0, []
    if checkpointPath.exists():
        checkpoint = torch.load(checkpointPath, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimiser.load_state_dict(checkpoint["optimiser"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint["scaler"])
        startEpoch, history = checkpoint["epoch"], checkpoint["history"]
        if startEpoch < epochs: print(f"Resuming {runFolder.name} from epoch {startEpoch + 1} of {epochs}")
    settings = {"experimentName": experimentName, "variantName": variantName, "encoderName": encoderName, "modelSettings": modelSettings,
                "lossSettings": lossSettings, "imageSize": imageSize, "seed": seed, "epochs": epochs, "batchSize": batchSize, "learningRate": learningRate}
    if startEpoch >= epochs:
        print(f"{runFolder.name} already finished")
        return checkpointPath
    loader = buildTrainingLoader(datasetRoot, imageSize, batchSize, getDefaultWorkerCount() if workers is None else workers, maximumCount)
    print(f"{variantName}: {countParameters(model):,} parameters, {len(loader.dataset)} training images, device {device}")
    for epoch in range(startEpoch, epochs):
        model.train()
        epochStart, runningParts = time.time(), {}
        for batch in loader:
            image, mask, edge = batch["image"].to(device), batch["mask"].to(device), batch["edge"].to(device)
            with torch.autocast(device_type="cuda" if useAmp else "cpu", enabled=useAmp):
                outputs = model(image)
            loss, parts = camoEdgeLoss(outputs, mask, edge, **lossSettings)
            optimiser.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimiser)
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradientClipNorm)
            scaler.step(optimiser)
            scaler.update()
            for partName, partValue in parts.items(): runningParts[partName] = runningParts.get(partName, 0.0) + partValue
        scheduler.step()
        epochRecord = {partName: partValue / len(loader) for partName, partValue in runningParts.items()}
        epochRecord.update({"epoch": epoch + 1, "seconds": round(time.time() - epochStart, 1)})
        history.append(epochRecord)
        print(f"epoch {epoch + 1}/{epochs}  loss {epochRecord['loss']:.4f}  mask {epochRecord['maskLoss']:.4f}  edge {epochRecord['edgeLoss']:.4f}  {epochRecord['seconds']}s")
        saveCheckpoint(checkpointPath, {"model": model.state_dict(), "optimiser": optimiser.state_dict(), "scheduler": scheduler.state_dict(),
                                        "scaler": scaler.state_dict(), "epoch": epoch + 1, "history": history, "settings": settings})
    return checkpointPath


# Save a slim copy of a checkpoint (weights and settings only) as <weightsFolder>/camoedge-<variant>-seed<k>.pt for the demo or a release
def exportFinalWeights(checkpointPath, weightsFolder):
    checkpoint = torch.load(checkpointPath, map_location="cpu", weights_only=False)
    settings = checkpoint["settings"]
    targetPath = Path(weightsFolder) / f"{taskName}-{settings['variantName']}-seed{settings['seed']}.pt"
    targetPath.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": checkpoint["model"], "settings": settings}, targetPath)
    return targetPath


# Command line entry point: train, then evaluate on the test sets and append results rows
def main():
    parser = argparse.ArgumentParser(description="Train one CamoEdge experiment")
    parser.add_argument("--experiment", default="E1", choices=sorted(experimentModelSettings))
    parser.add_argument("--dataset", required=True, help="dataset root that contains TrainDataset/ and TestDataset/")
    parser.add_argument("--project", required=True, help="folder for runs; use a folder on Google Drive in Colab")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=defaultImageSize)
    parser.add_argument("--lr", type=float, default=defaultLearningRate)
    parser.add_argument("--encoder", default="pvt_v2_b2")
    parser.add_argument("--gating", default="multiply", choices=["multiply", "concat"])
    parser.add_argument("--edge-weight", type=float, default=None)
    parser.add_argument("--subset", type=int, default=None, help="train on only this many images (local trial)")
    parser.add_argument("--device", default=None)
    parser.add_argument("--skip-evaluation", action="store_true")
    arguments = parser.parse_args()
    checkpointPath = trainCamoEdge(arguments.experiment, arguments.dataset, arguments.project, arguments.seed, arguments.epochs, arguments.batch, arguments.imgsz,
                                   arguments.lr, arguments.encoder, arguments.gating, arguments.edge_weight, arguments.device, maximumCount=arguments.subset)
    if not arguments.skip_evaluation:
        from camoedge.evaluate import evaluateAndLog
        evaluateAndLog(checkpointPath, arguments.dataset, arguments.device, notes=f"subset={arguments.subset}" if arguments.subset else "")


if __name__ == "__main__":
    main()
