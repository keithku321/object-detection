# Custom layers shared by Plan A (underwater) and Plan C (small objects): Coordinate Attention, ECA, SimAM, DySample, weighted fusion.
# Steward: plan C owner (plans/05-team-workflow.md). Register them with common/registerModules.py before building a model from a YAML.
#
# Why some layers build themselves on the first forward pass:
# Ultralytics parse_model passes an unknown layer only its YAML arguments and does not tell it the input channel count
# (and never applies the width multiplier to those arguments). Layers that need the channel count therefore create their
# convolutions on the first forward pass. Ultralytics already runs one dummy forward pass while it builds a model (to
# compute strides), so every parameter exists before optimisers, EMA or weight loading touch the model.
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# Bias that makes a sigmoid gate start at almost 1, so a newly inserted attention layer begins close to the identity
# and does not disturb pretrained features (sigmoid(4) is about 0.982)
identityGateBias = 4.0


# Base class for layers that need the input channel count: subclasses implement buildLayers, which runs once on the first forward
class ChannelAwareModule(nn.Module):
    # Start without layers; channels stays None until the first forward pass
    def __init__(self):
        super().__init__()
        self.channels = None

    # Create the layers for the given input channel count; subclasses must override this
    def buildLayers(self, channels):
        raise NotImplementedError

    # Build on first use, then run the layer's computation
    def forward(self, x):
        if self.channels is None:
            self.channels = x.shape[1]
            self.buildLayers(self.channels)
            self.to(device=x.device, dtype=x.dtype)
        return self.applyLayers(x)

    # The actual computation once layers exist; subclasses must override this
    def applyLayers(self, x):
        raise NotImplementedError


# ECA (Efficient Channel Attention, Wang et al. CVPR 2020): global average pool, a 1D convolution across channels, sigmoid gate
# kernelSize 0 picks the size from the channel count with the paper's formula; otherwise the given odd size is used
class ECA(ChannelAwareModule):
    # Store the settings; the convolution is created on the first forward pass
    def __init__(self, kernelSize=0, gamma=2, offset=1):
        super().__init__()
        self.kernelSize, self.gamma, self.offset = int(kernelSize), gamma, offset

    # Kernel size from the channel count: nearest odd number to (log2(C) + offset) / gamma
    @staticmethod
    def adaptiveKernelSize(channels, gamma=2, offset=1):
        size = int(abs((math.log2(channels) + offset) / gamma))
        return size if size % 2 == 1 else size + 1

    # Create the 1D convolution; its bias starts at identityGateBias and its weights near zero so the gate starts near 1
    def buildLayers(self, channels):
        size = self.kernelSize or self.adaptiveKernelSize(channels, self.gamma, self.offset)
        self.conv = nn.Conv1d(1, 1, kernel_size=size, padding=size // 2, bias=True)
        nn.init.normal_(self.conv.weight, std=0.01)
        nn.init.constant_(self.conv.bias, identityGateBias)

    # Scale every channel by its learned gate
    def applyLayers(self, x):
        pooled = x.mean(dim=(2, 3)).unsqueeze(1)  # batch, 1, channels
        gate = torch.sigmoid(self.conv(pooled)).transpose(1, 2).unsqueeze(-1)  # batch, channels, 1, 1
        return x * gate


# SimAM (Yang et al. ICML 2021): parameter-free attention from an energy function; adds no parameters, which suits a "lightweight" claim
class SimAM(nn.Module):
    # lambdaValue is the small regulariser of the energy function
    def __init__(self, lambdaValue=1e-4):
        super().__init__()
        self.lambdaValue = lambdaValue

    # Weight every activation by sigmoid of its inverse energy
    def forward(self, x):
        pixelCount = x.shape[2] * x.shape[3] - 1
        squaredDifference = (x - x.mean(dim=(2, 3), keepdim=True)).pow(2)
        variance = squaredDifference.sum(dim=(2, 3), keepdim=True) / pixelCount
        inverseEnergy = squaredDifference / (4 * (variance + self.lambdaValue)) + 0.5
        return x * torch.sigmoid(inverseEnergy)


# Coordinate Attention (Hou et al. CVPR 2021): encodes position along height and width into two direction-aware channel gates
class CoordinateAttention(ChannelAwareModule):
    # reduction sets the width of the shared bottleneck (channels divided by reduction, at least 8)
    def __init__(self, reduction=32):
        super().__init__()
        self.reduction = int(reduction)

    # Create the shared 1x1 convolution and the two direction-specific 1x1 convolutions; gate biases start at identityGateBias
    def buildLayers(self, channels):
        bottleneck = max(8, channels // self.reduction)
        self.shared = nn.Sequential(nn.Conv2d(channels, bottleneck, 1), nn.BatchNorm2d(bottleneck), nn.Hardswish())
        self.heightGate = nn.Conv2d(bottleneck, channels, 1)
        self.widthGate = nn.Conv2d(bottleneck, channels, 1)
        for gate in (self.heightGate, self.widthGate):
            nn.init.normal_(gate.weight, std=0.01)
            nn.init.constant_(gate.bias, identityGateBias)

    # Pool along each axis, mix the two descriptors, and gate the input with the height and width attention maps
    def applyLayers(self, x):
        height, width = x.shape[2], x.shape[3]
        pooledHeight = x.mean(dim=3, keepdim=True)  # batch, channels, height, 1
        pooledWidth = x.mean(dim=2, keepdim=True).transpose(2, 3)  # batch, channels, width, 1
        mixed = self.shared(torch.cat([pooledHeight, pooledWidth], dim=2))
        mixedHeight, mixedWidth = torch.split(mixed, [height, width], dim=2)
        gateHeight = torch.sigmoid(self.heightGate(mixedHeight))  # batch, channels, height, 1
        gateWidth = torch.sigmoid(self.widthGate(mixedWidth.transpose(2, 3)))  # batch, channels, 1, width
        return x * gateHeight * gateWidth


# DySample (Liu et al. ICCV 2023, "lp" style): content-aware upsampling that predicts sampling offsets and resamples with grid_sample
# It replaces nn.Upsample in the neck. With zero offsets it equals bilinear upsampling, so it starts as a sane upsampler.
class DySample(ChannelAwareModule):
    # scale is the upsampling factor; groups splits the channels so each group gets its own offsets
    def __init__(self, scale=2, groups=4):
        super().__init__()
        self.scale, self.groups = int(scale), int(groups)

    # Create the offset convolution and the fixed initial sampling positions; channels must be divisible by groups
    def buildLayers(self, channels):
        if channels % self.groups != 0:
            raise ValueError(f"DySample needs the channel count ({channels}) to be divisible by groups ({self.groups})")
        self.offset = nn.Conv2d(channels, 2 * self.groups * self.scale ** 2, 1)
        nn.init.normal_(self.offset.weight, std=0.001)
        nn.init.constant_(self.offset.bias, 0.0)
        self.register_buffer("initialPosition", self.makeInitialPosition(), persistent=False)

    # Offsets (in units of input pixels) that place the sub-pixel samples of each output pixel at its bilinear position
    def makeInitialPosition(self):
        axis = torch.arange((-self.scale + 1) / 2, (self.scale - 1) / 2 + 1) / self.scale
        grid = torch.stack(torch.meshgrid([axis, axis], indexing="ij")).transpose(1, 2)
        return grid.repeat(1, self.groups, 1).reshape(1, -1, 1, 1)

    # Turn offsets into sampling coordinates and resample the input at the higher resolution
    # reshape is used instead of view throughout, because Ultralytics validates in channels_last memory format, where view fails
    def sample(self, x, offset):
        batch, _, height, width = offset.shape
        offset = offset.reshape(batch, 2, -1, height, width)
        coordinatesHeight = torch.arange(height, device=x.device, dtype=x.dtype) + 0.5
        coordinatesWidth = torch.arange(width, device=x.device, dtype=x.dtype) + 0.5
        coordinates = torch.stack(torch.meshgrid([coordinatesWidth, coordinatesHeight], indexing="ij")).transpose(1, 2).unsqueeze(1).unsqueeze(0)
        normaliser = torch.tensor([width, height], dtype=x.dtype, device=x.device).reshape(1, 2, 1, 1, 1)
        coordinates = 2 * (coordinates + offset) / normaliser - 1
        coordinates = F.pixel_shuffle(coordinates.reshape(batch, -1, height, width), self.scale)
        coordinates = coordinates.reshape(batch, 2, -1, self.scale * height, self.scale * width).permute(0, 2, 3, 4, 1).contiguous().flatten(0, 1)
        sampled = F.grid_sample(x.reshape(batch * self.groups, -1, height, width), coordinates.to(x.dtype), mode="bilinear", align_corners=False, padding_mode="border")
        return sampled.reshape(batch, -1, self.scale * height, self.scale * width)

    # Predict small offsets from the content and upsample with them
    def applyLayers(self, x):
        return self.sample(x, self.offset(x) * 0.25 + self.initialPosition.to(x.dtype))


# Learnable weighted concatenation (BiFPN-style fast normalised fusion applied to Concat): each input gets a non-negative weight,
# the weights are normalised to sum to the number of inputs, and the weighted inputs are concatenated.
# It cannot be written as a YAML row (parse_model cannot compute the channels of an unknown multi-input layer); put it into a model
# with common/registerModules.py replaceConcatWithWeighted, which swaps it in for a chosen Concat layer after the model is built.
class WeightedConcat(nn.Module):
    # dimension is the concatenation axis; inputCount is the number of inputs of the replaced Concat layer
    def __init__(self, inputCount=2, dimension=1, epsilon=1e-4):
        super().__init__()
        self.d = dimension
        self.epsilon = epsilon
        self.fusionWeights = nn.Parameter(torch.ones(inputCount))

    # Normalise the weights and concatenate the scaled inputs
    def forward(self, inputs):
        positiveWeights = F.relu(self.fusionWeights)
        normalised = positiveWeights / (positiveWeights.sum() + self.epsilon) * len(inputs)
        return torch.cat([normalised[index].to(tensor.dtype) * tensor for index, tensor in enumerate(inputs)], self.d)
