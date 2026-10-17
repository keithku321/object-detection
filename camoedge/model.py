# CamoEdge network: a pretrained encoder, an Edge Guidance Branch (EGB), Edge-Attention Gating (EAG) and an FPN-style decoder.
# The same class covers every row of the experiment matrix, so each step is a small change (plans/04-camoedge.md):
#   E1, E2  CamoEdgeNet(useEdgeBranch=False)                              encoder + plain decoder
#   E3, E4  CamoEdgeNet(useEdgeBranch=True, useEdgeGating=False)           edge branch as an auxiliary output (E4 adds the edge loss)
#   E5      CamoEdgeNet(useEdgeBranch=True, useEdgeGating=True)            full CamoEdge
#   E6      gatingMode="concat" instead of "multiply"                      gating variant
#   E7      encoderName="resnet50" instead of "pvt_v2_b2"                  encoder comparison
import torch
import torch.nn as nn
import torch.nn.functional as F

# Channel width of the decoder and of the edge features
defaultDecoderChannels = 64
# Encoder names in the timm library; confirm them with timm.list_models("pvt_v2*") if timm is upgraded
encoderNames = ("pvt_v2_b2", "resnet50")
# Named model settings for the experiment matrix
experimentModelSettings = {
    "E1": {"useEdgeBranch": False, "useEdgeGating": False},
    "E2": {"useEdgeBranch": False, "useEdgeGating": False},
    "E3": {"useEdgeBranch": True, "useEdgeGating": False},
    "E4": {"useEdgeBranch": True, "useEdgeGating": False},
    "E5": {"useEdgeBranch": True, "useEdgeGating": True},
}


# 3x3 convolution, batch normalisation and ReLU, the building block of the decoder and the edge branch
def convBlock(inChannels, outChannels, kernelSize=3):
    return nn.Sequential(nn.Conv2d(inChannels, outChannels, kernelSize, padding=kernelSize // 2, bias=False), nn.BatchNorm2d(outChannels), nn.ReLU(inplace=True))


# Edge Guidance Branch: a shallow convolution stream on the high-resolution (stride 4) feature, optionally with the image gradient
# Returns the edge feature map (for gating) and the edge prediction logits, both at stride 4
class EdgeGuidanceBranch(nn.Module):
    # inChannels is the channel count of the stride-4 encoder feature
    def __init__(self, inChannels, edgeChannels=defaultDecoderChannels, useImageGradient=True):
        super().__init__()
        self.useImageGradient = useImageGradient
        sobelX = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]])
        self.register_buffer("sobelKernels", torch.stack([sobelX, sobelX.t()]).unsqueeze(1), persistent=False)
        self.stream = nn.Sequential(convBlock(inChannels + (1 if useImageGradient else 0), edgeChannels), convBlock(edgeChannels, edgeChannels))
        self.predict = nn.Conv2d(edgeChannels, 1, 1)

    # Gradient magnitude of the grey image, resized to the size of the feature map; has no learnable parameters
    def imageGradient(self, image, size):
        grey = image.mean(dim=1, keepdim=True)
        gradients = F.conv2d(grey, self.sobelKernels.to(grey.dtype), padding=1)
        magnitude = torch.sqrt(gradients.pow(2).sum(dim=1, keepdim=True) + 1e-6)
        return F.interpolate(magnitude, size=size, mode="bilinear", align_corners=False)

    # Compute the edge feature and the edge logits
    def forward(self, feature, image):
        if self.useImageGradient: feature = torch.cat([feature, self.imageGradient(image, feature.shape[-2:])], dim=1)
        edgeFeature = self.stream(feature)
        return edgeFeature, self.predict(edgeFeature)


# Edge-Attention Gating: inject the edge feature into a decoder feature
# "multiply": f' = f * (1 + sigmoid(conv(edge))), so the gate can only emphasise features, never erase them
# "concat":   f' = conv(concat(f, edge)), the ablation alternative
class EdgeAttentionGate(nn.Module):
    # edgeChannels and featureChannels are the channel counts of the edge feature and of the decoder feature
    def __init__(self, edgeChannels, featureChannels, mode="multiply"):
        super().__init__()
        if mode not in ("multiply", "concat"): raise ValueError("mode must be 'multiply' or 'concat'")
        self.mode = mode
        self.gate = nn.Conv2d(edgeChannels, featureChannels, 3, padding=1) if mode == "multiply" else convBlock(edgeChannels + featureChannels, featureChannels)

    # Resize the edge feature to the decoder feature and apply the gate
    def forward(self, feature, edgeFeature):
        edgeFeature = F.interpolate(edgeFeature, size=feature.shape[-2:], mode="bilinear", align_corners=False)
        if self.mode == "multiply": return feature * (1.0 + torch.sigmoid(self.gate(edgeFeature)))
        return self.gate(torch.cat([feature, edgeFeature], dim=1))


# The full network. forward returns {"masks": [final logits, side logits ...], "edge": edge logits or None}, all at the input size.
class CamoEdgeNet(nn.Module):
    # pretrained=True loads ImageNet weights for the encoder through timm (needs a download the first time)
    def __init__(self, encoderName="pvt_v2_b2", decoderChannels=defaultDecoderChannels, useEdgeBranch=True, useEdgeGating=True,
                 gatingMode="multiply", useImageGradient=True, pretrained=True):
        super().__init__()
        import timm
        self.encoder = timm.create_model(encoderName, pretrained=pretrained, features_only=True)
        reductions, channels = self.encoder.feature_info.reduction(), self.encoder.feature_info.channels()
        self.featureIndices = [reductions.index(stride) for stride in (4, 8, 16, 32)]
        encoderChannels = [channels[index] for index in self.featureIndices]
        self.useEdgeBranch, self.useEdgeGating = useEdgeBranch, useEdgeBranch and useEdgeGating
        self.lateral = nn.ModuleList(nn.Conv2d(channelCount, decoderChannels, 1) for channelCount in encoderChannels)
        self.smooth = nn.ModuleList(convBlock(decoderChannels, decoderChannels) for _ in encoderChannels)
        self.sideOutputs = nn.ModuleList(nn.Conv2d(decoderChannels, 1, 1) for _ in encoderChannels)
        self.edgeBranch = EdgeGuidanceBranch(encoderChannels[0], decoderChannels, useImageGradient) if useEdgeBranch else None
        self.gates = nn.ModuleList(EdgeAttentionGate(decoderChannels, decoderChannels, gatingMode) for _ in encoderChannels) if self.useEdgeGating else None

    # Encoder features at strides 4, 8, 16 and 32 as (batch, channels, height, width) tensors
    def encode(self, image):
        features = self.encoder(image)
        return [features[index] for index in self.featureIndices]

    # Run the network on a normalised image batch
    def forward(self, image):
        inputSize = image.shape[-2:]
        encoderFeatures = self.encode(image)
        edgeFeature, edgeLogits = self.edgeBranch(encoderFeatures[0], image) if self.edgeBranch is not None else (None, None)
        decoderFeatures, previous = [], None
        for level in reversed(range(len(encoderFeatures))):  # from the deepest (stride 32) to the shallowest (stride 4)
            feature = self.lateral[level](encoderFeatures[level])
            if previous is not None: feature = feature + F.interpolate(previous, size=feature.shape[-2:], mode="bilinear", align_corners=False)
            if self.gates is not None: feature = self.gates[level](feature, edgeFeature)
            previous = self.smooth[level](feature)
            decoderFeatures.append((level, previous))
        masks = [F.interpolate(self.sideOutputs[level](feature), size=inputSize, mode="bilinear", align_corners=False) for level, feature in reversed(decoderFeatures)]
        edge = F.interpolate(edgeLogits, size=inputSize, mode="bilinear", align_corners=False) if edgeLogits is not None else None
        return {"masks": masks, "edge": edge}


# Build the network for a named experiment (E1 to E5), with optional overrides such as gatingMode or encoderName
def buildExperimentModel(experimentName, encoderName="pvt_v2_b2", pretrained=True, **overrides):
    settings = {**experimentModelSettings[experimentName], **overrides}
    return CamoEdgeNet(encoderName=encoderName, pretrained=pretrained, **settings)


# Number of parameters of a model
def countParameters(model):
    return sum(parameter.numel() for parameter in model.parameters())
