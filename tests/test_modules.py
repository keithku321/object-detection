# Tests for the custom layers (common/modules.py) and the YAML tools (common/registerModules.py); they need PyTorch but no weights
import pytest

torch = pytest.importorskip("torch")

from common.modules import CoordinateAttention, DySample, ECA, SimAM, WeightedConcat, identityGateBias  # noqa: E402
from common.registerModules import channelsPlaceholder, findConcatLayers, insertLayersIntoYaml, prefixLayerMap, scaledYamlName, writeModelYaml  # noqa: E402

# A cut-down model YAML with the same wiring pattern as yolo26.yaml: backbone, a neck with upsample and concat, and a head row
tinyYaml = {
    "nc": 80, "end2end": True, "reg_max": 1, "scales": {"n": [0.5, 0.25, 1024]},
    "backbone": [[-1, 1, "Conv", [64, 3, 2]], [-1, 1, "Conv", [128, 3, 2]], [-1, 2, "C3k2", [256, False, 0.25]], [-1, 1, "Conv", [256, 3, 2]]],
    "head": [[-1, 1, "nn.Upsample", ["None", 2, "nearest"]], [[-1, 2], 1, "Concat", [1]], [-1, 2, "C3k2", [256, True]], [[6, 3], 1, "Detect", ["nc"]]],
}


# Attention layers keep the shape of their input and create their parameters on the first forward pass
@pytest.mark.parametrize("layerFactory", [lambda: ECA(), lambda: ECA(5), lambda: SimAM(), lambda: CoordinateAttention()])
def testAttentionKeepsShape(layerFactory):
    layer, features = layerFactory(), torch.randn(2, 64, 20, 24)
    assert layer(features).shape == features.shape
    assert layer(torch.randn(1, 64, 8, 8)).shape == (1, 64, 8, 8)


# ECA and Coordinate Attention start close to the identity, so inserting them does not disturb pretrained features
@pytest.mark.parametrize("layerFactory", [lambda: ECA(), lambda: CoordinateAttention()])
def testGatesStartNearIdentity(layerFactory):
    torch.manual_seed(0)
    layer, features = layerFactory().eval(), torch.randn(2, 64, 16, 16)
    relativeChange = (layer(features) - features).abs().mean() / features.abs().mean()
    assert relativeChange < 0.1
    assert torch.sigmoid(torch.tensor(identityGateBias)) > 0.98


# The adaptive ECA kernel follows the paper's formula and is always odd
def testEcaKernelSize():
    assert [ECA.adaptiveKernelSize(channels) for channels in (64, 128, 256, 512)] == [3, 5, 5, 5]


# A layer that needs the channel count has no parameters until it has seen an input, then a normal state dict
def testChannelAwareLayerBuildsOnce():
    layer = CoordinateAttention()
    assert sum(parameter.numel() for parameter in layer.parameters()) == 0
    layer(torch.randn(1, 128, 8, 8))
    parameterCount = sum(parameter.numel() for parameter in layer.parameters())
    assert parameterCount > 0 and layer.channels == 128
    layer(torch.randn(1, 128, 4, 4))
    assert sum(parameter.numel() for parameter in layer.parameters()) == parameterCount


# With zero offsets DySample is exactly bilinear upsampling, which is the starting point the paper designs for
def testDySampleStartsAsBilinear():
    torch.manual_seed(0)
    layer, features = DySample(scale=2, groups=4), torch.randn(2, 16, 10, 12)
    assert layer(features).shape == (2, 16, 20, 24)
    torch.nn.init.zeros_(layer.offset.weight)
    expected = torch.nn.functional.interpolate(features, scale_factor=2, mode="bilinear", align_corners=False)
    assert torch.allclose(layer(features)[:, :, 1:-1, 1:-1], expected[:, :, 1:-1, 1:-1], atol=1e-5)


# Every custom layer gives the same result in channels_last memory format, which Ultralytics uses for validation and inference
@pytest.mark.parametrize("layerFactory", [lambda: ECA(), lambda: SimAM(), lambda: CoordinateAttention(), lambda: DySample(scale=2, groups=4)])
def testLayersWorkInChannelsLast(layerFactory):
    torch.manual_seed(0)
    layer, features = layerFactory().eval(), torch.randn(2, 16, 10, 12)
    expected = layer(features)
    layer = layer.to(memory_format=torch.channels_last)
    assert torch.allclose(layer(features.to(memory_format=torch.channels_last)), expected, atol=1e-5)


# DySample refuses a channel count that its groups do not divide
def testDySampleChecksGroups():
    with pytest.raises(ValueError):
        DySample(scale=2, groups=4)(torch.randn(1, 6, 4, 4))


# A new weighted Concat equals a plain concatenation, and its weights receive gradients
def testWeightedConcatStartsAsConcat():
    layer, first, second = WeightedConcat(inputCount=2), torch.randn(1, 4, 5, 5), torch.randn(1, 6, 5, 5)
    output = layer([first, second])
    assert torch.allclose(output, torch.cat([first, second], 1), atol=1e-3)
    output.sum().backward()
    assert layer.fusionWeights.grad is not None


# Inserting a layer shifts later indices, and every reference to the layer it follows is rewired to the new layer
def testInsertLayerRewiresReferences():
    newYaml, layerMap = insertLayersIntoYaml(tinyYaml, insertions=[(2, "ECA", [0])])
    rows = newYaml["backbone"] + newYaml["head"]
    assert rows[3] == [-1, 1, "ECA", [0]]
    assert layerMap == {0: 0, 1: 1, 2: 2, 3: 4, 4: 5, 5: 6, 6: 7, 7: 8}
    assert rows[6][0] == [-1, 3]  # the Concat that read base layer 2 now reads the attention layer
    assert rows[8][0] == [7, 4]  # the head reads the shifted neck layer and the shifted backbone layer
    assert len(newYaml["backbone"]) == 5 and len(newYaml["head"]) == 4
    assert tinyYaml["head"][1][0] == [-1, 2]  # the base YAML is not modified


# Insertions after layers that feed the head rewire the head, and a replacement keeps every index
def testInsertAfterHeadInputsAndReplace():
    newYaml, layerMap = insertLayersIntoYaml(tinyYaml, insertions=[(3, "SimAM", [0.0001]), (6, "SimAM", [0.0001])], replacements={4: ("DySample", [2, 4])})
    rows = newYaml["backbone"] + newYaml["head"]
    assert rows[layerMap[4]][2] == "DySample"
    assert rows[-1][0] == [layerMap[6] + 1, layerMap[3] + 1]
    assert findConcatLayers(newYaml) == [layerMap[5]]


# The channel placeholder is filled with the channel count of the layer the insertion follows
def testChannelPlaceholderIsFilled():
    newYaml, _ = insertLayersIntoYaml(tinyYaml, insertions=[(2, "CBAM", [channelsPlaceholder, 7])], layerChannels={2: 64})
    assert newYaml["backbone"][3] == [-1, 1, "CBAM", [64, 7]]
    with pytest.raises(ValueError):
        insertLayersIntoYaml(tinyYaml, insertions=[(2, "CBAM", [channelsPlaceholder, 7])])


# A written model YAML loads back to the same dict, so nothing is lost between generation and training
def testWrittenYamlLoadsBack(tmp_path):
    yaml = pytest.importorskip("yaml")
    newYaml, _ = insertLayersIntoYaml(tinyYaml, insertions=[(2, "ECA", [0])])
    yamlPath = writeModelYaml(newYaml, tmp_path / "yolo26n-eca.yaml", "generated for a test")
    assert yaml.safe_load(yamlPath.read_text(encoding="utf-8")) == newYaml


# The scale letter goes right after the family name, where Ultralytics looks for it
def testScaledYamlName():
    assert scaledYamlName("yolo26.yaml", "n") == "yolo26n.yaml"
    assert scaledYamlName("yolo26-p2.yaml", "s", "eca") == "yolo26s-p2-eca.yaml"
    assert prefixLayerMap(3) == {0: 0, 1: 1, 2: 2}
