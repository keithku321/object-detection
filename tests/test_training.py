# Tests for the shared training loop helpers (common/training.py) that need no training run
import pytest

from common.training import checkRunEpochs, readRunEpochs


# A run folder started with another epoch count is refused, the same count passes, and a folder without args.yaml is not checked
def testRunFolderOfAnotherEpochCountIsRefused(tmp_path):
    (tmp_path / "args.yaml").write_text("epochs: 20\nimgsz: 640\n", encoding="utf-8")
    assert readRunEpochs(tmp_path) == 20
    checkRunEpochs(tmp_path, 20)
    with pytest.raises(ValueError, match="20 epochs, not 50"):
        checkRunEpochs(tmp_path, 50)
    checkRunEpochs(tmp_path / "missing", 50)
