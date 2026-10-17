# Small helpers for telling Colab, Windows and local machines apart, so scripts pick safe defaults
import platform
import sys

# Dataloader worker processes to use on Windows, where many workers often hang or run slower
windowsWorkerCount = 2
# Dataloader worker processes to use on Linux and Colab
linuxWorkerCount = 4


# True when the code runs inside Google Colab (checked without importing google.colab)
def isColab():
    return "google.colab" in sys.modules


# True on Windows, where dataloader workers and file paths need extra care
def isWindows():
    return platform.system() == "Windows"


# Number of dataloader workers that is safe for the current machine
def getDefaultWorkerCount():
    return windowsWorkerCount if isWindows() else linuxWorkerCount


# Pick "cuda:0" when a GPU is usable and "cpu" otherwise
def getDefaultDevice():
    try:
        import torch
    except ImportError:
        return "cpu"
    return "cuda:0" if torch.cuda.is_available() else "cpu"


# Print Python, PyTorch, Ultralytics and GPU details that belong in every bug report and every results row
def printEnvironment():
    print(f"Python {platform.python_version()} on {platform.system()}")
    try:
        import torch
        print(f"torch {torch.__version__}, CUDA build {torch.version.cuda}, GPU available: {torch.cuda.is_available()}")
        if torch.cuda.is_available(): print(f"GPU: {torch.cuda.get_device_name(0)}")
        elif torch.version.cuda is None: print("WARNING: this is a CPU-only PyTorch build; see requirements-local.txt")
    except ImportError:
        print("torch is not installed")
    try:
        import ultralytics
        print(f"ultralytics {ultralytics.__version__}")
    except ImportError:
        print("ultralytics is not installed")
