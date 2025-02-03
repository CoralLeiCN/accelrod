import subprocess

import torch


def get_gpu_free_memory():
    result = subprocess.check_output(
        "nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits",
        shell=True,
        encoding="utf-8",
    )
    return float(result)


# determine device
def get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


def get_device_properties():
    if torch.cuda.is_available():
        return torch.cuda.get_device_properties(torch.cuda.device("cuda"))
    elif torch.backends.mps.is_available():
        return
    else:
        return
