from torch.utils import benchmark
import torch
import pandas as pd
import matplotlib.pyplot as plt
import subprocess


def get_power_of_two_sequence(N):
    """
    Returns a list of integers where each number is a power of 2, the largest number <= N.

    Args:
        N (int): The upper limit of the sequence

    Returns:
        list: A list of powers of 2 up to N
    """
    sequence = []
    power = 0
    while 2**power <= N:
        sequence.append(2**power)
        power += 1
    return sequence


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


# get bytes based on the dtype
def get_bytes_by_dtype(dtype):
    bytes_per_element = torch.tensor([], dtype=dtype).element_size()
    return bytes_per_element


def to_pandas(result):
    df = pd.DataFrame(result, columns=["tflops", "time", "arithmetic_intensity"])
    df["median_time"] = df["time"].apply(lambda x: x.median)
    return df


def plot_result(df):
    # plot the results, tflops against arithmetic intensity
    plt.plot(df["arithmetic_intensity"], df["tflops"], "o-")
    plt.xlabel("Arithmetic Intensity")
    plt.ylabel("TFLOPS")
    plt.title("Performance")
    plt.show()


def benchmark_GEMM(matrix_shape, dtype=torch.float16, device=None, number=50):
    if device is None:
        device = get_device()
        print(f"device is None, automatically set to {device}")

    device = torch.device(device)
    # get bytes based on the dtype
    bytes_per_element = get_bytes_by_dtype(dtype)
    print(f"dtype is {dtype}, bytes_per_element: {bytes_per_element}")

    (m, k, n) = matrix_shape
    print(f"matrix shape: {matrix_shape}")
    a = torch.randn(m, k, dtype=dtype, device=device)
    b = torch.randn(k, n, dtype=dtype, device=device)
    c = torch.randn(m, n, dtype=dtype, device=device)
    print(a.device)
    print(b.device)
    print(get_gpu_free_memory())
    t = benchmark.Timer(
        stmt=f"a @ b + c; torch.{device}.synchronize()",
        globals={"a": a, "b": b, "c": c},
    )

    x = t.timeit(number=number)
    number_FLOPS = 2 * m * n * k
    number_bytes_accesses = bytes_per_element * (
        m * k + k * n + m * n
    )  # acccess all the data one time
    # arithmetic intensity to the ops:byte ratio of the GPU
    arithmetic_intensity = number_FLOPS / number_bytes_accesses

    # median tflops
    tflops = number_FLOPS / x.mean / 1e12

    print(f"tflops: {tflops}, x: {x.mean}, arithmetic_intensity: {arithmetic_intensity}")

    return tflops, x, arithmetic_intensity
