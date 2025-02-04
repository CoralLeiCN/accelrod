import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils import benchmark

from accelrod.device import get_device, get_gpu_free_memory
from accelrod.utils import get_power_of_two_sequence


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


def benchmark_GEMM_wrapper(dtype=torch.float32):
    """
    run the benchmark for GEMM with different matrix size
    """
    # convert MB to bytes
    total_free_bytes = get_gpu_free_memory() * 0.8 * 1024**2
    # temporary assume benchmarking start with float64 so 8 bytes
    max_n = np.sqrt(total_free_bytes / 5 / 8)

    # Using your existing max_n value
    sequence = get_power_of_two_sequence(max_n)
    max_n = max(sequence)

    result = []
    for n in sequence:
        # for n in [1024]:
        result.append(
            benchmark_GEMM(
                matrix_shape=(max_n, n, max_n),
                dtype=dtype,
                number=20,
            )
        )
    return result


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
