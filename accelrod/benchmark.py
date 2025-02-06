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


def benchmark_GEMM_wrapper(dtype=torch.float32, number=50):
    """
    run the benchmark for GEMM with different matrix size
    """
    bytes_per_element = get_bytes_by_dtype(dtype)
    # convert MB to bytes
    total_free_bytes = get_gpu_free_memory() * 0.8 * 1024**2

    # calculate the max_n based on the free memory
    max_n = np.sqrt(total_free_bytes / 4 / bytes_per_element)

    # Using your existing max_n value
    sequence = get_power_of_two_sequence(max_n)
    max_n = max(sequence)
    print(get_gpu_free_memory())

    result = []
    for n in sequence:
        result.append(
            benchmark_GEMM(
                matrix_shape=(max_n, n, max_n),
                dtype=dtype,
                number=number,
            )
        )
    return result


def timer_GEMM(m, k, n, dtype=torch.float32, device=None, number=50) -> benchmark.Timer:
    a = torch.randn(m, k, dtype=dtype, device=device)
    b = torch.randn(k, n, dtype=dtype, device=device)
    c = torch.randn(m, n, dtype=dtype, device=device)

    t = benchmark.Timer(
        stmt=f"d = a @ b + c; torch.{device}.synchronize()",
        globals={"a": a, "b": b, "c": c},
    )
    x = t.timeit(number=number)

    return x


def benchmark_GEMM(matrix_shape, dtype=torch.float16, device=None, number=50):
    print(f"matrix shape: {matrix_shape}")

    if device is None:
        device = get_device()
        print(f"device is None, automatically set to {device}")
    device = torch.device(device)
    print(f"device is {device}")

    (m, k, n) = matrix_shape
    # get bytes based on the dtype
    bytes_per_element = get_bytes_by_dtype(dtype)
    print(f"dtype is {dtype}, bytes_per_element: {bytes_per_element}")
    x = timer_GEMM(m=m, k=k, n=n, dtype=dtype, device=device, number=number)

    number_FLOPS = 2 * m * n * k + m * n
    number_bytes_accesses = bytes_per_element * (
        m * k + k * n + 2 * m * n
    )  # acccess all the data one time
    # arithmetic intensity to the ops:byte ratio of the GPU
    arithmetic_intensity = number_FLOPS / number_bytes_accesses

    # median tflops
    tflops = number_FLOPS / x.mean / 1e12

    print(f"tflops: {tflops}, x: {x.mean}, arithmetic_intensity: {arithmetic_intensity}")

    return tflops, x, arithmetic_intensity
