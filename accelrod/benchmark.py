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


def benchmark_GEMM_wrapper(device=None, dtype=torch.float32, number=50):
    """Run the benchmark for General Matrix Multiplication (GEMM) with different matrix sizes and data types.

    Args:
        device (Union[str, torch.device, None]): The device to run the benchmark on.
            If None, it will automatically set to the available device. Defaults to None.
        dtype (torch.dtype, optional): The data type of the matrices.
            Defaults to torch.float32.
        number (int, optional): The number of times to run the benchmark for each matrix size.
            Defaults to 50.

    Returns:
        list: A list of benchmark results for each matrix size in the sequence.
    """

    bytes_per_element = get_bytes_by_dtype(dtype)
    print(f"dtype is {dtype}, bytes_per_element: {bytes_per_element}")
    # total free bytes to use, convert MB to bytes
    total_free_bytes = get_gpu_free_memory() * 0.8 * 1024**2

    # calculate the max_n based on the free memory, 4 is four matrices, A, B, C, and D.
    max_n = np.sqrt(total_free_bytes / 4 / bytes_per_element)

    # Using your existing max_n value
    sequence = get_power_of_two_sequence(max_n)
    max_n = max(sequence)

    print(f"Free memory is {get_gpu_free_memory()} MiB")
    print(f"maximum matrix size is {max_n}")

    result = []
    for n in sequence:
        result.append(
            benchmark_GEMM(
                matrix_shape=(max_n, n, max_n),
                dtype=dtype,
                device=device,
                number=number,
            )
        )
    return result


def timer_GEMM(m, k, n, dtype=torch.float32, device=None, number=50) -> benchmark.Timer:
    """Times the execution of a General Matrix Multiplication (GEMM) operation.

    Performs the operation D = A @ B + C where:
    - A is an m x k matrix
    - B is a k x n matrix
    - C is an m x n matrix
    The matrices are initialized with random values.

    Args:
        m (int): Number of rows in matrices A and C
        k (int): Number of columns in matrix A and rows in matrix B
        n (int): Number of columns in matrices B and C
        dtype (torch.dtype, optional): Data type of the matrices. Defaults to torch.float32.
        device (torch.device, optional): Device to run the computation on. Defaults to None.
        number (int, optional): Number of iterations for timing. Defaults to 50.

    Returns:
        benchmark.Timer: Timer object containing result for the GEMM operation.
    """
    a = torch.randn(m, k, dtype=dtype, device=device)
    b = torch.randn(k, n, dtype=dtype, device=device)
    c = torch.randn(m, n, dtype=dtype, device=device)

    t = benchmark.Timer(
        stmt=f"d = a @ b + c; torch.{device}.synchronize()",
        globals={"a": a, "b": b, "c": c},
    )
    x = t.timeit(number=number)

    return x


def calculate_arithmetic_intensity(m, k, n, dtype):
    """
    calculate the arithmetic intensity of the GEMM operation given the matrix size

    Args:
    m (int): Number of rows in matrices A and C
    k (int): Number of columns in matrix A and rows in matrix B
    n (int): Number of columns in matrices B and C
    dtype (torch.dtype): Data type of the matrx

    Returns:
        arithmetic_intensity (float): arithmetic intensity of the GEMM operation
        number_FLOPS (int): number of FLOPS of the GEMM operation


    """

    # get bytes based on the dtype
    bytes_per_element = get_bytes_by_dtype(dtype)

    number_FLOPS = 2 * m * n * k + m * n

    # acccess all the data one time, including read and write
    number_bytes_accesses = bytes_per_element * (m * k + k * n + 2 * m * n)
    # arithmetic intensity to the ops:byte ratio of the GPU
    arithmetic_intensity = number_FLOPS / number_bytes_accesses

    return arithmetic_intensity, number_FLOPS


def benchmark_GEMM(matrix_dim, dtype, device, number):
    """Benchmarks the General Matrix Multiply (GEMM) operation for a given matrix size.

    Args:
        matrix_dim (tuple): A tuple (m, k, n) representing the dimensions of the matrices
            involved in the GEMM operation.
        dtype (str): The data type of the matrices (e.g., 'torch.float32', 'torch.float16').
        device (str): The device on which to perform the GEMM operation (e.g., 'cpu', 'cuda').
        number (int): The number of times to repeat the GEMM operation for benchmarking.

    Returns:
        tuple:
            tflops (float): The median teraflops achieved during the GEMM operation.
            x (benchmark.Timer): Timer object containing result for the GEMM operation.
            arithmetic_intensity (float): The arithmetic intensity of the GEMM operation.
    """
    (m, k, n) = matrix_dim
    # get bytes based on the dtype
    x = timer_GEMM(m=m, k=k, n=n, dtype=dtype, device=device, number=number)

    arithmetic_intensity, number_FLOPS = calculate_arithmetic_intensity(m, k, n, dtype)

    # mean tflops
    # statistics in pytorch timer are bugged, for instance mean and median are the same.
    tflops = number_FLOPS / x.mean / 1e12
    print(f"tflops: {tflops}, x: {x.mean}, arithmetic_intensity: {arithmetic_intensity}")

    return tflops, x, arithmetic_intensity


def benchmark(algorithm="GEMM", device="auto"):
    """
    Main function to run the benchmark.
    """
    if algorithm == "GEMM":
        result = benchmark_GEMM_wrapper()
    else:
        raise ValueError(f"algorithm {algorithm} is not implemented")

    if device == "auto":
        device = get_device()
        print(f"device is auto, automatically set to {device}")
    else:
        device = torch.device(device)
    print(f"device is {device}")

    df = to_pandas(result)
    plot_result(df)
    return df
