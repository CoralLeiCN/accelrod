from torch.utils import benchmark
import torch


# determine device
def get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


def benchmark_GEMM(matrix_shape, dtype=torch.float16, device=None, number=50):
    if device is None:
        device = get_device()
        print(f"device is None, automatically set to {device}")

    device = torch.device(device)
    typ = dtype
    # get bytes based on the dtype
    bytes_per_element = torch.tensor([], dtype=dtype).element_size()
    print(f"dtype is {dtype}, bytes_per_element: {bytes_per_element}")

    (m, k, n) = matrix_shape
    a = torch.randn(m, k, dtype=typ, device=device)
    b = torch.randn(k, n, dtype=typ, device=device)
    c = torch.randn(m, n, dtype=typ, device=device)
    print(a.device)
    print(b.device)
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
