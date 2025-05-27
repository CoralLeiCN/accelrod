# %%
import torch

from accelrod.benchmark import benchrun, plot_result

# %%
result = benchrun(
    algorithm="GEMM",
    device="auto",
    as_dataframe="pandas",
    params={"max_dimension": 2048, "dtype": [torch.float32, torch.float16]},
)

# %%
import plotly.offline as pyo

pyo.init_notebook_mode()
plot_result(result)

# %%



