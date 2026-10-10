# Poisson signal confidence intervals

Source code for the Bayesian, IM, and NIM interval procedures and two observed-data applications discussed in the manuscript. The three simulation entry points use the shared implementation in `code.py`.

| Entry point | Procedure |
| --- | --- |
| `bayesian.py` | Bayesian initial interval and h-function refinement |
| `im.py` | IM initial interval and h-function refinement |
| `nim.py` | NIM initial interval and h-function refinement |

Keep `code.py` alongside the three simulation entry points. The H.E.S.S. application also imports it by path. Run the named entry points rather than importing `code` directly, since `code` is also a Python standard-library module. The KARMEN2 application contains its own numerical implementation.

## Installation

The manuscript reports Python 3.12.13 for its simulations. Install the direct
Python dependencies in a compatible environment:

```bash
python -m pip install -r requirements.txt
```

PyTorch installation depends on the available CPU or CUDA environment. For GPU runs, install a PyTorch build compatible with the local CUDA setup. The full manuscript grids can require substantial compute time and GPU memory.

## Simulation entry points

Run one method at a time from the project root. For example:

```bash
python bayesian.py --m 20 --alpha 0.05
```

Use `im.py` or `nim.py` for the other two methods.

`--alpha` is the significance level: `0.10` and `0.05` request 90% and 95% confidence intervals, respectively. `--m` and `--alpha` accept multiple values. Each entry point fixes its method; run `python code.py --help` for the remaining shared options. Outputs are written to the current working directory. Use a separate working directory for each method if you need to retain the runtime and nesting summary CSVs, whose names do not include the method.

## Observed-data applications

### KARMEN2

`experiments/karmen2/run_karmen2.py` uses the observed count `x=15`, background estimate `15.8`, and background standard deviation `0.5`:

```bash
python experiments/karmen2/run_karmen2.py
```

The `EmpiricalConfig` block at the end of the file selects methods and refinement iterations. Its current defaults run only NIM with `max_iter=0`; in this configuration C1 and the reported final interval copy C0. To request all three procedures and refinements for comparison with manuscript Table 8, set `methods_to_run=["Bayes", "IM", "NIM"]` and `max_iter` to at least 1. Allow additional iterations when assessing convergence. The script writes its interval and optional lookup-table CSVs to the current working directory. Its KARMEN-specific implementation uses `float32` precision.

### H.E.S.S. GRB 221009A

`experiments/hess_grb221009a/run_hess_single_bin.py` uses `X=0`, `W=13`, `alpha_off=0.0741`, and a 90% confidence level:

```bash
python experiments/hess_grb221009a/run_hess_single_bin.py
```

The script uses `code.py` and writes results to an output directory. See the
[H.E.S.S. run instructions](experiments/hess_grb221009a/README.md) for method
selection, settings, and output files. Check each method's `Converged` value
in `experiment_manifest.json`: `C_final` is the last computed stage and does
not by itself indicate convergence. This source package does not include
numerical result files.
