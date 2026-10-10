# H.E.S.S. GRB 221009A single-bin analysis

This runner applies the current repository implementation to the official
all-nights H.E.S.S. auxiliary-table bin 6.309–9.999 TeV:

- `X = N_ON = 0`
- `W = N_OFF = 13`
- `alpha_off = 0.0741`, so `m = 1 / alpha_off`
- target confidence level 0.90

Run from the project root with Python and the dependencies used by `code.py`
(`torch`, `numpy`, and `tqdm`) installed:

```bash
python experiments/hess_grb221009a/run_hess_single_bin.py
```

The script loads `code.py` from the project root, so keep that file in place
when running or packaging this example. Use `--methods Bayes IM NIM` to select
methods and `--output-dir PATH` to change the output directory; the default
runs all three methods.

It reuses `code.py` for Bayesian, IM, NIM, Berger–Boos bounds,
quantile truncation, and interval refinement. It recursively builds only the
stage tables needed by the observed-cell interval, rather than refining every
cell in the outer LUT at every stage. It uses the code's Bayesian
prior values `a = b = 1`, IM/NIM auxiliary sample size 10,000, seed 2026,
`beta = 1e-4`, `eta = 1e-9`, 1,000 nuisance-grid points, 500 lambda-grid
points, at most 10 refinement iterations, and convergence tolerance `1e-4`.
The RNG is reset to seed 2026 separately for each method and support retry.
For IM/NIM, changing the LUT shape can change which random draws map to each
count pair; the accepted support dimensions and environment are saved so the
final run can be reproduced.
The runner's `ALPHA_CI = 0.10` is the interval significance level; it is
distinct from the rounded H.E.S.S. ON/OFF ratio `ALPHA_OFF = 0.0741`.

Bayesian C0 uses a 200-observation batch (the core function's default) to fit
the expanded H.E.S.S.-specific support domain within GPU memory. This is only
a batching change; the posterior formula and lambda grid are unchanged. The
previous GPU invocation set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
to reduce fragmentation. Set that environment variable in the shell if needed;
the runner records its value in the manifest.

The runner does not use the simulation-only truth parameter `eb` and does not
compute coverage or expected length. It starts from the observed-cell C0 upper
endpoint and uses the manuscript's dynamic Poisson-quantile truncation rule.
For each Ck, it recursively builds the preceding-stage interval values over
the count pairs needed by that h-function sum. If the C0 base table is too
small, it expands the table and reruns the method with the same seed. The retry
ceiling (8 attempts) and expansion guard (2 integer cells) are operational
safeguards, not inferential tuning parameters; both are recorded in the
manifest. The convergence threshold is applied to the observed cell's endpoint
change, as this is a single-observation run; the manifest records this scope.

Outputs are written to `results_hess_grb221009a_20261009_targeted/` by default:

- `intervals.csv`: C0, C1, and last-computed (`C_final`) endpoint and length
  for each method.
- `experiment_manifest.json`: input provenance, settings, code hash, runtime
  environment, iteration count, convergence, and observed-cell nesting checks.
  Check each method's `Converged` value: `C_final` means the last computed
  stage and does not itself establish convergence.
- `support_audit.csv` and `support_attempts.csv`: recursive lookup requests
  and any required support expansions.

This returns intervals for expected signal event counts. It does not convert
them to flux limits.
