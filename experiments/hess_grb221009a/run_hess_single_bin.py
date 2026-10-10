#!/usr/bin/env python3
"""Run the paper's three interval procedures on the selected H.E.S.S. bin.

This runner imports the repository's ``code.py`` and reuses its initial
interval and h-function implementations. It does not call the simulation
driver and does not assign a true background mean to the observed dataset.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
CORE_PATH = PROJECT_ROOT / "code.py"
DEFAULT_OUTPUT = PROJECT_ROOT / "results_hess_grb221009a_20261009_targeted"

# Official H.E.S.S. auxiliary table, all nights, 6.309--9.999 TeV.
X_OBS = 0
W_OBS = 13
ALPHA_OFF = 0.0741
M = 1.0 / ALPHA_OFF

# Manuscript / current implementation settings. Here alpha_ci is the test
# significance level, distinct from the H.E.S.S. ON/OFF ratio above.
TARGET_CI = 0.90
ALPHA_CI = 1.0 - TARGET_CI
BETA = 1e-4
ETA = 1e-9
AUXILIARY_N = 10_000
SEED = 2026
MAX_ITER = 10
CONVERGENCE_TOL = 1e-4
NESTING_TOLERANCE = 1e-4
NUM_B_GRID = 1000
NUM_LAMBDA_GRID = 500
METHODS = ("Bayes", "IM", "NIM")

INTERVAL_FIELDS = [
    "Method", "Stage", "Iteration", "Lower", "Upper", "Length",
    "X", "W", "m", "Alpha_OFF", "Target_CI", "Alpha_CI", "Beta", "Eta",
]
SUPPORT_AUDIT_FIELDS = [
    "Method", "Attempt", "Stage", "Requested_X_Max", "Requested_W_Max",
    "Required_Lookup_X_Max", "Required_Lookup_W_Max",
    "Base_LUT_X_Max", "Base_LUT_W_Max", "Status",
]
SUPPORT_ATTEMPT_FIELDS = [
    "Method", "Attempt", "X_Max", "W_Max", "Status",
    "Required_X_Max", "Required_W_Max",
]

# Support-domain retry mechanics are engineering safeguards, not model
# parameters. A failed dependency audit causes a fresh run on a larger LUT.
MAX_SUPPORT_ATTEMPTS = 8
SUPPORT_GUARD_CELLS = 2


class SupportTooSmall(RuntimeError):
    """Request a fresh C0 table covering the full recursive lookup support."""

    def __init__(self, required_x: int, required_w: int,
                 audit_rows: list[dict[str, Any]]):
        super().__init__(
            f"Current LUT is too small; dependency cone requires at least "
            f"X_max={required_x}, W_max={required_w}."
        )
        self.required_x = required_x
        self.required_w = required_w
        self.audit_rows = audit_rows


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_core():
    """Load the shared implementation from the repository root."""

    if not CORE_PATH.is_file():
        raise FileNotFoundError(f"Core implementation not found: {CORE_PATH}")
    spec = importlib.util.spec_from_file_location("gpuhf_hess_core", CORE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load core implementation: {CORE_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(path)


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def seed_all(core, seed: int = SEED) -> None:
    """Reset the NumPy and PyTorch generators used by the shared core."""

    core.torch.manual_seed(seed)
    core.np.random.seed(seed)
    if core.torch.cuda.is_available():
        core.torch.cuda.manual_seed_all(seed)


def build_runtime(core):
    """Use the repository's device-specific batch recommendations."""
    recommended = core._get_recommended_batch_sizes()
    return SimpleNamespace(
        # The repository's Bayes core defaults to 200. The A100's automatic
        # 1,000-row batch OOMed on this data-specific support domain; this
        # changes memory batching only, not the posterior or lambda grid.
        chunk_size=min(recommended[0], 200),
        h_chunk_size=recommended[1],
        lut_chunk_size=recommended[2],
        h_lambda_batch_size=recommended[3],
        h_w_chunk_size=recommended[4],
        h_b_block_size=recommended[5],
        max_iter=MAX_ITER,
        convergence_tol=CONVERGENCE_TOL,
        num_b_grid=NUM_B_GRID,
        num_lambda_grid=NUM_LAMBDA_GRID,
        N=AUXILIARY_N,
        random_seed=SEED,
    )


def method_function(core, method: str, runtime):
    """Select the initial interval implementation for one method."""

    if method == "Bayes":
        def bayes_adapter(x, w, m_param, alpha, **_kwargs):
            return core.calculate_bayes_hpd_core(
                x, w, m_param, 1.0 - alpha,
                chunk_size=runtime.chunk_size,
            )
        return bayes_adapter, {}
    if method == "IM":
        return core.calculate_im_intervals, {"N": runtime.N}
    if method == "NIM":
        return core.calculate_nim_intervals, {"N": runtime.N}
    raise ValueError(f"Unsupported method: {method}")


def point_initial_interval(core, core_func, kwargs, runtime) -> tuple[float, float]:
    """Get a C0 point estimate for sizing the initial, data-specific LUT."""
    torch = core.torch
    devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    numpy_state = core.np.random.get_state()
    try:
        # Keep this sizing probe from advancing the RNG used by the actual LUT.
        with torch.random.fork_rng(devices=devices):
            seed_all(core, runtime.random_seed)
            x = torch.tensor([float(X_OBS)], device=core.device, dtype=core.dtype)
            w = torch.tensor([float(W_OBS)], device=core.device, dtype=core.dtype)
            lower, upper = core_func(x, w, M, ALPHA_CI, **kwargs)
            return float(lower[0].item()), float(upper[0].item())
    finally:
        core.np.random.set_state(numpy_state)


def get_target_support(core, upper: float) -> tuple[int, int, float, float]:
    """Size the first C0 table using the observed pair and its C0 upper bound."""

    torch = core.torch
    w = torch.tensor([float(W_OBS)], device=core.device, dtype=core.dtype)
    upper_t = torch.tensor([float(upper)], device=core.device, dtype=core.dtype)
    eps_lower_t, eps_upper_t = core.compute_berger_boos_bounds(w, M, BETA)
    x_cut, w_cut = core.compute_dynamic_sample_limits(
        w, upper_t, M, BETA, ETA
    )
    return (
        max(X_OBS, int(x_cut[0].item())),
        max(W_OBS, int(w_cut[0].item())),
        float(eps_lower_t[0].item()),
        float(eps_upper_t[0].item()),
    )


class StagewiseTargetCalculator:
    """Build only the interval tables needed by the observed-cell recursion.

    Every C_k table is computed on demand. Before refining an observation
    rectangle, the preceding-stage table is recursively expanded to contain
    the union of its data-specific Poisson-quantile supports. C0 is analytic
    or Monte Carlo and is stored once over the current base support.
    """

    def __init__(self, core, method: str, runtime, core_func, kwargs,
                 base_lower, base_upper, base_x: int, base_w: int,
                 attempt: int):
        self.core = core
        self.method = method
        self.runtime = runtime
        self.core_func = core_func
        self.kwargs = kwargs
        self.base_lower = base_lower
        self.base_upper = base_upper
        self.base_x = base_x
        self.base_w = base_w
        self.attempt = attempt
        self.cache: dict[int, list[tuple[int, int, Any, Any]]] = {}
        self.h_calculators: dict[int, Any] = {}
        self.support_rows: list[dict[str, Any]] = []

    def _cache_slice(self, stage: int, x_max: int, w_max: int):
        for cached_x, cached_w, lower, upper in self.cache.get(stage, []):
            if cached_x >= x_max and cached_w >= w_max:
                return lower[:x_max + 1, :w_max + 1], upper[:x_max + 1, :w_max + 1]
        return None

    def _record(self, stage: int, x_max: int, w_max: int,
                required_x: int, required_w: int, status: str) -> None:
        self.support_rows.append({
            "Method": self.method,
            "Attempt": self.attempt,
            "Stage": f"C{stage}",
            "Requested_X_Max": x_max,
            "Requested_W_Max": w_max,
            "Required_Lookup_X_Max": required_x,
            "Required_Lookup_W_Max": required_w,
            "Base_LUT_X_Max": self.base_x,
            "Base_LUT_W_Max": self.base_w,
            "Status": status,
        })

    def interval_table(self, stage: int, x_max: int, w_max: int):
        cached = self._cache_slice(stage, x_max, w_max)
        if cached is not None:
            return cached

        if stage == 0:
            if x_max > self.base_x or w_max > self.base_w:
                self._record(stage, x_max, w_max, x_max, w_max, "base_support_too_small")
                raise SupportTooSmall(x_max, w_max, self.support_rows)
            lower = self.base_lower[:x_max + 1, :w_max + 1]
            upper = self.base_upper[:x_max + 1, :w_max + 1]
            self.cache.setdefault(0, []).append((x_max, w_max, lower, upper))
            self._record(stage, x_max, w_max, x_max, w_max, "base_table_slice")
            return lower, upper

        lower_input, upper_input = self.interval_table(stage - 1, x_max, w_max)
        torch = self.core.torch
        x_values = torch.arange(x_max + 1, device=self.core.device, dtype=self.core.dtype)
        w_values = torch.arange(w_max + 1, device=self.core.device, dtype=self.core.dtype)
        _, w_mesh = torch.meshgrid(x_values, w_values, indexing="ij")
        x_cut, w_cut = self.core.compute_dynamic_sample_limits(
            w_mesh.reshape(-1), upper_input.reshape(-1), M, BETA, ETA
        )
        required_x = max(x_max, int(x_cut.max().item()))
        required_w = max(w_max, int(w_cut.max().item()))

        try:
            lower_lookup, upper_lookup = self.interval_table(
                stage - 1, required_x, required_w
            )
        except SupportTooSmall:
            self._record(stage, x_max, w_max, required_x, required_w,
                         "recursive_support_expansion_required")
            raise

        self._record(stage, x_max, w_max, required_x, required_w,
                     "recursive_support_covered")
        h_calc = self.h_calculators.get(stage)
        if h_calc is None:
            h_calc = self.core.HFunctionCalculator(
                M, BETA, ETA,
                num_b_grid=self.runtime.num_b_grid,
                num_lambda_grid=self.runtime.num_lambda_grid,
                lambda_eval_batch_size=self.runtime.h_lambda_batch_size,
                b_block_size=self.runtime.h_b_block_size,
            )
            self.h_calculators[stage] = h_calc
        h_calc.set_lut(lower_lookup, upper_lookup)

        lower_new, upper_new = self.core.refine_entire_lut_optimized(
            h_calc, lower_input, upper_input, ALPHA_CI,
            h_chunk_size=self.runtime.h_chunk_size,
            h_w_chunk_size=self.runtime.h_w_chunk_size,
            desc=f"{self.method} C{stage} required cells",
            is_first_iter=(stage == 1),
        )
        if stage >= 2:
            lower_new = torch.maximum(lower_new, lower_input)
            upper_new = torch.minimum(upper_new, upper_input)

        self.cache.setdefault(stage, []).append((x_max, w_max, lower_new, upper_new))
        return lower_new, upper_new


def compute_target_stages(core, method: str, runtime, core_func, kwargs,
                          x_max: int, w_max: int, attempt: int
                          ) -> tuple[list[tuple[Any, Any]], list[float], list[dict[str, Any]]]:
    """Run C0 -> C1 -> C_final for the observed pair only.

    The h-function itself is unchanged. Recursive calls populate only the
    preceding-stage cells needed by the requested observed-cell interval.
    Convergence is checked on the observed cell's endpoints, matching the
    single-data-case Algorithm 2 stopping rule.
    """
    torch = core.torch
    seed_all(core, runtime.random_seed)
    lower0, upper0 = core.generate_lut(
        core_func, M, ALPHA_CI, x_max, w_max,
        chunk_size=runtime.lut_chunk_size, **kwargs,
    )
    lower0 = torch.clamp(lower0, min=0.0)
    upper0 = torch.clamp(upper0, min=0.0)
    calculator = StagewiseTargetCalculator(
        core, method, runtime, core_func, kwargs,
        lower0, upper0, x_max, w_max, attempt,
    )

    stages = [(
        lower0[:X_OBS + 1, :W_OBS + 1],
        upper0[:X_OBS + 1, :W_OBS + 1],
    )]
    deltas: list[float] = []
    for iteration in range(1, runtime.max_iter + 1):
        lower_new, upper_new = calculator.interval_table(
            iteration, X_OBS, W_OBS
        )
        prev_lower, prev_upper = stages[-1]
        delta = max(
            abs(float(lower_new[X_OBS, W_OBS].item())
                - float(prev_lower[X_OBS, W_OBS].item())),
            abs(float(upper_new[X_OBS, W_OBS].item())
                - float(prev_upper[X_OBS, W_OBS].item())),
        )
        stages.append((lower_new, upper_new))
        deltas.append(delta)
        print(
            f"[{method}] iteration={iteration}, target_endpoint_change={delta:.8g}, "
            f"target_CI=[{lower_new[X_OBS, W_OBS].item():.8g}, "
            f"{upper_new[X_OBS, W_OBS].item():.8g}], "
            f"recursive_support_calls={len(calculator.support_rows)}",
            flush=True,
        )
        if delta < runtime.convergence_tol:
            break
    return stages, deltas, calculator.support_rows


def extract_interval(stages, stage_index: int) -> tuple[float, float]:
    """Read the observed count pair from one computed interval stage."""

    lower, upper = stages[stage_index]
    return (
        float(lower[X_OBS, W_OBS].detach().cpu().item()),
        float(upper[X_OBS, W_OBS].detach().cpu().item()),
    )


def build_manifest(core, runtime, methods: list[str],
                   eps_lower: float, eps_upper: float) -> dict[str, Any]:
    """Record inputs, configuration, provenance, and runtime environment."""
    return {
        "experiment": "H.E.S.S. GRB 221009A, highest energy bin",
        "status": "running",
        "started_at_utc": utc_now(),
        "source": {
            "paper_doi": "10.3847/2041-8213/acc405",
            "paper_arxiv": "https://arxiv.org/abs/2303.10558",
            "official_auxiliary_page": "https://www.mpi-hd.mpg.de/hfm/HESS/pages/publications/auxiliary/2023_GRB_221009A/",
            "energy_bin_TeV": [6.309, 9.999],
        },
        "data": {
            "X_ON": X_OBS,
            "W_OFF": W_OBS,
            "alpha_off_rounded": ALPHA_OFF,
            "m_inverse_alpha_off": M,
            "estimated_ON_background_W_over_m": W_OBS / M,
            "berger_boos_epsilon_interval": [eps_lower, eps_upper],
        },
        "model": {
            "X": "Poisson(lambda + epsilon)",
            "W": "Poisson(m * epsilon)",
            "epsilon_is_unknown_nuisance": True,
            "simulation_truth_epsilon_used": False,
        },
        "settings": {
            "target_confidence": TARGET_CI,
            "alpha_ci": ALPHA_CI,
            "beta": BETA,
            "eta": ETA,
            "N_IM_NIM": AUXILIARY_N,
            "seed_per_method": SEED,
            "rng_reset_policy": (
                "reset seed 2026 per method and support retry; IM/NIM cell-to-draw "
                "assignment depends on the accepted LUT shape and chunking"
            ),
            "max_iter": MAX_ITER,
            "convergence_tolerance": CONVERGENCE_TOL,
            "observed_cell_nesting_tolerance": NESTING_TOLERANCE,
            "num_b_grid": NUM_B_GRID,
            "num_lambda_grid": NUM_LAMBDA_GRID,
            "methods": methods,
            "interval_computation_scope": "target-focused recursive stage tables",
            "convergence_scope": "observed-cell endpoint change",
            "bayesian_prior_from_core_code": {"a": 1.0, "b": 1.0},
            "support_epsilon_upper_used_for_initial_LUT_only": eps_upper,
            "support_retry_guard_cells": SUPPORT_GUARD_CELLS,
            "max_support_attempts": MAX_SUPPORT_ATTEMPTS,
        },
        "core_code": {
            "path": str(CORE_PATH),
            "sha256": sha256(CORE_PATH),
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": core.torch.__version__,
            "torch_cuda": core.torch.version.cuda,
            "cuda_available": bool(core.torch.cuda.is_available()),
            "gpu": (core.torch.cuda.get_device_name(core.device)
                    if core.torch.cuda.is_available() else None),
            "numpy": core.np.__version__,
            "pytorch_cuda_allocator_config": os.environ.get(
                "PYTORCH_CUDA_ALLOC_CONF", ""
            ),
        },
        "runtime_batch_sizes": {
            "bayes_chunk": runtime.chunk_size,
            "lut_chunk": runtime.lut_chunk_size,
            "h_chunk": runtime.h_chunk_size,
            "h_lambda_batch": runtime.h_lambda_batch_size,
            "h_w_chunk": runtime.h_w_chunk_size,
            "h_b_block": runtime.h_b_block_size,
        },
        "completed_methods": [],
    }


def parse_args():
    """Accept an output directory and an optional subset of methods."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    return parser.parse_args()


def run_method(core, runtime, method: str):
    """Compute one method, including any support expansion retries."""
    result_rows: list[dict[str, Any]] = []
    support_rows: list[dict[str, Any]] = []
    attempt_rows: list[dict[str, Any]] = []

    method_start = time.perf_counter()
    core_func, kwargs = method_function(core, method, runtime)
    _, upper_probe = point_initial_interval(
        core, core_func, kwargs, runtime
    )
    initial_x_max, initial_w_max, eps_lower, eps_upper = get_target_support(
        core, upper_probe
    )
    x_max, w_max = initial_x_max, initial_w_max
    print(
        f"[{method}] C0 support seed: U0={upper_probe:.8g}, "
        f"epsilon_BB=[{eps_lower:.8g},{eps_upper:.8g}], "
        f"X_max={x_max}, W_max={w_max}",
        flush=True,
    )

    completed = None
    for attempt in range(1, MAX_SUPPORT_ATTEMPTS + 1):
        start = time.perf_counter()
        try:
            stages, deltas, this_audit = compute_target_stages(
                core, method, runtime, core_func, kwargs,
                x_max, w_max, attempt,
            )
            support_rows.extend(this_audit)
            required_x = max(
                row["Required_Lookup_X_Max"] for row in this_audit
            )
            required_w = max(
                row["Required_Lookup_W_Max"] for row in this_audit
            )
            completed = (stages, deltas, time.perf_counter() - start, attempt)
            attempt_rows.append({
                "Method": method, "Attempt": attempt,
                "X_Max": x_max, "W_Max": w_max,
                "Status": "accepted",
                "Required_X_Max": required_x,
                "Required_W_Max": required_w,
            })
            break
        except SupportTooSmall as error:
            support_rows.extend(error.audit_rows)
            attempt_rows.append({
                "Method": method, "Attempt": attempt,
                "X_Max": x_max, "W_Max": w_max,
                "Status": "expanded_after_failed_dependency_audit",
                "Required_X_Max": error.required_x,
                "Required_W_Max": error.required_w,
            })
            if attempt == MAX_SUPPORT_ATTEMPTS:
                raise RuntimeError(
                    f"Support audit did not pass for {method} after "
                    f"{MAX_SUPPORT_ATTEMPTS} attempts."
                ) from error
            x_max = max(x_max + SUPPORT_GUARD_CELLS,
                        error.required_x + SUPPORT_GUARD_CELLS)
            w_max = max(w_max + SUPPORT_GUARD_CELLS,
                        error.required_w + SUPPORT_GUARD_CELLS)
            if core.torch.cuda.is_available():
                core.torch.cuda.empty_cache()

    assert completed is not None
    stages, deltas, elapsed_s, accepted_attempt = completed
    c0_lower, c0_upper = extract_interval(stages, 0)
    c1_lower, c1_upper = extract_interval(stages, 1)
    final_lower, final_upper = extract_interval(stages, len(stages) - 1)
    converged = bool(deltas and deltas[-1] < CONVERGENCE_TOL)

    interval_items = [
        ("C0", c0_lower, c0_upper, 0),
        ("C1", c1_lower, c1_upper, 1),
        ("C_final", final_lower, final_upper, len(stages) - 1),
    ]
    for stage_name, lower, upper, stage_iteration in interval_items:
        if not (math.isfinite(lower) and math.isfinite(upper)):
            raise RuntimeError(f"Non-finite endpoint for {method} {stage_name}.")
        if lower < -1e-10 or lower > upper + 1e-10:
            raise RuntimeError(f"Invalid interval for {method} {stage_name}: [{lower}, {upper}].")
        result_rows.append({
            "Method": method,
            "Stage": stage_name,
            "Iteration": stage_iteration,
            "Lower": lower,
            "Upper": upper,
            "Length": upper - lower,
            "X": X_OBS,
            "W": W_OBS,
            "m": M,
            "Alpha_OFF": ALPHA_OFF,
            "Target_CI": TARGET_CI,
            "Alpha_CI": ALPHA_CI,
            "Beta": BETA,
            "Eta": ETA,
        })

    c1_subset_c0 = (
        c1_lower >= c0_lower - NESTING_TOLERANCE
        and c1_upper <= c0_upper + NESTING_TOLERANCE
    )
    final_subset_c1 = (
        final_lower >= c1_lower - NESTING_TOLERANCE
        and final_upper <= c1_upper + NESTING_TOLERANCE
    )
    if not final_subset_c1:
        raise RuntimeError(
            f"Final interval for {method} is not nested within C1 at "
            f"(X,W)=({X_OBS},{W_OBS}) beyond tolerance "
            f"{NESTING_TOLERANCE:g}."
        )
    method_summary = {
        "Method": method,
        "C0_Lower": c0_lower,
        "C0_Upper": c0_upper,
        "C1_Lower": c1_lower,
        "C1_Upper": c1_upper,
        "Final_Lower": final_lower,
        "Final_Upper": final_upper,
        "Iterations": len(stages) - 1,
        "Converged": converged,
        "Last_Target_Endpoint_Change": deltas[-1] if deltas else 0.0,
        "Convergence_Scope": "observed cell (X,W)=(0,13)",
        "C1_Subset_C0_At_Observed_Cell": c1_subset_c0,
        "Final_Subset_C1_At_Observed_Cell": final_subset_c1,
        "LUT_X_Max": x_max,
        "LUT_W_Max": w_max,
        "Accepted_Support_Attempt": accepted_attempt,
        "Accepted_Attempt_Seconds": elapsed_s,
        "Total_Seconds_Including_Support_Retries": time.perf_counter() - method_start,
        "Max_Endpoint_Change_By_Iteration": deltas,
    }
    return result_rows, support_rows, attempt_rows, method_summary


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    core = load_core()
    runtime = build_runtime(core)

    eps_lower_target, eps_upper_target = core.compute_berger_boos_bounds(
        core.torch.tensor([float(W_OBS)], device=core.device, dtype=core.dtype),
        M, BETA,
    )
    eps_lower = float(eps_lower_target[0].item())
    eps_upper = float(eps_upper_target[0].item())

    manifest = build_manifest(
        core, runtime, args.methods, eps_lower, eps_upper
    )
    write_json_atomic(output_dir / "experiment_manifest.json", manifest)

    result_rows: list[dict[str, Any]] = []
    support_rows: list[dict[str, Any]] = []
    attempt_rows: list[dict[str, Any]] = []
    method_summaries: list[dict[str, Any]] = []

    for method in args.methods:
        method_rows, method_support, method_attempts, method_summary = run_method(
            core, runtime, method
        )
        result_rows.extend(method_rows)
        support_rows.extend(method_support)
        attempt_rows.extend(method_attempts)
        method_summaries.append(method_summary)
        manifest["completed_methods"].append(method)
        manifest["method_summaries"] = method_summaries
        manifest["updated_at_utc"] = utc_now()
        write_json_atomic(output_dir / "experiment_manifest.json", manifest)

        write_csv(output_dir / "intervals.csv", result_rows, INTERVAL_FIELDS)
        write_csv(output_dir / "support_audit.csv", support_rows, SUPPORT_AUDIT_FIELDS)
        write_csv(output_dir / "support_attempts.csv", attempt_rows, SUPPORT_ATTEMPT_FIELDS)
        print(f"[{method}] result rows and support audit saved in {output_dir}", flush=True)

    manifest["status"] = "complete"
    manifest["completed_at_utc"] = utc_now()
    manifest["method_summaries"] = method_summaries
    write_json_atomic(output_dir / "experiment_manifest.json", manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
