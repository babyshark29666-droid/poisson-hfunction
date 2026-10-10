"""KARMEN2 confidence intervals with an unknown Poisson background.

Build Bayesian, IM, or NIM initial intervals and optionally refine them with
the dynamically truncated, Berger–Boos corrected h-function. The configuration
in ``__main__`` selects the methods and number of refinement iterations.
CSV files are written to the current working directory.

Author: Shuyuan Zhao
"""

import csv
import gc
import math
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm

# Device and precision used by all numerical routines.

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float32
torch.set_default_dtype(dtype)

print("=" * 60)
print("KARMEN2 empirical analysis")
print(f"Device: {device}")
print(f"Precision: {dtype}")
print("=" * 60)

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

def _get_chunk_workspace_budget_bytes() -> int:
    """Choose a workspace budget for batched h-function calculations."""
    if not torch.cuda.is_available():
        return 256 * 1024 * 1024
    env_mb = os.environ.get("HFUNC_CHUNK_MB")
    if env_mb:
        try:
            return max(int(float(env_mb) * (1024 ** 2)), 64 * 1024 * 1024)
        except ValueError:
            pass
    total_mem = torch.cuda.get_device_properties(device).total_memory
    return int(min(total_mem * 0.08, 2 * 1024 ** 3))


_CHUNK_WORKSPACE_BUDGET_BYTES: int = _get_chunk_workspace_budget_bytes()
_DTYPE_BYTES: int = torch.tensor([], dtype=dtype).element_size()


@dataclass
class EmpiricalConfig:
    """Inputs, numerical settings, and output options for the KARMEN2 run."""
    methods_to_run: List[str] = field(default_factory=lambda: ["Bayes", "NIM", "IM"])
    target_cis: List[float] = field(default_factory=lambda: [0.90])
    lut_modes: List[str] = field(default_factory=lambda: ["karmen2_optimized"])

    N: int = 10000

    chunk_size: int = 500
    h_chunk_size: int = 500
    lut_chunk_size: int = 500

    use_iterative_refinement: bool = True
    max_iter: int = 30
    convergence_tol: float = 1e-4

    berger_boos_beta: float = 0.0001
    truncation_eta: float = 1e-9
    num_b_grid: int = 1000
    num_lambda_grid: int = 500
    random_seed: int = 27
    verbose: bool = True

    karmen2_x_obs: int = 15
    karmen2_b_est: float = 15.8
    karmen2_sigma_b: float = 0.5
    karmen2_y_obs_override: Optional[int] = None

    export_lut_tables: bool = True
    export_summary_csv: bool = True


@dataclass
class Karmen2IntervalResult:
    """Observed-data C0, C1, and final intervals with run metadata."""
    method: str
    target_ci: float
    lut_mode: str
    x_obs: int
    y_obs: int
    m_param: float
    b_est: float
    sigma_b: float
    l_c0: float
    u_c0: float
    len_c0: float
    l_c1: float
    u_c1: float
    len_c1: float
    l_final: float
    u_final: float
    len_final: float
    iterations: int
    x_max: int
    w_min: int
    w_max: int
    timestamp: str


# Distribution functions and quantiles used by the initial intervals.

@torch.jit.script
def vector_gammainc_safe(shape: torch.Tensor, val: torch.Tensor) -> torch.Tensor:
    """Evaluate regularized lower incomplete gamma, including zero shapes."""
    zero_shape_mask = shape == 0
    safe_shape = torch.where(zero_shape_mask, torch.ones_like(shape), shape)
    res = torch.special.gammainc(safe_shape, val)
    if res.shape != zero_shape_mask.shape:
        zero_shape_mask = zero_shape_mask.expand_as(res)
    return torch.where(zero_shape_mask, torch.ones_like(res), res)


@torch.jit.script
def custom_gamma_icdf_gpu(p: torch.Tensor, k: torch.Tensor, num_iters: int = 5) -> torch.Tensor:
    """Approximate gamma quantiles with an initial guess and Newton updates."""
    eps = 1e-9
    p = torch.clamp(p, eps, 1.0 - eps)
    two = torch.tensor(2.0, device=p.device, dtype=p.dtype)
    z = torch.sqrt(two) * torch.erfinv(2 * p - 1)
    term1 = 1.0 - 1.0 / (9.0 * k)
    term2 = z * torch.sqrt(1.0 / (9.0 * k))
    x_wh = k * torch.pow(term1 + term2, 3)
    x_small = torch.exp((torch.log(p) + torch.lgamma(k + 1)) / k)
    x = torch.where(k > 1.0, x_wh, x_small)
    x = torch.clamp(x, min=eps)
    for _ in range(num_iters):
        cdf = torch.special.gammainc(k, x)
        log_pdf = (k - 1) * torch.log(x) - x - torch.lgamma(k)
        pdf = torch.exp(log_pdf)
        delta = (cdf - p) / (pdf + 1e-10)
        x = x - delta
        x = torch.clamp(x, min=eps)
    return x


@torch.jit.script
def safe_gamma_icdf(value: torch.Tensor, concentration: torch.Tensor, rate: float = 1.0) -> torch.Tensor:
    """Return gamma quantiles, setting nonpositive shape entries to zero."""
    safe_conc = torch.where(concentration > 0, concentration, torch.ones_like(concentration))
    std_icdf = custom_gamma_icdf_gpu(value, safe_conc, 5)
    computed_values = std_icdf / rate
    return torch.where(concentration > 0, computed_values, torch.zeros_like(computed_values))


def get_poisson_quantile(p_fail: float, lam: float) -> int:
    """Estimate an upper Poisson cutoff at probability ``1 - p_fail``."""
    if lam <= 0:
        return 0
    prob = max(min(1.0 - p_fail, 1.0 - 1e-9), 1e-9)
    mu = torch.tensor([lam], device=device, dtype=dtype)
    q = poisson_ppf_tensor(prob, mu)[0].item()
    if not math.isfinite(q):
        q = lam + 10.0 * math.sqrt(max(lam, 0.0) + 1.0) + 10.0
    return int(max(q, 0.0))



def get_poisson_lower_quantile(prob: float, lam: float) -> int:
    """Estimate a lower Poisson cutoff at probability ``prob``."""
    if lam <= 0:
        return 0
    p = max(min(prob, 1.0 - 1e-9), 1e-9)
    mu = torch.tensor([lam], device=device, dtype=dtype)
    q = poisson_ppf_tensor(p, mu)[0].item()
    if not math.isfinite(q):
        q = 0.0
    return int(max(q, 0.0))


@torch.jit.script
def _norm_ppf_torch(prob: torch.Tensor) -> torch.Tensor:
    eps = 1e-7
    p = torch.clamp(prob, eps, 1.0 - eps)
    return torch.sqrt(torch.tensor(2.0, device=p.device, dtype=p.dtype)) * torch.erfinv(2.0 * p - 1.0)



def poisson_ppf_tensor(prob: float, mu: torch.Tensor) -> torch.Tensor:
    """Approximate Poisson quantiles and correct them using the CDF."""
    mu_safe = torch.clamp(mu, min=0.0)
    p = torch.full_like(mu_safe, float(prob))
    z = _norm_ppf_torch(p)
    z = torch.nan_to_num(z, nan=0.0, posinf=8.0, neginf=-8.0)

    approx = mu_safe + z * torch.sqrt(mu_safe + 0.25) + (z * z - 1.0) / 6.0
    approx = torch.nan_to_num(approx, nan=0.0, posinf=1e8, neginf=0.0)
    q = torch.floor(torch.clamp(approx, min=0.0))

    for _ in range(3):
        shape = q + 1.0
        cdf_q = torch.special.gammaincc(shape, mu_safe)
        q = torch.where(cdf_q < p, q + 1.0, q)
        q_minus = torch.clamp(q - 1.0, min=0.0)
        cdf_minus = torch.special.gammaincc(q_minus + 1.0, mu_safe)
        q = torch.where((q > 0.0) & (cdf_minus >= p), q_minus, q)

    q = torch.nan_to_num(q, nan=0.0, posinf=1e8, neginf=0.0)
    return q.to(dtype=mu.dtype)



def chi2_ppf_tensor(prob: torch.Tensor, df: torch.Tensor) -> torch.Tensor:
    """Approximate chi-square quantiles through the gamma quantile routine."""
    p = torch.clamp(prob, min=1e-9, max=1.0 - 1e-9)
    d = torch.clamp(df, min=0.0)
    shape = torch.clamp(d / 2.0, min=1e-9)
    return 2.0 * custom_gamma_icdf_gpu(p, shape, 6)



def build_lambda_grid(lower: torch.Tensor, upper: torch.Tensor, num_points: int) -> torch.Tensor:
    """Create a separate evenly spaced lambda grid for each interval."""
    t = torch.linspace(0.0, 1.0, num_points, device=lower.device, dtype=lower.dtype)
    return lower.unsqueeze(1) + (upper - lower).unsqueeze(1) * t.unsqueeze(0)


# Bayesian, IM, and NIM initial interval calculations.

@torch.jit.script
def f1_batch_nim(theta: torch.Tensor, x: torch.Tensor, a1: torch.Tensor, u1: torch.Tensor) -> torch.Tensor:
    """Evaluate the first randomized NIM root equation in batches."""
    p1 = vector_gammainc_safe(x, theta)
    p2 = vector_gammainc_safe(x + 1, theta)
    return a1 * p1 + (1.0 - a1) * p2 - u1


@torch.jit.script
def f2_batch_nim(e: torch.Tensor, w: torch.Tensor, a2: torch.Tensor, u2: torch.Tensor) -> torch.Tensor:
    """Evaluate the background randomized NIM root equation in batches."""
    p1 = vector_gammainc_safe(w, e)
    p2 = vector_gammainc_safe(w + 1, e)
    return a2 * p1 + (1.0 - a2) * p2 - u2



def solve_roots_bisection_optimized(func, target_args, lower: float = 0.0, upper: float = 5000.0, iterations: int = 30):
    """Solve batched increasing root equations by fixed-step bisection."""
    batch_shape = target_args[0].shape
    low = torch.full(batch_shape, lower, device=device, dtype=dtype)
    high = torch.full(batch_shape, upper, device=device, dtype=dtype)
    for _ in range(iterations):
        mid = (low + high) / 2
        val = func(mid, *target_args)
        mask = val > 0
        high = torch.where(mask, mid, high)
        low = torch.where(~mask, mid, low)
    return (low + high) / 2



def calculate_nim_intervals(X_vec: torch.Tensor, W_vec: torch.Tensor, m_param: float, alpha: float, N: int):
    """Form NIM intervals from randomized root-difference quantiles."""
    batch_size = X_vec.shape[0]
    X_exp = X_vec.unsqueeze(1)
    W_exp = W_vec.unsqueeze(1)
    a1 = torch.rand((batch_size, N), device=device, dtype=dtype)
    a2 = torch.rand((batch_size, N), device=device, dtype=dtype)
    u1 = torch.rand((batch_size, N), device=device, dtype=dtype)
    u2 = torch.rand((batch_size, N), device=device, dtype=dtype)
    root1 = solve_roots_bisection_optimized(f1_batch_nim, (X_exp, a1, u1), iterations=30)
    root2 = solve_roots_bisection_optimized(f2_batch_nim, (W_exp, a2, u2), iterations=30)
    r = root1 - root2 / m_param
    q_lower = alpha / 2.0
    q_upper = 1.0 - alpha / 2.0
    nim1 = torch.quantile(r, q_lower, dim=1)
    nim2 = torch.quantile(r, q_upper, dim=1)
    return torch.clamp(nim1, min=0), torch.clamp(nim2, min=0)



def calculate_im_intervals(X_vec: torch.Tensor, W_vec: torch.Tensor, m_param: float, alpha: float, N: int):
    """Form IM intervals from sampled gamma quantile differences."""
    batch_size = X_vec.shape[0]
    X_exp = X_vec.unsqueeze(1)
    W_exp = W_vec.unsqueeze(1)
    u = torch.rand((batch_size, N), device=device, dtype=dtype)
    v = torch.rand((batch_size, N), device=device, dtype=dtype)
    term1_a1 = safe_gamma_icdf(u, X_exp, rate=1.0)
    term2_a1 = safe_gamma_icdf(v, W_exp + 1, rate=1.0)
    a1 = term1_a1 - term2_a1 / m_param
    term1_a2 = safe_gamma_icdf(u, X_exp + 1, rate=1.0)
    term2_a2 = safe_gamma_icdf(v, W_exp, rate=1.0)
    a2 = term1_a2 - term2_a2 / m_param
    q_lower = alpha / 2.0
    q_upper = 1.0 - alpha / 2.0
    f1 = torch.quantile(a1, q_lower, dim=1)
    f2 = torch.quantile(a2, q_upper, dim=1)
    return torch.clamp(f1, min=0), torch.clamp(f2, min=0)


@torch.no_grad()
def calculate_bayes_hpd_core(x_vec: torch.Tensor, w_vec: torch.Tensor, m_param: float, target_ci: float, chunk_size: int = 200):
    """Find Bayesian highest posterior density intervals on a lambda grid."""
    M = x_vec.shape[0]
    a, b = 1.0, 1.0
    grid_start, grid_end, grid_step = 0.0, 50.0, 0.005
    grid_size = int((grid_end - grid_start) / grid_step) + 1
    grid_tensor = torch.linspace(grid_start, grid_end, steps=grid_size, device=device, dtype=dtype)
    L_out = torch.zeros(M, device=device, dtype=dtype)
    U_out = torch.zeros(M, device=device, dtype=dtype)

    for i in range(0, M, chunk_size):
        end = min(i + chunk_size, M)
        curr_bs = end - i
        x = x_vec[i:end]
        w = w_vec[i:end]

        max_x = int(x.max().item())
        k_indices = torch.arange(max_x + 1, device=device, dtype=dtype).unsqueeze(0)
        x_expanded = x.unsqueeze(1)
        mask = k_indices <= x_expanded
        w_expanded = w.unsqueeze(1)

        term_a = w_expanded * math.log(m_param)
        term_b = torch.lgamma(a + w_expanded + k_indices)
        term_c = torch.lgamma(w_expanded + 1.0)
        term_d = torch.lgamma(k_indices + 1.0)
        term_e = (a + w_expanded + k_indices) * math.log(b + m_param + 1.0)

        log_common = term_a + term_b - term_c - term_d - term_e
        log_common_masked = log_common.clone()
        log_common_masked[~mask] = -float("inf")
        log_pwx = torch.logsumexp(log_common_masked, dim=1, keepdim=True)
        log_weights = log_common_masked - log_pwx

        shape_param = x_expanded - k_indices + 1
        shape_safe = torch.clamp(shape_param, min=1.0)
        shape_safe = shape_safe.unsqueeze(1)

        def calculate_G(lam_vals: torch.Tensor):
            lam_bc = lam_vals.unsqueeze(2)
            p_val = torch.special.gammainc(shape_safe, lam_bc)
            weights_bc = log_weights.unsqueeze(1)
            return torch.sum(torch.exp(weights_bc) * p_val, dim=2)

        def calculate_g(lam_vals: torch.Tensor):
            lam_bc = lam_vals.unsqueeze(2)
            lam_safe = torch.clamp(lam_bc, min=1e-10)
            pow_term = (x_expanded.unsqueeze(1) - k_indices.unsqueeze(1)) * torch.log(lam_safe)
            exp_term = -lam_bc
            val_for_lgamma = x_expanded.unsqueeze(1) - k_indices.unsqueeze(1) + 1.0
            val_for_lgamma_safe = torch.clamp(val_for_lgamma, min=1.0)
            fact_term = -torch.lgamma(val_for_lgamma_safe)
            w_term = log_weights.unsqueeze(1)
            log_terms_k = pow_term + exp_term + fact_term + w_term
            return torch.exp(torch.logsumexp(log_terms_k, dim=2))

        target_G = torch.full((curr_bs, 1), target_ci, device=device, dtype=dtype)
        l = torch.full((curr_bs, 1), 1e-6, device=device, dtype=dtype)
        u = torch.full((curr_bs, 1), 1000.0, device=device, dtype=dtype)
        for _ in range(30):
            mid = (l + u) / 2.0
            vals = calculate_G(mid)
            cond = vals < target_G
            l = torch.where(cond, mid, l)
            u = torch.where(cond, u, mid)
        z = (l + u) / 2.0

        best_vals = torch.full((curr_bs,), -1.0, device=device, dtype=dtype)
        best_indices = torch.zeros((curr_bs,), dtype=torch.long, device=device)
        grid_chunk_size = 5000
        g_chunks = int(math.ceil(grid_size / grid_chunk_size))
        for gc_idx in range(g_chunks):
            gst = gc_idx * grid_chunk_size
            ged = min((gc_idx + 1) * grid_chunk_size, grid_size)
            sub_grid = grid_tensor[gst:ged].unsqueeze(0).expand(curr_bs, -1)
            g_vals = calculate_g(sub_grid)
            max_v, max_i = torch.max(g_vals, dim=1)
            update = max_v > best_vals
            best_vals[update] = max_v[update]
            best_indices[update] = max_i[update] + gst
        mod = grid_tensor[best_indices].unsqueeze(1)

        g_0 = calculate_g(torch.zeros((curr_bs, 1), device=device, dtype=dtype))
        g_z = calculate_g(z)
        l_final = torch.zeros((curr_bs, 1), device=device, dtype=dtype)
        u_final = torch.zeros((curr_bs, 1), device=device, dtype=dtype)

        cond1 = (g_0 >= g_z).squeeze()
        l_final[cond1] = 0.0
        u_final[cond1] = z[cond1]
        idx_complex = ~cond1
        if idx_complex.any():
            curr_c0 = g_0.clone()
            curr_c1 = g_z.clone()
            for _ in range(25):
                c_mid = (curr_c0 + curr_c1) / 2.0
                ly, uy = torch.zeros((curr_bs, 1), device=device, dtype=dtype), mod.clone()
                for _ in range(20):
                    my = (ly + uy) / 2.0
                    gy = calculate_g(my)
                    too_small = gy < c_mid
                    ly = torch.where(too_small, my, ly)
                    uy = torch.where(too_small, uy, my)
                y_res = (ly + uy) / 2.0
                lz, uz = mod.clone(), torch.full((curr_bs, 1), 1000.0, device=device, dtype=dtype)
                for _ in range(20):
                    mz = (lz + uz) / 2.0
                    gz = calculate_g(mz)
                    too_small_z = gz < c_mid
                    uz = torch.where(too_small_z, mz, uz)
                    lz = torch.where(too_small_z, lz, mz)
                zz_res = (lz + uz) / 2.0
                mass = calculate_G(zz_res) - calculate_G(y_res)
                diff = mass - target_ci
                mass_too_small = diff <= 0
                update = idx_complex.unsqueeze(1)
                curr_c1 = torch.where(update & mass_too_small, c_mid, curr_c1)
                curr_c0 = torch.where(update & (~mass_too_small), c_mid, curr_c0)
                l_final = torch.where(update, y_res, l_final)
                u_final = torch.where(update, zz_res, u_final)

        L_out[i:end] = l_final.squeeze()
        U_out[i:end] = u_final.squeeze()

    return L_out, U_out


# Berger–Boos nuisance bounds and dynamic support limits.

def compute_berger_boos_bounds(w_obs: torch.Tensor, m: float, beta: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return chi-square confidence bounds for the background rate."""
    beta_half = torch.full_like(w_obs, beta / 2.0)
    one_minus_beta_half = torch.full_like(w_obs, 1.0 - beta / 2.0)
    df_min = 2.0 * w_obs
    df_max = 2.0 * (w_obs + 1.0)
    chi2_min = chi2_ppf_tensor(beta_half, df_min)
    chi2_max = chi2_ppf_tensor(one_minus_beta_half, df_max)
    b_min = torch.where(w_obs <= 0, torch.zeros_like(w_obs), chi2_min / (2.0 * m))
    b_max = chi2_max / (2.0 * m)
    return b_min, b_max



def compute_dynamic_lambda_upper(interval_upper: torch.Tensor) -> torch.Tensor:
    """Extend each interval's upper endpoint for lambda grid evaluation."""
    return torch.clamp(interval_upper + 3.0 * torch.sqrt(torch.clamp(interval_upper, min=0.0)), min=0.0)



def compute_dynamic_sample_limits(
    w_obs: torch.Tensor,
    interval_upper: torch.Tensor,
    m: float,
    beta: float,
    eta: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Set Poisson X and W support cutoffs from nuisance and lambda bounds."""
    _, eps_max = compute_berger_boos_bounds(w_obs, m, beta)
    lam_max = compute_dynamic_lambda_upper(interval_upper)
    mu_x_max = lam_max + eps_max
    mu_w_max = m * eps_max
    x_cut = poisson_ppf_tensor(1.0 - eta, mu_x_max)
    w_cut = poisson_ppf_tensor(1.0 - eta, mu_w_max)
    return x_cut, w_cut


class HFunctionCalculator:
    """Evaluate and invert the Berger–Boos corrected h-function."""

    def __init__(
        self,
        m: float,
        bb_beta: float,
        eta: float,
        num_b_grid: int = 1000,
        num_lambda_grid: int = 500,
    ):
        self.m = m
        self.beta = bb_beta
        self.eta = eta
        self.num_b_grid = num_b_grid
        self.num_lambda_grid = num_lambda_grid
        self.lambda_eval_chunk_size = 16
        self.L0_grid: Optional[torch.Tensor] = None
        self.U0_grid: Optional[torch.Tensor] = None
        self.x_max = 0
        self.w_max = 0
        self.w_offset = 0

        self._lgamma_w_cache: Optional[torch.Tensor] = None
        self._lgamma_x_cache: Optional[torch.Tensor] = None
        self._w_values: Optional[torch.Tensor] = None
        self._x_range_cache: Optional[torch.Tensor] = None
        self._b_interp_t_cache: Optional[torch.Tensor] = None

    def set_lut(self, L0: torch.Tensor, U0: torch.Tensor, w_offset: int = 0):
        """Attach an interval lookup table and cache its support terms."""
        self.L0_grid = L0.contiguous()
        self.U0_grid = U0.contiguous()
        self.x_max = L0.shape[0] - 1
        self.w_max = L0.shape[1] - 1
        self.w_offset = w_offset

        self._w_values = torch.arange(
            self.w_offset,
            self.w_offset + self.w_max + 1,
            device=device,
            dtype=dtype,
        )
        self._lgamma_w_cache = torch.lgamma(self._w_values + 1.0)

        x_range = torch.arange(self.x_max + 1, device=device, dtype=dtype)
        self._lgamma_x_cache = torch.lgamma(x_range + 1.0)
        self._x_range_cache = x_range

    def prepare_b_grid_and_pw(self, w_obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Prepare nuisance-rate grids and Poisson W probabilities."""
        if self._w_values is None or self._lgamma_w_cache is None:
            raise RuntimeError("set_lut must be called before preparing Berger-Boos grids.")

        batch_size = w_obs.shape[0]
        b_min, b_max = compute_berger_boos_bounds(w_obs, self.m, self.beta)
        if (
            self._b_interp_t_cache is None
            or self._b_interp_t_cache.numel() != self.num_b_grid
            or self._b_interp_t_cache.device != device
            or self._b_interp_t_cache.dtype != dtype
        ):
            self._b_interp_t_cache = torch.linspace(0.0, 1.0, self.num_b_grid, device=device, dtype=dtype)
        t = self._b_interp_t_cache.unsqueeze(0).expand(batch_size, -1)
        b_grid = b_min.unsqueeze(1) + (b_max - b_min).unsqueeze(1) * t
        mb = self.m * b_grid

        log_w = self._w_values.view(1, 1, -1)
        lgamma_w = self._lgamma_w_cache.view(1, 1, -1)
        mb_exp = mb.unsqueeze(2)
        log_pw = log_w * torch.log(mb_exp + 1e-12) - mb_exp - lgamma_w
        pw = torch.exp(log_pw)
        return b_grid, pw

    def compute_h_with_precalc(
        self,
        x_obs: torch.Tensor,
        w_obs: torch.Tensor,
        lambda0: torch.Tensor,
        b_grid: torch.Tensor,
        pw: torch.Tensor,
        x_cut: torch.Tensor,
        w_cut: torch.Tensor,
    ) -> torch.Tensor:
        """Maximize the truncated h-function tail probability over backgrounds."""
        if (
            self.L0_grid is None
            or self.U0_grid is None
            or self._w_values is None
            or self._lgamma_x_cache is None
            or self._x_range_cache is None
        ):
            raise RuntimeError("set_lut must be called before computing h-function values.")

        batch_size = x_obs.shape[0]
        num_b = b_grid.shape[1]

        idx_x = torch.clamp(x_obs.long(), 0, self.x_max)
        idx_w = torch.clamp(w_obs.long() - self.w_offset, 0, self.w_max)
        L0_obs = self.L0_grid[idx_x, idx_w]
        U0_obs = self.U0_grid[idx_x, idx_w]
        T2_obs = torch.minimum(lambda0 - L0_obs, U0_obs - lambda0)

        if b_grid.shape[0] == 1 and batch_size > 1:
            b_grid = b_grid.expand(batch_size, -1)
        elif b_grid.shape[0] != batch_size:
            raise RuntimeError(
                f"b_grid leading dimension ({b_grid.shape[0]}) is incompatible with batch_size ({batch_size})."
            )
        if pw.shape[0] == 1 and batch_size > 1:
            pw = pw.expand(batch_size, -1, -1)
        elif pw.shape[0] != batch_size:
            raise RuntimeError(
                f"pw leading dimension ({pw.shape[0]}) is incompatible with batch_size ({batch_size})."
            )

        x_range = self._x_range_cache
        lambda0_col = lambda0.view(batch_size, 1)
        lambda0_exp = lambda0.view(batch_size, 1, 1)
        T2_obs_exp = T2_obs.view(batch_size, 1, 1)

        x_valid = x_range.view(1, -1) <= x_cut.view(-1, 1)
        w_valid = self._w_values.view(1, -1) <= w_cut.view(-1, 1)

        num_x = self.x_max + 1
        num_w = self.w_max + 1

        base_w_block = int(max(16, min(num_w, 256)))
        bytes_per_w = max(batch_size * num_x * _DTYPE_BYTES * 4, 1)
        budget_w_block = int(max(16, min(num_w, _CHUNK_WORKSPACE_BUDGET_BYTES // bytes_per_w)))
        w_block_size = min(base_w_block, budget_w_block)

        base_b_block = int(max(32, min(num_b, 256)))
        bytes_per_b = max(batch_size * num_x * _DTYPE_BYTES * 3, 1)
        budget_b_block = int(max(32, min(num_b, _CHUNK_WORKSPACE_BUDGET_BYTES // bytes_per_b)))
        b_block_size = min(base_b_block, budget_b_block)

        prob_sum = torch.zeros((batch_size, num_b), device=device, dtype=dtype)
        for b_start in range(0, num_b, b_block_size):
            b_end = min(b_start + b_block_size, num_b)
            b_grid_sub = b_grid[:, b_start:b_end]
            theta_sub = lambda0_col + b_grid_sub
            theta_sub_exp = theta_sub.unsqueeze(2)
            log_px_sub = (
                x_range.view(1, 1, -1) * torch.log(theta_sub_exp + 1e-12)
                - theta_sub_exp
                - self._lgamma_x_cache.view(1, 1, -1)
            )
            px_sub = torch.exp(log_px_sub)
            prob_sum_sub = torch.zeros((batch_size, b_end - b_start), device=device, dtype=dtype)
            pw_sub_all = pw[:, b_start:b_end, :]

            for w_start in range(0, self.w_max + 1, w_block_size):
                w_end = min(w_start + w_block_size, self.w_max + 1)
                L0_sub = self.L0_grid[:, w_start:w_end]
                U0_sub = self.U0_grid[:, w_start:w_end]
                T2_sub = torch.minimum(
                    lambda0_exp - L0_sub.unsqueeze(0),
                    U0_sub.unsqueeze(0) - lambda0_exp,
                )
                w_valid_sub = w_valid[:, w_start:w_end]
                trunc_mask_sub = x_valid.unsqueeze(2) & w_valid_sub.unsqueeze(1)
                mask_sub = ((T2_sub <= T2_obs_exp) & trunc_mask_sub).to(dtype)
                pw_sub = pw_sub_all[:, :, w_start:w_end].permute(0, 2, 1).contiguous()
                temp_sum_w = torch.bmm(mask_sub.contiguous(), pw_sub)
                prob_sum_sub += torch.sum(px_sub * temp_sum_w.permute(0, 2, 1), dim=2)

            prob_sum[:, b_start:b_end] = prob_sum_sub

        h_sup, _ = prob_sum.max(dim=1)
        return torch.clamp(h_sup + self.beta, max=1.0)

    def refine_intervals_with_precalc_input(
        self,
        x_obs: torch.Tensor,
        w_obs: torch.Tensor,
        alpha: float,
        L_init: torch.Tensor,
        U_init: torch.Tensor,
        b_grid: torch.Tensor,
        pw: torch.Tensor,
        is_first_iter: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Invert the h-function to refine a batch of interval endpoints."""
        threshold = alpha - self.eta
        x_cut, w_cut = compute_dynamic_sample_limits(w_obs, U_init, self.m, self.beta, self.eta)

        if is_first_iter:
            grid_low = torch.zeros_like(L_init)
            grid_high = compute_dynamic_lambda_upper(U_init)
        else:
            grid_low = torch.clamp(L_init, min=0.0)
            grid_high = torch.max(U_init, grid_low)

        lambda_grid = build_lambda_grid(grid_low, grid_high, self.num_lambda_grid)
        h_grid = torch.empty_like(lambda_grid)

        est_bytes_per_lambda = max(x_obs.shape[0] * (self.x_max + 1) * _DTYPE_BYTES * 6, 1)
        max_chunk_by_budget = max(1, _CHUNK_WORKSPACE_BUDGET_BYTES // est_bytes_per_lambda)
        effective_lambda_chunk = int(max(1, min(self.lambda_eval_chunk_size, max_chunk_by_budget, self.num_lambda_grid)))

        for j_start in range(0, self.num_lambda_grid, effective_lambda_chunk):
            j_end = min(j_start + effective_lambda_chunk, self.num_lambda_grid)
            lam_chunk = lambda_grid[:, j_start:j_end]
            chunk_size = j_end - j_start

            if chunk_size == 1:
                h_grid[:, j_start] = self.compute_h_with_precalc(
                    x_obs,
                    w_obs,
                    lam_chunk[:, 0],
                    b_grid,
                    pw,
                    x_cut,
                    w_cut,
                )
                continue

            x_rep = x_obs.unsqueeze(1).expand(-1, chunk_size).reshape(-1)
            w_rep = w_obs.unsqueeze(1).expand(-1, chunk_size).reshape(-1)
            x_cut_rep = x_cut.unsqueeze(1).expand(-1, chunk_size).reshape(-1)
            w_cut_rep = w_cut.unsqueeze(1).expand(-1, chunk_size).reshape(-1)
            lam_flat = lam_chunk.reshape(-1)

            h_flat = self.compute_h_with_precalc(
                x_rep,
                w_rep,
                lam_flat,
                b_grid,
                pw,
                x_cut_rep,
                w_cut_rep,
            )
            h_grid[:, j_start:j_end] = h_flat.view(-1, chunk_size)

        inside = h_grid > threshold
        has_inside = inside.any(dim=1)

        idx_first = torch.argmax(inside.to(torch.int64), dim=1)
        idx_last = (self.num_lambda_grid - 1) - torch.argmax(
            torch.flip(inside.to(torch.int64), dims=[1]),
            dim=1,
        )

        left_idx0 = torch.clamp(idx_first - 1, min=0)
        left_idx1 = idx_first
        right_idx0 = idx_last
        right_idx1 = torch.clamp(idx_last + 1, max=self.num_lambda_grid - 1)

        l_low = torch.gather(lambda_grid, 1, left_idx0.unsqueeze(1)).squeeze(1)
        l_high = torch.gather(lambda_grid, 1, left_idx1.unsqueeze(1)).squeeze(1)
        u_low = torch.gather(lambda_grid, 1, right_idx0.unsqueeze(1)).squeeze(1)
        u_high = torch.gather(lambda_grid, 1, right_idx1.unsqueeze(1)).squeeze(1)

        batch_size = x_obs.shape[0]
        x_cat = torch.cat([x_obs, x_obs], dim=0)
        w_cat = torch.cat([w_obs, w_obs], dim=0)
        x_cut_cat = torch.cat([x_cut, x_cut], dim=0)
        w_cut_cat = torch.cat([w_cut, w_cut], dim=0)
        mid_cat = torch.empty((2 * batch_size,), device=device, dtype=dtype)

        for _ in range(30):
            mid_l = (l_low + l_high) / 2.0
            mid_u = (u_low + u_high) / 2.0
            mid_cat[:batch_size] = mid_l
            mid_cat[batch_size:] = mid_u
            h_cat = self.compute_h_with_precalc(x_cat, w_cat, mid_cat, b_grid, pw, x_cut_cat, w_cut_cat)
            h_l = h_cat[:batch_size]
            h_u = h_cat[batch_size:]

            mask_l = h_l > threshold
            l_high = torch.where(mask_l, mid_l, l_high)
            l_low = torch.where(~mask_l, mid_l, l_low)

            mask_u = h_u > threshold
            u_low = torch.where(mask_u, mid_u, u_low)
            u_high = torch.where(~mask_u, mid_u, u_high)

        L_final = torch.where(has_inside, (l_low + l_high) / 2.0, L_init)
        U_final = torch.where(has_inside, (u_low + u_high) / 2.0, U_init)

        zero_h = self.compute_h_with_precalc(
            x_obs,
            w_obs,
            torch.zeros_like(L_final),
            b_grid,
            pw,
            x_cut,
            w_cut,
        )
        L_final = torch.where(zero_h > threshold, torch.zeros_like(L_final), L_final)
        return L_final, U_final


# Lookup table construction and h-function refinement.

def generate_lut(
    method_core_func,
    m_param: float,
    alpha: float,
    x_max: int,
    w_max: int,
    chunk_size: int = 2000,
    w_min: int = 0,
    **kwargs,
):
    """Evaluate an initial interval method over a rectangular X-W grid."""
    w_dim = w_max - w_min + 1
    x_range = torch.arange(x_max + 1, device=device, dtype=dtype)
    w_range = torch.arange(w_min, w_max + 1, device=device, dtype=dtype)
    X_grid, W_grid = torch.meshgrid(x_range, w_range, indexing="ij")
    X_flat, W_flat = X_grid.flatten(), W_grid.flatten()
    L_flat = torch.zeros_like(X_flat)
    U_flat = torch.zeros_like(X_flat)
    total = X_flat.shape[0]

    for i in tqdm(range(0, total, chunk_size), desc=f"Gen C0 (alpha={alpha:.2f})"):
        end = min(i + chunk_size, total)
        X_sub = X_flat[i:end]
        W_sub = W_flat[i:end]
        l_sub, u_sub = method_core_func(X_sub, W_sub, m_param, alpha, **kwargs)
        L_flat[i:end] = l_sub
        U_flat[i:end] = u_sub

    return L_flat.view(x_max + 1, w_dim), U_flat.view(x_max + 1, w_dim)



def refine_entire_lut_optimized(
    h_calc: HFunctionCalculator,
    L_prev: torch.Tensor,
    U_prev: torch.Tensor,
    alpha: float,
    h_chunk_size: int = 500,
    desc: str = "Refining",
    is_first_iter: bool = False,
    w_offset: int = 0,
):
    """Refine every interval in a lookup table, grouped by W and X."""
    x_max = L_prev.shape[0] - 1
    w_max_rel = L_prev.shape[1] - 1

    L_new = torch.zeros_like(L_prev)
    U_new = torch.zeros_like(U_prev)

    for w_idx in tqdm(range(w_max_rel + 1), desc=desc, leave=False):
        w_actual = w_idx + w_offset
        w_tensor = torch.tensor([float(w_actual)], device=device, dtype=dtype)
        b_grid_1, pw_1 = h_calc.prepare_b_grid_and_pw(w_tensor)

        for x_start in range(0, x_max + 1, h_chunk_size):
            x_end = min(x_start + h_chunk_size, x_max + 1)
            current_batch_size = x_end - x_start

            x_batch = torch.arange(x_start, x_end, device=device, dtype=dtype)
            w_batch = torch.full((current_batch_size,), float(w_actual), device=device, dtype=dtype)

            l_init_batch = L_prev[x_start:x_end, w_idx]
            u_init_batch = U_prev[x_start:x_end, w_idx]

            l_new_sub, u_new_sub = h_calc.refine_intervals_with_precalc_input(
                x_batch,
                w_batch,
                alpha,
                l_init_batch,
                u_init_batch,
                b_grid_1,
                pw_1,
                is_first_iter=is_first_iter,
            )

            L_new[x_start:x_end, w_idx] = l_new_sub
            U_new[x_start:x_end, w_idx] = u_new_sub

    return L_new, U_new



def iterative_lut_generation(
    core_func,
    h_calc: HFunctionCalculator,
    config: EmpiricalConfig,
    alpha: float,
    x_max: int,
    w_max: int,
    kwargs: Dict[str, Any],
    m_val: float,
    w_min: int = 0,
):
    """Return C0, C1, and the last lookup table, plus the iteration count."""

    L_C0, U_C0 = generate_lut(
        core_func,
        m_val,
        alpha,
        x_max,
        w_max,
        chunk_size=config.lut_chunk_size,
        w_min=w_min,
        **kwargs,
    )
    L_C0 = torch.clamp(L_C0, min=0)
    U_C0 = torch.clamp(U_C0, min=0)

    L_curr, U_curr = L_C0.clone(), U_C0.clone()
    L_C1, U_C1 = None, None
    final_iter = 0

    for k in range(1, config.max_iter + 1):
        h_calc.set_lut(L_curr, U_curr, w_offset=w_min)

        L_new, U_new = refine_entire_lut_optimized(
            h_calc,
            L_curr,
            U_curr,
            alpha,
            h_chunk_size=config.h_chunk_size,
            desc=f"Iter {k} Refinement",
            is_first_iter=(k == 1),
            w_offset=w_min,
        )

        if k >= 2:
            L_new = torch.max(L_new, L_curr)
            U_new = torch.min(U_new, U_curr)

        if k == 1:
            L_C1, U_C1 = L_new.clone(), U_new.clone()

        max_diff_L = torch.max(torch.abs(L_new - L_curr)).item()
        max_diff_U = torch.max(torch.abs(U_new - U_curr)).item()
        max_diff = max(max_diff_L, max_diff_U)
        curr_total_len = torch.sum(U_new - L_new).item()
        print(f"    Iter {k}: MaxDiff={max_diff:.6f}, TotalLen={curr_total_len:.2f}")

        prev_L, prev_U = L_curr, U_curr
        L_curr, U_curr = L_new, U_new
        del prev_L, prev_U, L_new, U_new
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        final_iter = k
        if max_diff < config.convergence_tol:
            break

    h_calc.set_lut(L_curr, U_curr, w_offset=w_min)
    if L_C1 is None:
        L_C1, U_C1 = L_C0.clone(), U_C0.clone()

    return (L_C0, U_C0), (L_C1, U_C1), (L_curr, U_curr), final_iter


# KARMEN2 observations and lookup table support selection.

def _build_karmen2_parameters(config: EmpiricalConfig) -> Tuple[float, int]:
    """Convert the background estimate and uncertainty to ``m`` and W."""
    if config.karmen2_sigma_b <= 0:
        raise ValueError("karmen2_sigma_b must be greater than 0.")

    m_param = config.karmen2_b_est / (config.karmen2_sigma_b ** 2)
    if config.karmen2_y_obs_override is not None:
        y_obs = int(config.karmen2_y_obs_override)
    else:
        y_obs = int(round(config.karmen2_b_est * m_param))
    return m_param, y_obs



def determine_karmen2_optimized_lut_limits(
    x_obs: int,
    y_obs: int,
    m_val: float,
    beta: float,
    eta: float,
) -> Tuple[int, int, int, float, float]:
    """Choose a local lookup table around the KARMEN2 observation."""
    w_obs_tensor = torch.tensor([float(y_obs)], device=device, dtype=dtype)
    b_lo, b_hi = compute_berger_boos_bounds(w_obs_tensor, m_val, beta)
    b_lo_val = float(b_lo.item())
    b_hi_val = float(b_hi.item())

    lambda_max_guess = max(20.0, float(x_obs) + 5.0)
    x_max = max(x_obs, get_poisson_quantile(eta, lambda_max_guess + b_hi_val))

    mu_w_lo = m_val * b_lo_val
    mu_w_hi = m_val * b_hi_val
    w_min = max(0, get_poisson_lower_quantile(eta, mu_w_lo))
    w_max = max(y_obs, get_poisson_quantile(eta, mu_w_hi))

    return x_max, w_min, w_max, b_lo_val, b_hi_val



def determine_full_general_lut_limits(
    x_obs: int,
    y_obs: int,
    m_val: float,
    beta: float,
    eta: float,
    core_func,
    alpha: float,
    lut_chunk_size: int,
    kwargs: Dict[str, Any],
) -> Tuple[int, int, int]:
    """Choose full-grid limits using preliminary C0 upper endpoints."""
    w_obs_tensor = torch.tensor([float(y_obs)], device=device, dtype=dtype)
    _, b_hi_obs = compute_berger_boos_bounds(w_obs_tensor, m_val, beta)
    b_hi_obs_val = float(b_hi_obs.item())

    lambda_guess = max(20.0, float(x_obs) + 5.0)
    obs_x_max = max(x_obs, get_poisson_quantile(eta, lambda_guess + b_hi_obs_val))
    obs_w_max = max(y_obs, get_poisson_quantile(eta, m_val * b_hi_obs_val))

    L_obs, U_obs = generate_lut(
        core_func,
        m_val,
        alpha,
        obs_x_max,
        obs_w_max,
        chunk_size=lut_chunk_size,
        w_min=0,
        **kwargs,
    )
    max_u0 = torch.max(U_obs).item()

    w_obs_max_tensor = torch.tensor([float(obs_w_max)], device=device, dtype=dtype)
    _, eps_max = compute_berger_boos_bounds(w_obs_max_tensor, m_val, beta)
    global_lambda_upper = max(4.0 * max_u0, lambda_guess, 0.0)
    mu_x_max = global_lambda_upper + float(eps_max.item())
    mu_w_max = m_val * float(eps_max.item())
    x_max_lut = max(obs_x_max, get_poisson_quantile(eta, mu_x_max))
    w_max_lut = max(obs_w_max, get_poisson_quantile(eta, mu_w_max))

    del L_obs, U_obs
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    return x_max_lut, 0, w_max_lut


# CSV exports for lookup tables and observed-data intervals.

def export_lut_comparison(
    L_init: torch.Tensor,
    U_init: torch.Tensor,
    L_c1: torch.Tensor,
    U_c1: torch.Tensor,
    L_final: torch.Tensor,
    U_final: torch.Tensor,
    method: str,
    target_ci: float,
    lut_mode: str,
    w_min: int,
    filename: Optional[str] = None,
):
    """Write C0, C1, and final intervals for each lookup table cell."""
    if filename is None:
        filename = f"KARMEN2-LUT-{method}-{target_ci:.2f}-{lut_mode}.csv"

    x_dim = L_init.shape[0]
    w_dim = L_init.shape[1]
    with open(filename, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        header = [
            "X",
            "W",
            "L_C0",
            "U_C0",
            "Len_C0",
            "L_C1",
            "U_C1",
            "Len_C1",
            "L_Final",
            "U_Final",
            "Len_Final",
            "L_Diff_C0_C1",
            "U_Diff_C0_C1",
            "Len_Change_C0_C1",
            "L_Diff_C1_Final",
            "U_Diff_C1_Final",
            "Len_Change_C1_Final",
            "L_Diff_Total",
            "U_Diff_Total",
            "Len_Change_Total",
            "Status",
        ]
        writer.writerow(header)

        for x in range(x_dim):
            for w_rel in range(w_dim):
                u0 = U_init[x, w_rel].item()
                if u0 < 1e-6:
                    continue

                w_actual = w_rel + w_min
                l0 = L_init[x, w_rel].item()
                l1 = L_c1[x, w_rel].item()
                u1 = U_c1[x, w_rel].item()
                lf = L_final[x, w_rel].item()
                uf = U_final[x, w_rel].item()

                len0 = u0 - l0
                len1 = u1 - l1
                lenf = uf - lf

                l_diff_01 = l1 - l0
                u_diff_01 = u1 - u0
                len_change_01 = len1 - len0

                l_diff_1f = lf - l1
                u_diff_1f = uf - u1
                len_change_1f = lenf - len1

                l_diff_total = lf - l0
                u_diff_total = uf - u0
                len_change_total = lenf - len0

                status = "Same"
                if abs(len_change_total) > 1e-4:
                    status = "SHRUNK" if len_change_total < 0 else "EXPANDED"

                writer.writerow([
                    x,
                    w_actual,
                    f"{l0:.6f}",
                    f"{u0:.6f}",
                    f"{len0:.6f}",
                    f"{l1:.6f}",
                    f"{u1:.6f}",
                    f"{len1:.6f}",
                    f"{lf:.6f}",
                    f"{uf:.6f}",
                    f"{lenf:.6f}",
                    f"{l_diff_01:.6f}",
                    f"{u_diff_01:.6f}",
                    f"{len_change_01:.6f}",
                    f"{l_diff_1f:.6f}",
                    f"{u_diff_1f:.6f}",
                    f"{len_change_1f:.6f}",
                    f"{l_diff_total:.6f}",
                    f"{u_diff_total:.6f}",
                    f"{len_change_total:.6f}",
                    status,
                ])
    print(f"[KARMEN2] Wrote lookup table: {filename}")



def save_karmen2_results_csv(results: List[Karmen2IntervalResult], filename: str = "KARMEN2-intervals.csv"):
    """Write observed-data interval summaries to CSV."""
    fieldnames = [
        "Method",
        "Target_CI",
        "LUT_Mode",
        "X_obs",
        "Y_obs",
        "m",
        "b_est",
        "sigma_b",
        "L_C0",
        "U_C0",
        "Len_C0",
        "L_C1",
        "U_C1",
        "Len_C1",
        "L_Final",
        "U_Final",
        "Len_Final",
        "Iters",
        "X_max",
        "W_min",
        "W_max",
        "Timestamp",
    ]

    with open(filename, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow(
                {
                    "Method": r.method,
                    "Target_CI": r.target_ci,
                    "LUT_Mode": r.lut_mode,
                    "X_obs": r.x_obs,
                    "Y_obs": r.y_obs,
                    "m": f"{r.m_param:.6f}",
                    "b_est": f"{r.b_est:.6f}",
                    "sigma_b": f"{r.sigma_b:.6f}",
                    "L_C0": f"{r.l_c0:.6f}",
                    "U_C0": f"{r.u_c0:.6f}",
                    "Len_C0": f"{r.len_c0:.6f}",
                    "L_C1": f"{r.l_c1:.6f}",
                    "U_C1": f"{r.u_c1:.6f}",
                    "Len_C1": f"{r.len_c1:.6f}",
                    "L_Final": f"{r.l_final:.6f}",
                    "U_Final": f"{r.u_final:.6f}",
                    "Len_Final": f"{r.len_final:.6f}",
                    "Iters": r.iterations,
                    "X_max": r.x_max,
                    "W_min": r.w_min,
                    "W_max": r.w_max,
                    "Timestamp": r.timestamp,
                }
            )
    print(f"[KARMEN2] Wrote interval summary: {filename}")


# Empirical run and selected main configuration.

def run_karmen2_application(config: EmpiricalConfig) -> List[Karmen2IntervalResult]:
    """Run configured methods and return KARMEN2 observed-data intervals."""
    torch.manual_seed(config.random_seed)
    np.random.seed(config.random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.random_seed)

    m_val, y_obs = _build_karmen2_parameters(config)
    x_obs = int(config.karmen2_x_obs)
    b_est = float(config.karmen2_b_est)

    print(f"\nKARMEN2: X={x_obs}, W={y_obs}, m={m_val:.4f}, b={b_est}, sigma_b={config.karmen2_sigma_b}")
    print(f"Methods={config.methods_to_run}, confidence levels={config.target_cis}, max_iter={config.max_iter}")

    all_results: List[Karmen2IntervalResult] = []

    for lut_mode in config.lut_modes:
        if lut_mode not in {"karmen2_optimized", "full_general"}:
            raise ValueError(f"Unknown lut_mode: {lut_mode}")

        for current_method in config.methods_to_run:
            if current_method == "Bayes":
                def bayes_adapter(x, w, m, a, **kw):
                    return calculate_bayes_hpd_core(x, w, m, 1.0 - a, chunk_size=config.chunk_size)

                core_func = bayes_adapter
                kwargs: Dict[str, Any] = {}
            elif current_method == "NIM":
                core_func = calculate_nim_intervals
                kwargs = {"N": config.N}
            elif current_method == "IM":
                core_func = calculate_im_intervals
                kwargs = {"N": config.N}
            else:
                print(f"[KARMEN2] Skipping unknown method: {current_method}")
                continue

            for target_ci in config.target_cis:
                alpha = 1.0 - target_ci
                print(f"\n[KARMEN2] LUT={lut_mode}, Method={current_method}, CI={target_ci:.3f}")

                if lut_mode == "karmen2_optimized":
                    x_max, w_min, w_max, b_lo_bb, b_hi_bb = determine_karmen2_optimized_lut_limits(
                        x_obs,
                        y_obs,
                        m_val,
                        config.berger_boos_beta,
                        config.truncation_eta,
                    )
                    print(f"    Berger-Boos background range: [{b_lo_bb:.4f}, {b_hi_bb:.4f}]")
                else:
                    x_max, w_min, w_max = determine_full_general_lut_limits(
                        x_obs,
                        y_obs,
                        m_val,
                        config.berger_boos_beta,
                        config.truncation_eta,
                        core_func,
                        alpha,
                        config.lut_chunk_size,
                        kwargs,
                    )

                print(f"    Lookup table limits: X=0..{x_max}, W={w_min}..{w_max}")

                x_idx = min(x_obs, x_max)
                w_idx_rel = y_obs - w_min
                if w_idx_rel < 0 or y_obs > w_max:
                    raise ValueError(
                        f"Observed W={y_obs} is outside the LUT range [{w_min}, {w_max}]."
                    )

                h_calc_shared = HFunctionCalculator(
                    m_val,
                    config.berger_boos_beta,
                    config.truncation_eta,
                    num_b_grid=config.num_b_grid,
                    num_lambda_grid=config.num_lambda_grid,
                )

                if config.use_iterative_refinement:
                    (l0_lut, u0_lut), (l1_lut, u1_lut), (lf_lut, uf_lut), iter_count = iterative_lut_generation(
                        core_func,
                        h_calc_shared,
                        config,
                        alpha,
                        x_max,
                        w_max,
                        kwargs,
                        m_val,
                        w_min=w_min,
                    )
                else:
                    l0_lut, u0_lut = generate_lut(
                        core_func,
                        m_val,
                        alpha,
                        x_max,
                        w_max,
                        chunk_size=config.lut_chunk_size,
                        w_min=w_min,
                        **kwargs,
                    )
                    l1_lut, u1_lut = l0_lut, u0_lut
                    lf_lut, uf_lut = l0_lut, u0_lut
                    iter_count = 0

                if config.export_lut_tables:
                    lut_filename = f"KARMEN2-LUT-{current_method}-{target_ci:.2f}-{lut_mode}.csv"
                    export_lut_comparison(
                        l0_lut,
                        u0_lut,
                        l1_lut,
                        u1_lut,
                        lf_lut,
                        uf_lut,
                        current_method,
                        target_ci,
                        lut_mode,
                        w_min,
                        filename=lut_filename,
                    )

                l0 = l0_lut[x_idx, w_idx_rel].item()
                u0 = u0_lut[x_idx, w_idx_rel].item()
                l1 = l1_lut[x_idx, w_idx_rel].item()
                u1 = u1_lut[x_idx, w_idx_rel].item()
                lf = lf_lut[x_idx, w_idx_rel].item()
                uf = uf_lut[x_idx, w_idx_rel].item()

                row = Karmen2IntervalResult(
                    method=current_method,
                    target_ci=target_ci,
                    lut_mode=lut_mode,
                    x_obs=x_obs,
                    y_obs=y_obs,
                    m_param=m_val,
                    b_est=b_est,
                    sigma_b=config.karmen2_sigma_b,
                    l_c0=l0,
                    u_c0=u0,
                    len_c0=u0 - l0,
                    l_c1=l1,
                    u_c1=u1,
                    len_c1=u1 - l1,
                    l_final=lf,
                    u_final=uf,
                    len_final=uf - lf,
                    iterations=iter_count,
                    x_max=x_max,
                    w_min=w_min,
                    w_max=w_max,
                    timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                )
                all_results.append(row)

                print(f"    C0    : [{l0:.6f}, {u0:.6f}], len={u0 - l0:.6f}")
                print(f"    C1    : [{l1:.6f}, {u1:.6f}], len={u1 - l1:.6f}")
                print(f"    Final : [{lf:.6f}, {uf:.6f}], len={uf - lf:.6f}")

                del l0_lut, u0_lut, l1_lut, u1_lut, lf_lut, uf_lut, h_calc_shared
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()

    if config.export_summary_csv:
        save_karmen2_results_csv(all_results)
    return all_results


if __name__ == "__main__":
    # To reproduce all three method columns, set methods_to_run to
    # ["Bayes", "IM", "NIM"]. Set max_iter >= 1 for C1 and further iterations.
    # With max_iter=0 below, C1 and Final are copies of C0.
    cfg = EmpiricalConfig(
        methods_to_run=["NIM"],
        target_cis=[0.90],

        # Add "full_general" to compare the alternative lookup table support.
        lut_modes=["karmen2_optimized"],

        N=10000,
        use_iterative_refinement=True,
        max_iter=0,
        convergence_tol=1e-4,
        chunk_size=1000,
        h_chunk_size=1000,
        lut_chunk_size=1000,
        num_b_grid=1000,
        num_lambda_grid=500,

        karmen2_x_obs=15,
        karmen2_b_est=15.8,
        karmen2_sigma_b=0.5,
        karmen2_y_obs_override=None,

        export_lut_tables=True,
        export_summary_csv=True,
    )

    run_karmen2_application(cfg)
