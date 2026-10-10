"""
Exact confidence intervals for Poisson parameters using the h-function
========================================================
Methods: Bayesian, IM, and NIM

Parameter settings:
 m = 1, 5, 10, 15, 20, 50, 100, 300
 b = 3
 ci = 0.90 / 0.95
 lambda = 0.0(0.1)10.0 

Author: Shuyuan Zhao
"""

import argparse
import sys
import os
import math
import time
import gc
import csv
from typing import Tuple, Optional, Dict, List, Union, Any
from dataclasses import dataclass, asdict, field
from datetime import datetime

# This script is named code.py, which otherwise shadows Python's standard
# library module of the same name while PyTorch imports pdb. Hide the script
# directory only during PyTorch's import, then restore normal local imports.
_script_dir = os.path.dirname(os.path.abspath(__file__))
_saved_sys_path = sys.path[:]
try:
    sys.path[:] = [entry for entry in sys.path
                   if os.path.abspath(entry or os.getcwd()) != _script_dir]
    import torch
finally:
    sys.path[:] = _saved_sys_path
del _script_dir, _saved_sys_path

import numpy as np
from tqdm import tqdm

# ============================================================
# Global settings
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float64
torch.set_default_dtype(dtype)

print(f"="*60)
print(f"Device: {device}")
print(f"Precision: {dtype}")
if device.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(device)}")
print(f"="*60)

if torch.cuda.is_available():
    # The simulation uses float64 throughout; TF32 does not accelerate these
    # operations and is disabled to keep the precision policy explicit.
    torch.backends.cuda.matmul.allow_tf32 = False


def _get_recommended_batch_sizes() -> Tuple[int, int, int, int, int, int]:
    """Choose conservative starting batches from the available device memory."""
    if not torch.cuda.is_available():
        return 500, 500, 500, 1, 1, 512

    total_mem = torch.cuda.get_device_properties(device).total_memory
    try:
        free_mem, _ = torch.cuda.mem_get_info(device)
    except RuntimeError:
        free_mem = total_mem

    if total_mem >= 32 * (1024 ** 3) and free_mem >= 20 * (1024 ** 3):
        # A100 40 GB and similar accelerators.
        return 1000, 250, 4000, 4, 1, 512
    if free_mem >= 12 * (1024 ** 3):
        return 750, 250, 2500, 3, 1, 512
    if free_mem >= 6 * (1024 ** 3):
        return 500, 250, 1000, 2, 1, 512
    return 250, 125, 500, 1, 1, 512

def _get_chunk_workspace_budget_bytes() -> int:
    """Return the temporary workspace budget available for chunked computation, in bytes."""
    if not torch.cuda.is_available():
        return 256 * 1024 * 1024

    try:
        free_mem, _ = torch.cuda.mem_get_info(device)
    except RuntimeError:
        free_mem = int(torch.cuda.get_device_properties(device).total_memory * 0.5)

    env_mb = os.environ.get("HFUNC_CHUNK_MB")
    if env_mb:
        try:
            requested = max(int(float(env_mb) * (1024 ** 2)), 64 * 1024 * 1024)
            return min(requested, max(64 * 1024 * 1024, int(free_mem * 0.5)))
        except ValueError:
            pass

    # Reserve most of the currently free VRAM for tensors outside the chunked
    # workspace and cap the default to avoid monopolizing a shared GPU.
    return max(64 * 1024 * 1024, min(int(free_mem * 0.35), 12 * (1024 ** 3)))

_CHUNK_WORKSPACE_BUDGET_BYTES: int = _get_chunk_workspace_budget_bytes()
print(f"Chunk workspace budget: {_CHUNK_WORKSPACE_BUDGET_BYTES / (1024 ** 3):.2f} GiB")

@dataclass
class SimulationConfig:
    """Main program configuration.

    This data class collects parameters for the paper's numerical experiments:
    1. Parameter grids and target confidence levels;
    2. Numerical controls for initial and refined intervals;
    3. IM/NIM auxiliary draw count and random seed;
    4. Berger-Boos correction, dynamic truncation, and convergence settings.
    """
    lambda_vals: List[float] = field(default_factory=lambda: [0.0]) 
    target_cis: List[float] = field(default_factory=lambda: [0.90]) 
    m_params: List[float] = field(default_factory=lambda: [100.0])
    
    # Initial interval methods to run, in order.
    methods_to_run: List[str] = field(default_factory=lambda: ['Bayes', 'NIM', 'IM'])
    
    # Monte Carlo sample size for auxiliary variables in the NIM/IM initial intervals.
    N: int = 10000                
    # True background parameter used in CP/EL Poisson probability sums.
    eb: float = 3.0             
    
    # Batch size for chunked Bayesian initial interval calculations.
    chunk_size: Optional[int] = None
    # Batch size for chunked LUT refinement.
    h_chunk_size: Optional[int] = None
    # Number of lambda search grid points evaluated in parallel.
    h_lambda_batch_size: Optional[int] = None
    # Number of observed w slices processed together in each batch.
    h_w_chunk_size: Optional[int] = None
    # Maximum number of Berger-Boos grid points per h-function block.
    h_b_block_size: Optional[int] = None
    # Chunk size for generating the initial LUT.
    lut_chunk_size: Optional[int] = None
    
    # Whether to perform the first refinement and subsequent iterations.
    use_iterative_refinement: bool = True 
    # Maximum number of refinement iterations.
    max_iter: int = 30           
    # Convergence tolerance.
    convergence_tol: float = 1e-4 
    
    # Berger-Boos penalty beta.
    berger_boos_beta: float = 0.0001
    # Endpoint tolerance when checking whether C1 is contained in the initial interval C0.
    nesting_tolerance: float = 1e-4
    # Allowed tail error eta for dynamic truncation.
    truncation_eta: float = 1e-9  
    # Number of grid points in the nuisance parameter confidence set.
    num_b_grid: int = 1000
    # Number of grid points in the lambda search range.
    num_lambda_grid: int = 500
    # Program-wide random seed for reproducibility.
    random_seed: int = 2026
    # Whether to print detailed progress messages.
    verbose: bool = True

    def __post_init__(self):
        recommended = _get_recommended_batch_sizes()
        if self.chunk_size is None:
            self.chunk_size = recommended[0]
        if self.h_chunk_size is None:
            self.h_chunk_size = recommended[1]
        if self.lut_chunk_size is None:
            self.lut_chunk_size = recommended[2]
        if self.h_lambda_batch_size is None:
            self.h_lambda_batch_size = recommended[3]
        if self.h_w_chunk_size is None:
            self.h_w_chunk_size = recommended[4]
        if self.h_b_block_size is None:
            self.h_b_block_size = recommended[5]


@dataclass 
class SimulationResult:
    """Results for one parameter setting.

    Each record corresponds to a fixed (method, m, lambda, target_ci, beta)
    combination and stores performance metrics for C0, C1, and the final interval.
    """
    method: str
    lambda_val: float
    m_param: float
    eb: float
    target_ci: float  
    beta: float
    cp_0: float
    el_0: float
    ci_0_example: str
    cp_1: float
    el_1: float
    ci_1_example: str
    cp_final: float
    el_final: float
    ci_final_example: str
    iterations: int
    time_elapsed: float
    timestamp: str
    # Per (method, m, target CI) group; includes LUT bounds, construction,
    # refinement, and LUT comparison export.
    lut_time_elapsed: float = 0.0
    # Group compute time through CP/EL evaluation; excludes result CSV writing.
    combination_time_elapsed: float = 0.0


# ============================================================
# Module 1: Numerical utilities
# ============================================================

@torch.jit.script
def vector_gammainc_safe(shape: torch.Tensor, val: torch.Tensor) -> torch.Tensor:
    """Regularized incomplete gamma function with boundary protection.

    When the shape parameter is zero, return the limiting value one to avoid
    numerical issues in torch.special.gammainc at this boundary. Used by NIM.
    """
    zero_shape_mask = (shape == 0)
    safe_shape = torch.where(zero_shape_mask, torch.ones_like(shape), shape)
    res = torch.special.gammainc(safe_shape, val)
    if res.shape != zero_shape_mask.shape:
        zero_shape_mask = zero_shape_mask.expand_as(res)
    return torch.where(zero_shape_mask, torch.ones_like(res), res)

@torch.jit.script
def custom_gamma_icdf_gpu(p: torch.Tensor, k: torch.Tensor, num_iters: int = 5) -> torch.Tensor:
    """GPU implementation of the gamma distribution quantile function.

    Initialize with a Wilson-Hilferty approximation, then refine with Newton
    iterations to reduce reliance on CPU statistical libraries. Primarily used
    for NIM/IM initial intervals and Berger-Boos quantiles.
    """
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
    """Gamma quantile function with nonnegative concentration handling."""
    safe_conc = torch.where(concentration > 0, concentration, torch.ones_like(concentration))
    std_icdf = custom_gamma_icdf_gpu(value, safe_conc, 5) 
    computed_values = std_icdf / rate
    return torch.where(concentration > 0, computed_values, torch.zeros_like(computed_values))

def get_poisson_quantile(p_fail: float, lam: float) -> int:
    """Return the integer boundary for the 1-p_fail quantile of Poisson(lambda)."""
    if lam <= 0:
        return 0
    prob = max(min(1.0 - p_fail, 1.0 - 1e-9), 1e-9)
    mu = torch.tensor([lam], device=device, dtype=dtype)
    q = poisson_ppf_tensor(prob, mu)[0].item()
    return int(max(q, 0.0))

@torch.jit.script
def _norm_ppf_torch(prob: torch.Tensor) -> torch.Tensor:
    eps = 1e-9
    p = torch.clamp(prob, eps, 1.0 - eps)
    return torch.sqrt(torch.tensor(2.0, device=p.device, dtype=p.dtype)) * torch.erfinv(2.0 * p - 1.0)

def poisson_ppf_tensor(prob: float, mu: torch.Tensor) -> torch.Tensor:
    """Compute Poisson quantiles elementwise for a tensor of means using torch."""
    mu_safe = torch.clamp(mu, min=0.0)
    p = torch.full_like(mu_safe, float(prob))
    z = _norm_ppf_torch(p)

    # Start with an Anscombe approximation, then apply a few discrete Poisson CDF corrections.
    approx = mu_safe + z * torch.sqrt(mu_safe + 0.25) + (z * z - 1.0) / 6.0
    q = torch.floor(torch.clamp(approx, min=0.0))

    # Limit correction iterations to control computation cost.
    for _ in range(3):
        shape = q + 1.0
        cdf_q = torch.special.gammaincc(shape, mu_safe)
        q = torch.where(cdf_q < p, q + 1.0, q)
        q_minus = torch.clamp(q - 1.0, min=0.0)
        cdf_minus = torch.special.gammaincc(q_minus + 1.0, mu_safe)
        q = torch.where((q > 0.0) & (cdf_minus >= p), q_minus, q)

    return q.to(dtype=mu.dtype)

def chi2_ppf_tensor(prob: torch.Tensor, df: torch.Tensor) -> torch.Tensor:
    """Compute chi-square quantiles elementwise for a tensor of degrees of freedom."""
    p = torch.clamp(prob, min=1e-9, max=1.0 - 1e-9)
    d = torch.clamp(df, min=0.0)
    shape = torch.clamp(d / 2.0, min=1e-9)
    # If Y ~ Gamma(k=df/2, rate=1), then 2Y ~ Chi2(df).
    return 2.0 * custom_gamma_icdf_gpu(p, shape, 6)

def build_lambda_grid(lower: torch.Tensor, upper: torch.Tensor, num_points: int) -> torch.Tensor:
    """Build a row-wise linear lambda search grid between the given bounds."""
    t = torch.linspace(0.0, 1.0, num_points, device=lower.device, dtype=lower.dtype)
    return lower.unsqueeze(1) + (upper - lower).unsqueeze(1) * t.unsqueeze(0)

# ============================================================
# Module 2: Probability calculations
# ============================================================
def compute_poisson_prob_matrix(lam_val: float, m_val: float, eb: float, x_max: int, w_max: int):
    """
    Compute the joint probability matrix P(X=x, W=w) on a truncated sample space.

    Parameters
    ----
    lam_val : float
        Signal parameter lambda.
    m_val : float
        Background rate multiplier m.
    eb : float
        True background parameter used in the simulation study.
    x_max, w_max : int
        Upper boundaries of the truncated sample space along x and w.

    Returns
    ----
    torch.Tensor
        Joint probability matrix with shape [x_max+1, w_max+1].
    """
    # Mean parameters for the two independent Poisson variables.
    theta_x = lam_val + eb
    theta_w = m_val * eb
    
    # Step 1: Compute the marginal probability vector for X.
    x_range = torch.arange(x_max + 1, device=device, dtype=dtype)
    # Use the log domain to reduce underflow from large factorials.
    log_px = x_range * math.log(theta_x) - theta_x - torch.lgamma(x_range + 1.0)
    px = torch.exp(log_px)
    
    # Step 2: Compute the marginal probability vector for W.
    w_range = torch.arange(w_max + 1, device=device, dtype=dtype)
    log_pw = w_range * math.log(theta_w) - theta_w - torch.lgamma(w_range + 1.0)
    pw = torch.exp(log_pw)
    
    # Step 3: Form the joint matrix by independence; entry (i, j) has x=i, w=j.
    prob_matrix = torch.ger(px, pw)
    
    return prob_matrix


# ============================================================
# Module 3: Core initial interval calculations
# ============================================================

@torch.jit.script
def f1_batch_nim(theta: torch.Tensor, x: torch.Tensor, a1: torch.Tensor, u1: torch.Tensor) -> torch.Tensor:
    """Randomly weighted equation for the signal component in NIM."""
    p1 = vector_gammainc_safe(x, theta)
    p2 = vector_gammainc_safe(x + 1, theta)
    return a1 * p1 + (1.0 - a1) * p2 - u1

@torch.jit.script
def f2_batch_nim(e: torch.Tensor, w: torch.Tensor, a2: torch.Tensor, u2: torch.Tensor) -> torch.Tensor:
    """Randomly weighted equation for the background component in NIM."""
    p1 = vector_gammainc_safe(w, e)
    p2 = vector_gammainc_safe(w + 1, e)
    return a2 * p1 + (1.0 - a2) * p2 - u2

@torch.jit.script
def solve_nim_signal_roots(x: torch.Tensor, a1: torch.Tensor, u1: torch.Tensor,
                           iterations: int = 30) -> torch.Tensor:
    """TorchScript-batched bisection for the NIM signal roots."""
    low = torch.zeros_like(a1)
    high = torch.full_like(a1, 5000.0)
    for _ in range(iterations):
        mid = (low + high) / 2.0
        val = f1_batch_nim(mid, x, a1, u1)
        mask = val > 0
        high = torch.where(mask, mid, high)
        low = torch.where(~mask, mid, low)
    return (low + high) / 2.0

@torch.jit.script
def solve_nim_background_roots(w: torch.Tensor, a2: torch.Tensor, u2: torch.Tensor,
                               iterations: int = 30) -> torch.Tensor:
    """TorchScript-batched bisection for the NIM background roots."""
    low = torch.zeros_like(a2)
    high = torch.full_like(a2, 5000.0)
    for _ in range(iterations):
        mid = (low + high) / 2.0
        val = f2_batch_nim(mid, w, a2, u2)
        mask = val > 0
        high = torch.where(mask, mid, high)
        low = torch.where(~mask, mid, low)
    return (low + high) / 2.0

def solve_roots_bisection_optimized(func, target_args, lower=0.0, upper=5000.0, iterations=30):
    """Solve monotone equations in batches by bisection."""
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

def calculate_nim_intervals(X_vec, W_vec, m_param, alpha, N):
    """Compute the NIM initial interval."""
    batch_size = X_vec.shape[0]
    X_exp = X_vec.unsqueeze(1)
    W_exp = W_vec.unsqueeze(1)
    a1 = torch.rand((batch_size, N), device=device, dtype=dtype)
    a2 = torch.rand((batch_size, N), device=device, dtype=dtype)
    u1 = torch.rand((batch_size, N), device=device, dtype=dtype)
    u2 = torch.rand((batch_size, N), device=device, dtype=dtype)
    root1 = solve_nim_signal_roots(X_exp, a1, u1, iterations=30)
    root2 = solve_nim_background_roots(W_exp, a2, u2, iterations=30)
    r = root1 - root2 / m_param
    q_lower = alpha / 2.0
    q_upper = 1.0 - alpha / 2.0
    nim1 = torch.quantile(r, q_lower, dim=1)
    nim2 = torch.quantile(r, q_upper, dim=1)
    return torch.clamp(nim1, min=0), torch.clamp(nim2, min=0)

def calculate_im_intervals(X_vec, W_vec, m_param, alpha, N):
    """Compute the IM initial interval."""
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
def calculate_bayes_hpd_core(x_vec, w_vec, m_param, target_ci, chunk_size=200):
    """Compute the Bayesian interval."""
    M = x_vec.shape[0]
    a, b = 1.0, 1.0
    grid_start, grid_end, grid_step = 0.0, 50.0, 0.005 
    grid_size = int((grid_end - grid_start) / grid_step) + 1
    grid_tensor = torch.linspace(grid_start, grid_end, steps=grid_size, device=device)
    L_out = torch.zeros(M, device=device, dtype=dtype)
    U_out = torch.zeros(M, device=device, dtype=dtype)
    
    for i in range(0, M, chunk_size):
        end = min(i + chunk_size, M)
        curr_bs = end - i
        x = x_vec[i:end]
        w = w_vec[i:end]
        
        max_x = int(x.max().item())
        k_indices = torch.arange(max_x + 1, device=device).unsqueeze(0)
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
        log_common_masked[~mask] = -float('inf')
        log_pwx = torch.logsumexp(log_common_masked, dim=1, keepdim=True)
        log_weights = log_common_masked - log_pwx

        shape_param   = (x_expanded - k_indices + 1)           
        shape_safe    = torch.clamp(shape_param, min=1.0)      
        shape_safe    = shape_safe.unsqueeze(1)                

        def calculate_G(lam_vals):
            lam_bc = lam_vals.unsqueeze(2)
            p_val = torch.special.gammainc(shape_safe, lam_bc)
            weights_bc = log_weights.unsqueeze(1)
            return torch.sum(torch.exp(weights_bc) * p_val, dim=2)

        def calculate_g(lam_vals):
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

        target_G = torch.full((curr_bs, 1), target_ci, device=device)
        l, u = torch.full((curr_bs, 1), 1e-6, device=device), torch.full((curr_bs, 1), 1000.0, device=device)
        for _ in range(30):
            mid = (l + u) / 2.0
            vals = calculate_G(mid)
            cond = vals < target_G
            l = torch.where(cond, mid, l)
            u = torch.where(cond, u, mid)
        z = (l + u) / 2.0
        
        best_vals = torch.full((curr_bs,), -1.0, device=device)
        best_indices = torch.zeros((curr_bs,), dtype=torch.long, device=device)
        grid_chunk_size = 5000
        g_chunks = int(math.ceil(grid_size / grid_chunk_size))
        for gc in range(g_chunks):
            gst = gc * grid_chunk_size
            ged = min((gc + 1) * grid_chunk_size, grid_size)
            sub_grid = grid_tensor[gst:ged].unsqueeze(0).expand(curr_bs, -1)
            g_vals = calculate_g(sub_grid)
            max_v, max_i = torch.max(g_vals, dim=1)
            update = max_v > best_vals
            best_vals[update] = max_v[update]
            best_indices[update] = max_i[update] + gst
        mod = grid_tensor[best_indices].unsqueeze(1)
        
        g_0 = calculate_g(torch.zeros((curr_bs, 1), device=device))
        g_z = calculate_g(z)
        l_final = torch.zeros((curr_bs, 1), device=device)
        u_final = torch.zeros((curr_bs, 1), device=device)
        
        cond1 = (g_0 >= g_z).squeeze()
        l_final[cond1] = 0.0
        u_final[cond1] = z[cond1] 
        idx_complex = ~cond1
        if idx_complex.any():
            curr_c0 = g_0.clone()
            curr_c1 = g_z.clone()
            for _ in range(25):
                c_mid = (curr_c0 + curr_c1) / 2.0
                ly, uy = torch.zeros((curr_bs, 1), device=device), mod.clone()
                for _ in range(20):
                    my = (ly + uy) / 2.0
                    gy = calculate_g(my)
                    too_small = gy < c_mid
                    ly = torch.where(too_small, my, ly)
                    uy = torch.where(too_small, uy, my)
                y_res = (ly + uy) / 2.0
                lz, uz = mod.clone(), torch.full((curr_bs, 1), 1000.0, device=device)
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

# ============================================================
# Module 4: h-function and dynamic truncation
# ============================================================

def compute_berger_boos_bounds(w_obs: torch.Tensor, m: float, beta: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute the 1-beta confidence set for the Berger-Boos background correction."""
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
    """Generate a numerical lambda search upper bound from the current interval."""
    return torch.clamp(interval_upper + 3.0 * torch.sqrt(torch.clamp(interval_upper, min=0.0)), min=0.0)

def compute_dynamic_sample_limits(w_obs: torch.Tensor, interval_upper: torch.Tensor, m: float,
                                  beta: float, eta: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute dynamic truncation limits X_max and W_max from the observation and interval."""
    _, eps_max = compute_berger_boos_bounds(w_obs, m, beta)
    lam_max = compute_dynamic_lambda_upper(interval_upper)
    mu_x_max = lam_max + eps_max
    mu_w_max = m * eps_max
    x_cut = poisson_ppf_tensor(1.0 - eta, mu_x_max)
    w_cut = poisson_ppf_tensor(1.0 - eta, mu_w_max)
    return x_cut, w_cut

def determine_global_lut_limits(lambda_vals: List[float], m_param: float, eb: float, eta: float,
                                beta: float, core_func, alpha: float, lut_chunk_size: int,
                                kwargs: Dict[str, Any]) -> Tuple[int, int]:
    """Estimate common sample space limits for the full LUT.
    The paper defines dynamic truncation separately for each observation. To
    generate the LUT in batches, the program first needs conservative global
    limits. It builds initial intervals on a smaller observation domain and uses
    their maximum upper endpoint to cover the full LUT. This batching detail leaves the statistical definition unchanged.
    """
    obs_x_max = get_poisson_quantile(eta, max(lambda_vals) + eb)
    obs_w_max = get_poisson_quantile(eta, m_param * eb)

    L_obs, U_obs = generate_lut(core_func, m_param, alpha, obs_x_max, obs_w_max,
                                chunk_size=lut_chunk_size, **kwargs)
    max_u0 = torch.max(U_obs).item()
    w_obs_max_tensor = torch.tensor([float(obs_w_max)], device=device, dtype=dtype)
    _, eps_max = compute_berger_boos_bounds(w_obs_max_tensor, m_param, beta)
    global_lambda_upper = max(max_u0 + torch.sqrt(torch.tensor([max_u0])), 0.0)
    mu_x_max = global_lambda_upper + eps_max.item()
    mu_w_max = m_param * eps_max.item()
    x_max_lut = get_poisson_quantile(eta, mu_x_max)
    w_max_lut = get_poisson_quantile(eta, mu_w_max)
    x_max_lut = max(x_max_lut, obs_x_max)
    w_max_lut = max(w_max_lut, obs_w_max)
    return x_max_lut, w_max_lut

class HFunctionCalculator:
    """h-function calculator.

    This class:
    1. Stores the current interval LUT;
    2. Builds the Berger-Boos grid and background probabilities;
    3. Evaluates the h-function at a specified lambda0;
    4. Performs one interval refinement using a fixed LUT.
    """
    def __init__(self, m: float, bb_beta: float, eta: float, num_b_grid: int = 1000,
                 num_lambda_grid: int = 500, lambda_eval_batch_size: int = 1,
                 b_block_size: int = 512):
        self.m = m
        self.beta = bb_beta
        self.eta = eta
        self.num_b_grid = num_b_grid
        self.num_lambda_grid = num_lambda_grid
        self.lambda_eval_batch_size = max(1, int(lambda_eval_batch_size))
        self.b_block_size = max(32, int(b_block_size))
        self.L0_grid = None
        self.U0_grid = None
        self.x_max = 0
        self.w_max = 0

        self._lgamma_w_cache: Optional[torch.Tensor] = None   
        self._lgamma_x_cache: Optional[torch.Tensor] = None   
        self._x_range_cache: Optional[torch.Tensor] = None
        self._w_range_cache: Optional[torch.Tensor] = None
        
    def set_lut(self, L0, U0):
        """Set the interval lookup table for the current iteration."""
        self.L0_grid = L0.contiguous()                                     
        self.U0_grid = U0.contiguous()                                     
        self.x_max = L0.shape[0] - 1
        self.w_max = L0.shape[1] - 1

        w_range = torch.arange(self.w_max + 1, device=device, dtype=dtype)
        self._lgamma_w_cache = torch.lgamma(w_range + 1.0)    
        self._w_range_cache = w_range

        x_range = torch.arange(self.x_max + 1, device=device, dtype=dtype)
        self._lgamma_x_cache = torch.lgamma(x_range + 1.0)    
        self._x_range_cache = x_range
        
    def prepare_b_grid_and_pw(self, w_obs):
        """Build the Berger-Boos grid and background probability table for observed w."""
        batch_size = w_obs.shape[0]
        b_min, b_max = compute_berger_boos_bounds(w_obs, self.m, self.beta)
        t = torch.linspace(0, 1, self.num_b_grid, device=device, dtype=dtype)
        t = t.unsqueeze(0).expand(batch_size, -1)
        b_grid = b_min.unsqueeze(1) + (b_max - b_min).unsqueeze(1) * t 
        mb = self.m * b_grid                                         

        w_range   = self._w_range_cache
        log_w     = w_range.view(1, 1, -1)                             
        lgamma_w  = self._lgamma_w_cache.view(1, 1, -1)                 

        mb_exp    = mb.unsqueeze(2)                                   
        log_pw    = log_w * torch.log(mb_exp + 1e-12) - mb_exp - lgamma_w
        pw        = torch.exp(log_pw)
        # Store the background probabilities as [batch, w, b], the layout
        # consumed by bmm below. Transpose once here instead of making a
        # contiguous copy for every lambda candidate inside the hot loop.
        pw = pw.permute(0, 2, 1).contiguous()
        return b_grid, pw
    
    def compute_h_with_precalc(self, x_obs, w_obs, lambda0, b_grid, pw, x_cut, w_cut,
                               pw_group_idx=None):
        """Evaluate the h-function for an observation and candidate lambda0.

        Notes:
        1. Evaluation uses dynamic sample-space truncation and the Berger-Boos set;
        2. Summation is chunked along the b and w dimensions to avoid OOM;
        3. Chunking changes summation order but not the statistic's definition.
        """
        batch_size = x_obs.shape[0]
        num_b      = b_grid.shape[1]
        
        idx_x  = torch.clamp(x_obs.long(), 0, self.x_max)
        idx_w  = torch.clamp(w_obs.long(), 0, self.w_max)
        L0_obs = self.L0_grid[idx_x, idx_w]
        U0_obs = self.U0_grid[idx_x, idx_w]
        T2_obs = torch.minimum(lambda0 - L0_obs, U0_obs - lambda0)
        
        # For a group of w slices, retain one b-grid/probability table per w
        # group and gather the matching b-grid row here. The large pw table is
        # gathered only for the current (b, w) tile below.
        if pw_group_idx is not None:
            b_grid_rows = b_grid.index_select(0, pw_group_idx)
        else:
            b_grid_rows = b_grid
            if b_grid.shape[0] == 1 and batch_size > 1:
                b_grid_rows = b_grid.expand(batch_size, -1)
            if pw.shape[0] == 1 and batch_size > 1:
                pw = pw.expand(batch_size, -1, -1)

        x_range    = self._x_range_cache
        lambda0_exp  = lambda0.view(batch_size, 1, 1)
        T2_obs_exp   = T2_obs.view(batch_size, 1, 1)

        x_valid = (x_range.view(1, -1)
                   <= x_cut.view(-1, 1))
        w_valid = (self._w_range_cache.view(1, -1)
                   <= w_cut.view(-1, 1))

        num_x = self.x_max + 1
        num_w = self.w_max + 1
        dtype_bytes = torch.tensor([], dtype=dtype).element_size()

        # Always chunk: adjust block sizes to the budget to avoid OOM and improve throughput.
        base_w_block = int(max(16, min(num_w, 512)))
        bytes_per_w = max(batch_size * num_x * dtype_bytes * 4, 1)
        budget_w_block = int(max(16, min(num_w, _CHUNK_WORKSPACE_BUDGET_BYTES // bytes_per_w)))
        w_block_size = min(base_w_block, budget_w_block)

        base_b_block = int(max(32, min(num_b, self.b_block_size)))
        bytes_per_b = max(batch_size * num_x * dtype_bytes * 3, 1)
        budget_b_block = int(max(32, min(num_b, _CHUNK_WORKSPACE_BUDGET_BYTES // bytes_per_b)))
        b_block_size = min(base_b_block, budget_b_block)

        prob_sum = torch.zeros((batch_size, num_b), device=device, dtype=dtype)
        for b_start in range(0, num_b, b_block_size):
            b_end = min(b_start + b_block_size, num_b)
            b_grid_sub = b_grid_rows[:, b_start:b_end]
            theta_sub = lambda0.unsqueeze(1) + b_grid_sub
            theta_sub_exp = theta_sub.unsqueeze(2)
            log_px_sub = (
                x_range.view(1, 1, -1) * torch.log(theta_sub_exp + 1e-12)
                - theta_sub_exp
                - self._lgamma_x_cache.view(1, 1, -1)
            )
            px_sub = torch.exp(log_px_sub)
            prob_sum_sub = torch.zeros((batch_size, b_end - b_start), device=device, dtype=dtype)
            pw_sub_all = pw[:, :, b_start:b_end]

            for w_start in range(0, num_w, w_block_size):
                w_end = min(w_start + w_block_size, num_w)
                L0_sub = self.L0_grid[:, w_start:w_end]
                U0_sub = self.U0_grid[:, w_start:w_end]
                T2_sub = torch.minimum(
                    lambda0_exp - L0_sub.unsqueeze(0),
                    U0_sub.unsqueeze(0) - lambda0_exp
                )
                w_valid_sub = w_valid[:, w_start:w_end]
                trunc_mask_sub = x_valid.unsqueeze(2) & w_valid_sub.unsqueeze(1)
                mask_sub = ((T2_sub <= T2_obs_exp) & trunc_mask_sub).to(dtype)
                if pw_group_idx is None:
                    pw_sub = pw_sub_all[:, w_start:w_end, :]
                else:
                    pw_sub = pw_sub_all.index_select(0, pw_group_idx)[:, w_start:w_end, :]
                temp_sum_w = torch.bmm(mask_sub.contiguous(), pw_sub)
                prob_sum_sub += torch.sum(px_sub * temp_sum_w.permute(0, 2, 1), dim=2)

            prob_sum[:, b_start:b_end] = prob_sum_sub
             
        h_sup, _ = prob_sum.max(dim=1)
        return torch.clamp(h_sup + self.beta, max=1.0)

    def refine_intervals_with_precalc_input(self, x_obs, w_obs, alpha, L_init, U_init,
                                            b_grid, pw, is_first_iter=False,
                                            pw_group_idx=None):
        """Refine intervals using a prepared Berger-Boos grid and probability table.

        The first refinement searches a wider lambda range. Later iterations
        search within the previous interval to ensure nested shrinkage (Algorithm 2).
        """
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

        # Evaluate adjacent lambda candidates together when the b-grid and
        # nuisance probability tables are either shared or indexed by group.
        # Flattening (observation, lambda) keeps the h-function definition
        # unchanged while reducing Python and CUDA launch overhead.
        can_batch_lambdas = (
            self.lambda_eval_batch_size > 1
            and (b_grid.shape[0] == 1 or pw_group_idx is not None)
        )
        if can_batch_lambdas:
            batch_size = x_obs.shape[0]
            lambda_batch_size = self.lambda_eval_batch_size
            for j_start in range(0, self.num_lambda_grid, lambda_batch_size):
                j_end = min(j_start + lambda_batch_size, self.num_lambda_grid)
                current_lambda_batch = j_end - j_start
                lambda_values = lambda_grid[:, j_start:j_end]
                x_eval = x_obs.unsqueeze(1).expand(-1, current_lambda_batch).reshape(-1)
                w_eval = w_obs.unsqueeze(1).expand(-1, current_lambda_batch).reshape(-1)
                x_cut_eval = x_cut.unsqueeze(1).expand(-1, current_lambda_batch).reshape(-1)
                w_cut_eval = w_cut.unsqueeze(1).expand(-1, current_lambda_batch).reshape(-1)
                group_idx_eval = None
                if pw_group_idx is not None:
                    group_idx_eval = pw_group_idx.unsqueeze(1).expand(
                        -1, current_lambda_batch
                    ).reshape(-1)
                h_values = self.compute_h_with_precalc(
                    x_eval,
                    w_eval,
                    lambda_values.reshape(-1),
                    b_grid,
                    pw,
                    x_cut_eval,
                    w_cut_eval,
                    group_idx_eval,
                )
                h_grid[:, j_start:j_end] = h_values.view(batch_size, current_lambda_batch)
        else:
            for j in range(self.num_lambda_grid):
                h_grid[:, j] = self.compute_h_with_precalc(
                    x_obs,
                    w_obs,
                    lambda_grid[:, j],
                    b_grid,
                    pw,
                    x_cut,
                    w_cut,
                    pw_group_idx,
                )

        inside = h_grid > threshold
        has_inside = inside.any(dim=1)

        idx_first = torch.argmax(inside.to(torch.int64), dim=1)
        idx_last = (self.num_lambda_grid - 1) - torch.argmax(torch.flip(inside.to(torch.int64), dims=[1]), dim=1)

        left_idx0 = torch.clamp(idx_first - 1, min=0)
        left_idx1 = idx_first
        right_idx0 = idx_last
        right_idx1 = torch.clamp(idx_last + 1, max=self.num_lambda_grid - 1)

        l_low = torch.gather(lambda_grid, 1, left_idx0.unsqueeze(1)).squeeze(1)
        l_high = torch.gather(lambda_grid, 1, left_idx1.unsqueeze(1)).squeeze(1)
        u_low = torch.gather(lambda_grid, 1, right_idx0.unsqueeze(1)).squeeze(1)
        u_high = torch.gather(lambda_grid, 1, right_idx1.unsqueeze(1)).squeeze(1)

        if device.type == "cuda":
            # The lower and upper endpoint searches are independent at each
            # bisection step. Concatenate them into one GPU batch to halve the
            # number of h-function launches while applying the same updates.
            batch_size = x_obs.shape[0]
            x_pair = torch.cat((x_obs, x_obs))
            w_pair = torch.cat((w_obs, w_obs))
            x_cut_pair = torch.cat((x_cut, x_cut))
            w_cut_pair = torch.cat((w_cut, w_cut))
            if pw_group_idx is not None:
                b_grid_pair, pw_pair = b_grid, pw
                group_idx_pair = torch.cat((pw_group_idx, pw_group_idx))
            elif b_grid.shape[0] == 1:
                b_grid_pair, pw_pair = b_grid, pw
                group_idx_pair = None
            else:
                b_grid_pair = torch.cat((b_grid, b_grid))
                pw_pair = torch.cat((pw, pw))
                group_idx_pair = None

            for _ in range(30):
                mid_lower = (l_low + l_high) / 2.0
                mid_upper = (u_low + u_high) / 2.0
                h_pair = self.compute_h_with_precalc(
                    x_pair, w_pair, torch.cat((mid_lower, mid_upper)),
                    b_grid_pair, pw_pair, x_cut_pair, w_cut_pair, group_idx_pair
                )
                h_lower, h_upper = h_pair[:batch_size], h_pair[batch_size:]
                lower_inside = h_lower > threshold
                upper_inside = h_upper > threshold
                l_high = torch.where(lower_inside, mid_lower, l_high)
                l_low = torch.where(~lower_inside, mid_lower, l_low)
                u_low = torch.where(upper_inside, mid_upper, u_low)
                u_high = torch.where(~upper_inside, mid_upper, u_high)
        else:
            for _ in range(30):
                mid = (l_low + l_high) / 2.0
                h = self.compute_h_with_precalc(
                    x_obs, w_obs, mid, b_grid, pw, x_cut, w_cut, pw_group_idx
                )
                mask_in = h > threshold
                l_high = torch.where(mask_in, mid, l_high)
                l_low = torch.where(~mask_in, mid, l_low)

            for _ in range(30):
                mid = (u_low + u_high) / 2.0
                h = self.compute_h_with_precalc(
                    x_obs, w_obs, mid, b_grid, pw, x_cut, w_cut, pw_group_idx
                )
                mask_in = h > threshold
                u_low = torch.where(mask_in, mid, u_low)
                u_high = torch.where(~mask_in, mid, u_high)

        L_final = torch.where(has_inside, (l_low + l_high) / 2.0, L_init)
        U_final = torch.where(has_inside, (u_low + u_high) / 2.0, U_init)

        zero_h = self.compute_h_with_precalc(
            x_obs, w_obs, torch.zeros_like(L_final), b_grid, pw, x_cut, w_cut,
            pw_group_idx
        )
        L_final = torch.where(zero_h > threshold, torch.zeros_like(L_final), L_final)
        return L_final, U_final
    
    def refine_intervals(self, x_obs, w_obs, alpha, L_init, U_init):
        """Public interface for one interval refinement."""
        b_grid, pw = self.prepare_b_grid_and_pw(w_obs)
        return self.refine_intervals_with_precalc_input(x_obs, w_obs, alpha, L_init, U_init, b_grid, pw, is_first_iter=True)


# ============================================================
# Module 5: Interval lookup tables and iterative refinement
# ============================================================

def generate_lut(method_core_func, m_param, alpha, x_max, w_max, chunk_size=2000, **kwargs):
    """Generate the initial interval LUT on the truncated sample space."""
    x_range = torch.arange(x_max + 1, device=device, dtype=dtype)
    w_range = torch.arange(w_max + 1, device=device, dtype=dtype)
    X_grid, W_grid = torch.meshgrid(x_range, w_range, indexing='ij')
    X_flat, W_flat = X_grid.flatten(), W_grid.flatten()
    L_flat = torch.zeros_like(X_flat)
    U_flat = torch.zeros_like(X_flat)
    total = X_flat.shape[0]
    for i in tqdm(range(0, total, chunk_size), desc=f"Gen C0 (alpha={alpha:.2f})"):
        end = min(i+chunk_size, total)
        X_sub = X_flat[i:end]
        W_sub = W_flat[i:end]
        l_sub, u_sub = method_core_func(X_sub, W_sub, m_param, alpha, **kwargs)
        L_flat[i:end] = l_sub
        U_flat[i:end] = u_sub
    return L_flat.view(x_max+1, w_max+1), U_flat.view(x_max+1, w_max+1)

def refine_entire_lut_optimized(h_calc, L_prev, U_prev, alpha, h_chunk_size=500,
                                h_w_chunk_size=1, desc="Refining", is_first_iter=False):
    """Refine the full LUT in one pass."""
    x_max = L_prev.shape[0] - 1
    w_max = L_prev.shape[1] - 1
    
    L_new = torch.zeros_like(L_prev)
    U_new = torch.zeros_like(U_prev)
    
    h_w_chunk_size = max(1, int(h_w_chunk_size))
    for w_start in tqdm(range(0, w_max + 1, h_w_chunk_size), desc=desc, leave=False):
        w_end = min(w_start + h_w_chunk_size, w_max + 1)
        w_values = torch.arange(w_start, w_end, device=device, dtype=dtype)
        group_count = w_end - w_start
        b_grid_group, pw_group = h_calc.prepare_b_grid_and_pw(w_values)
        group_ids = torch.arange(group_count, device=device, dtype=torch.long)
        
        for x_start in range(0, x_max + 1, h_chunk_size):
            x_end = min(x_start + h_chunk_size, x_max + 1)
            current_batch_size = x_end - x_start
            
            x_values = torch.arange(x_start, x_end, device=device, dtype=dtype)
            x_batch = x_values.repeat(group_count)
            w_batch = w_values.repeat_interleave(current_batch_size)
            pw_group_idx = group_ids.repeat_interleave(current_batch_size)
            
            l_init_batch = L_prev[x_start:x_end, w_start:w_end].transpose(0, 1).contiguous().view(-1)
            u_init_batch = U_prev[x_start:x_end, w_start:w_end].transpose(0, 1).contiguous().view(-1)

            l_new_sub, u_new_sub = h_calc.refine_intervals_with_precalc_input(
                x_batch, w_batch, alpha, l_init_batch, u_init_batch,
                b_grid_group, pw_group, is_first_iter=is_first_iter,
                pw_group_idx=pw_group_idx,
            )
            
            L_new[x_start:x_end, w_start:w_end] = l_new_sub.view(
                group_count, current_batch_size
            ).transpose(0, 1)
            U_new[x_start:x_end, w_start:w_end] = u_new_sub.view(
                group_count, current_batch_size
            ).transpose(0, 1)
            
    return L_new, U_new

def iterative_lut_generation(core_func, h_calc, config, alpha, x_max, w_max, kwargs, m_val):
    """Generate C0, C1, and CF following Algorithms 1 and 2 in the paper."""
    print(f"\n[Iterative refinement] m={m_val}, alpha={alpha:.3f}...")
    
    initial_lut_t0 = time.perf_counter()
    L_C0, U_C0 = generate_lut(core_func, m_val, alpha, x_max, w_max, 
                              chunk_size=config.lut_chunk_size, **kwargs)
    initial_lut_elapsed = time.perf_counter() - initial_lut_t0
    L_C0 = torch.clamp(L_C0, min=0)
    U_C0 = torch.clamp(U_C0, min=0)
    
    L_curr, U_curr = L_C0.clone(), U_C0.clone()
    L_C1, U_C1 = None, None
    final_iter = 0
    c1_improvement_elapsed = 0.0
    additional_refinement_elapsed = 0.0
    
    for k in range(1, config.max_iter + 1):
        iteration_t0 = time.perf_counter()
        h_calc.set_lut(L_curr, U_curr)
        
        # Iteration 1 is the first refinement; later iterations search within the previous interval.
        L_new, U_new = refine_entire_lut_optimized(h_calc, L_curr, U_curr, alpha, 
                                                 h_chunk_size=config.h_chunk_size,
                                                 h_w_chunk_size=config.h_w_chunk_size,
                                                 desc=f"Iter {k} Refinement",
                                                 is_first_iter=(k==1))
        
        # From iteration 2 onward, explicitly enforce the nesting constraint.
        if k >= 2:
            L_new = torch.max(L_new, L_curr)
            U_new = torch.min(U_new, U_curr)
        
        if k == 1:
            L_C1, U_C1 = L_new.clone(), U_new.clone()
        
        max_diff_L = torch.max(torch.abs(L_new - L_curr)).item()
        max_diff_U = torch.max(torch.abs(U_new - U_curr)).item()
        max_diff   = max(max_diff_L, max_diff_U)
        
        curr_total_len = torch.sum(U_new - L_new).item()
        print(f"    Iter {k}: MaxDiff={max_diff:.6f}, TotalLen={curr_total_len:.2f}")

        prev_L, prev_U = L_curr, U_curr
        L_curr, U_curr = L_new, U_new
        del prev_L, prev_U, L_new, U_new
        # Keep freed tensors in PyTorch's caching allocator. Releasing the
        # allocator cache here forces repeated driver synchronization and
        # slows the next refinement round.

        final_iter = k
        iteration_elapsed = time.perf_counter() - iteration_t0
        if k == 1:
            c1_improvement_elapsed = iteration_elapsed
        else:
            additional_refinement_elapsed += iteration_elapsed
        if max_diff < config.convergence_tol:
            break
            
    h_calc.set_lut(L_curr, U_curr)
    if L_C1 is None: L_C1, U_C1 = L_C0.clone(), U_C0.clone()
    
    return ((L_C0, U_C0), (L_C1, U_C1), (L_curr, U_curr), final_iter,
            initial_lut_elapsed, c1_improvement_elapsed,
            additional_refinement_elapsed)

def exact_lut_metrics(L_grid: torch.Tensor, U_grid: torch.Tensor, lam_val: float, m_val: float,
                      eb: float) -> Tuple[float, float]:
    """Compute CP and EL by exact summation on the current LUT support."""
    theta_x = lam_val + eb
    theta_w = m_val * eb

    x_range = torch.arange(L_grid.shape[0], device=device, dtype=dtype)
    w_range = torch.arange(L_grid.shape[1], device=device, dtype=dtype)

    log_px = x_range * math.log(theta_x) - theta_x - torch.lgamma(x_range + 1.0)
    log_pw = w_range * math.log(theta_w) - theta_w - torch.lgamma(w_range + 1.0)
    px = torch.exp(log_px)
    pw = torch.exp(log_pw)

    cover_mask = ((L_grid <= lam_val) & (lam_val <= U_grid)).to(dtype)
    lengths = U_grid - L_grid

    # Factor the independent joint Poisson distribution. This avoids building
    # a second dense x-by-w probability matrix for every lambda value.
    cover_prob_by_x = torch.mv(cover_mask, pw)
    length_by_x = torch.mv(lengths, pw)
    cp = torch.dot(px, cover_prob_by_x)
    el = torch.dot(px, length_by_x)
    return cp.item(), el.item()


# ============================================================
# Module 6: Main workflow and result output
# ============================================================

def format_beta_tag(beta: float) -> str:
    """Return a compact, stable beta label for output filenames."""
    return f"{beta:.0e}"


def calculate_nesting_diagnostics(l0_grid, u0_grid, l1_grid, u1_grid, tolerance: float):
    """Summarize C1 subset C0 over the full computed LUT support."""
    finite = (np.isfinite(l0_grid) & np.isfinite(u0_grid)
              & np.isfinite(l1_grid) & np.isfinite(u1_grid))
    lower_excess = np.maximum(l0_grid - l1_grid, 0.0)
    upper_excess = np.maximum(u1_grid - u0_grid, 0.0)
    subset_mask = ((l1_grid >= l0_grid - tolerance)
                   & (u1_grid <= u0_grid + tolerance))
    cells_checked = int(np.count_nonzero(finite))
    violations = int(np.count_nonzero(finite & ~subset_mask))
    max_lower_excess = float(np.max(lower_excess[finite])) if cells_checked else 0.0
    max_upper_excess = float(np.max(upper_excess[finite])) if cells_checked else 0.0
    return {
        "cells_checked": cells_checked,
        "c0_nonempty_cells": int(np.count_nonzero(finite & (u0_grid >= 1e-6))),
        "nesting_violations": violations,
        "nesting_violation_rate": violations / cells_checked if cells_checked else 0.0,
        "max_lower_endpoint_excess": max_lower_excess,
        "max_upper_endpoint_excess": max_upper_excess,
        "max_endpoint_excess": max(max_lower_excess, max_upper_excess),
        "subset_mask": subset_mask,
        "lower_excess_grid": lower_excess,
        "upper_excess_grid": upper_excess,
        "finite_mask": finite,
    }


def export_lut_comparison(L_init, U_init, L_c1, U_c1, L_final, U_final,
                          method, m, target_ci, eb, beta=0.0001,
                          nesting_tolerance=1e-4, filename=None):
    """Export interval LUTs and pointwise C1-subset-C0 diagnostics."""
    beta_tag = format_beta_tag(beta)
    if filename is None:
        filename = f"LUT-{method}-{target_ci:.2f}-{int(m)}-{eb}-beta-{beta_tag}.csv"

    x_max = L_init.shape[0]
    w_max = L_init.shape[1]
    print(f"    [Export] Saving LUT to {filename} ...")

    # Transfer each LUT to the CPU once; avoid per-cell GPU synchronization.
    arrays = [tensor.detach().cpu().numpy() for tensor in
              (L_init, U_init, L_c1, U_c1, L_final, U_final)]
    l0_grid, u0_grid, l1_grid, u1_grid, lf_grid, uf_grid = arrays
    nesting = calculate_nesting_diagnostics(
        l0_grid, u0_grid, l1_grid, u1_grid, nesting_tolerance
    )

    with open(filename, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            "X", "W", "Beta", "Nesting_Tolerance",
            "L_C0", "U_C0", "Len_C0",
            "L_C1", "U_C1", "Len_C1", "C1_Subset_C0",
            "Lower_Nesting_Excess", "Upper_Nesting_Excess",
            "L_Final", "U_Final", "Len_Final",
            "L_Diff_C0_C1", "U_Diff_C0_C1", "Len_Change_C0_C1",
            "L_Diff_C1_Final", "U_Diff_C1_Final", "Len_Change_C1_Final",
            "L_Diff_Total", "U_Diff_Total", "Len_Change_Total", "Status"
        ])

        for x in range(x_max):
            for w in range(w_max):
                u0 = float(u0_grid[x, w])
                if u0 < 1e-6:
                    continue

                l0 = float(l0_grid[x, w])
                l1 = float(l1_grid[x, w])
                u1 = float(u1_grid[x, w])
                lf = float(lf_grid[x, w])
                uf = float(uf_grid[x, w])
                len0, len1, lenf = u0 - l0, u1 - l1, uf - lf
                l_diff_01, u_diff_01 = l1 - l0, u1 - u0
                len_change_01 = len1 - len0
                l_diff_1f, u_diff_1f = lf - l1, uf - u1
                len_change_1f = lenf - len1
                l_diff_total, u_diff_total = lf - l0, uf - u0
                len_change_total = lenf - len0
                status = "Same"
                if abs(len_change_total) > 1e-4:
                    status = "SHRUNK" if len_change_total < 0 else "EXPANDED"

                writer.writerow([
                    x, w, f"{beta:.8g}", f"{nesting_tolerance:.8g}",
                    f"{l0:.8f}", f"{u0:.8f}", f"{len0:.8f}",
                    f"{l1:.8f}", f"{u1:.8f}", f"{len1:.8f}",
                    bool(nesting["subset_mask"][x, w]),
                    f"{nesting['lower_excess_grid'][x, w]:.8g}",
                    f"{nesting['upper_excess_grid'][x, w]:.8g}",
                    f"{lf:.8f}", f"{uf:.8f}", f"{lenf:.8f}",
                    f"{l_diff_01:.8f}", f"{u_diff_01:.8f}", f"{len_change_01:.8f}",
                    f"{l_diff_1f:.8f}", f"{u_diff_1f:.8f}", f"{len_change_1f:.8f}",
                    f"{l_diff_total:.8f}", f"{u_diff_total:.8f}", f"{len_change_total:.8f}",
                    status
                ])

    print(f"    [Nesting diagnostic] C1⊆C0 violations={nesting['nesting_violations']}/"
          f"{nesting['cells_checked']}, tolerance={nesting_tolerance:g}, "
          f"maximum endpoint excess={nesting['max_endpoint_excess']:.8g}")
    return {key: value for key, value in nesting.items()
            if key not in {"subset_mask", "lower_excess_grid", "upper_excess_grid", "finite_mask"}}


def run_full_simulation(method_name: str, config: SimulationConfig) -> List[SimulationResult]:
    """Run the full simulation workflow and save the results.

    The workflow:
    1. Generates the initial interval LUT for each method and confidence level;
    2. Computes the first refined interval and the final interval;
    3. Evaluates CP, EL, and ICP by exact summation over the LUT support;
    4. Saves numerical results to CSV files.
    """
    all_results = []
    timing_rows: List[Dict[str, Any]] = []
    nesting_rows: List[Dict[str, Any]] = []
    torch.manual_seed(config.random_seed)
    np.random.seed(config.random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.random_seed)
    
    # Outer loop: background rate multiplier m.
    for m_val in config.m_params:
        print(f"\n{'='*60}")
        print(f"PROCESSING m = {m_val} (EXACT CP/EL EVALUATION)")
        print(f"{'='*60}")

        # Middle loop: initial interval method.
        for current_method in config.methods_to_run:
            print(f"\n  >>> METHOD: {current_method} <<<")
    
            # Select the core initial interval function by method name.
            if current_method == 'Bayes':
                def bayes_adapter(x, w, m, a, **kw):
                    return calculate_bayes_hpd_core(x, w, m, 1.0-a, chunk_size=config.chunk_size)
                core_func = bayes_adapter
                kwargs = {}
            elif current_method == 'NIM':
                core_func = calculate_nim_intervals
                kwargs = {'N': config.N}
            elif current_method == 'IM':
                core_func = calculate_im_intervals
                kwargs = {'N': config.N}
            else: 
                print(f"Warning: Unknown method {current_method}, skipping.")
                continue
            
            # Inner loop: target confidence levels.
            for target_ci in config.target_cis:
                combination_t0 = time.perf_counter()
                bounds_t0 = time.perf_counter()
                alpha = 1.0 - target_ci
                print(f"\n    --- CI = {target_ci*100:.1f}% ---")
                x_max, w_max = determine_global_lut_limits(
                    config.lambda_vals, m_val, config.eb, config.truncation_eta,
                    config.berger_boos_beta, core_func, alpha, config.lut_chunk_size, kwargs
                )
                lut_bounds_elapsed = time.perf_counter() - bounds_t0
                print(f"[Setup] m={m_val}, method={current_method}, CI={target_ci}: X_max={x_max}, W_max={w_max}")
                
                # Save results for the current (method, m, CI) combination.
                current_results = []
                
                h_calc_shared = HFunctionCalculator(
                    m_val,
                    config.berger_boos_beta,
                    config.truncation_eta,
                    num_b_grid=config.num_b_grid,
                    num_lambda_grid=config.num_lambda_grid,
                    lambda_eval_batch_size=config.h_lambda_batch_size,
                    b_block_size=config.h_b_block_size,
                )
                
                # Step A: Generate the initial interval LUT and refine if requested.
                if config.use_iterative_refinement:
                    ((l0_lut, u0_lut), (l1_lut, u1_lut), (lf_lut, uf_lut), iter_count,
                     c0_lut_elapsed, c1_elapsed, cf_refinement_elapsed) = iterative_lut_generation(
                         core_func, h_calc_shared, config, alpha, x_max, w_max, kwargs, m_val
                     )
                    
                    lut_export_t0 = time.perf_counter()
                    nesting_diag = export_lut_comparison(
                        l0_lut, u0_lut, 
                        l1_lut, u1_lut,
                        lf_lut, uf_lut,
                        current_method, m_val, target_ci, config.eb,
                        beta=config.berger_boos_beta,
                        nesting_tolerance=config.nesting_tolerance,
                    )
                    lut_export_elapsed = time.perf_counter() - lut_export_t0
                    
                else:
                    c0_lut_t0 = time.perf_counter()
                    l0_lut, u0_lut = generate_lut(core_func, m_val, alpha, x_max, w_max, 
                                                 chunk_size=config.lut_chunk_size, **kwargs)
                    c0_lut_elapsed = time.perf_counter() - c0_lut_t0
                    l1_lut, u1_lut = l0_lut, u0_lut
                    lf_lut, uf_lut = l0_lut, u0_lut
                    iter_count = 0
                    c1_elapsed = 0.0
                    cf_refinement_elapsed = 0.0
                    lut_export_elapsed = 0.0
                    nesting_raw = calculate_nesting_diagnostics(
                        l0_lut.detach().cpu().numpy(), u0_lut.detach().cpu().numpy(),
                        l1_lut.detach().cpu().numpy(), u1_lut.detach().cpu().numpy(),
                        config.nesting_tolerance,
                    )
                    nesting_diag = {
                        key: value for key, value in nesting_raw.items()
                        if key not in {"subset_mask", "lower_excess_grid", "upper_excess_grid", "finite_mask"}
                    }

                nesting_rows.append({
                    "Method": current_method,
                    "m": m_val,
                    "b": config.eb,
                    "Alpha": alpha,
                    "Target_CI": target_ci,
                    "Beta": config.berger_boos_beta,
                    "Nesting_Tolerance": config.nesting_tolerance,
                    "Cells_Checked": nesting_diag["cells_checked"],
                    "C0_Nonempty_Cells": nesting_diag["c0_nonempty_cells"],
                    "Nesting_Violations": nesting_diag["nesting_violations"],
                    "Nesting_Violation_Rate": nesting_diag["nesting_violation_rate"],
                    "Max_Lower_Endpoint_Excess": nesting_diag["max_lower_endpoint_excess"],
                    "Max_Upper_Endpoint_Excess": nesting_diag["max_upper_endpoint_excess"],
                    "Max_Endpoint_Excess": nesting_diag["max_endpoint_excess"],
                    "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                })

                lut_elapsed = (lut_bounds_elapsed + c0_lut_elapsed + c1_elapsed
                               + cf_refinement_elapsed + lut_export_elapsed)

                # Step B: Evaluate interval performance by exact summation on the LUT support.
                print(f"    [Simulation] Exact CP / EL / ICP for Method={current_method}, m={m_val}, CI={target_ci}...")
                
                # Temporary variable for tracking the minimum coverage probability ICP.
                # Initialize above one so subsequent values can update the minimum.
                icp_0, icp_1, icp_final = 1.1, 1.1, 1.1

                cp_el_t0 = time.perf_counter()
                for lam_val in tqdm(config.lambda_vals, desc=f"Exact evaluation {current_method}"):
                    t0 = time.time()
                    curr_x_max = lf_lut.shape[0] - 1
                    curr_w_max = lf_lut.shape[1] - 1
                    cp_0, el_0 = exact_lut_metrics(l0_lut, u0_lut, lam_val, m_val, config.eb)
                    cp_1, el_1 = exact_lut_metrics(l1_lut, u1_lut, lam_val, m_val, config.eb)
                    cp_f, el_f = exact_lut_metrics(lf_lut, uf_lut, lam_val, m_val, config.eb)
                    
                    # Update the minimum coverage probability found so far.
                    icp_0 = min(icp_0, cp_0)
                    icp_1 = min(icp_1, cp_1)
                    icp_final = min(icp_final, cp_f)
                    
                    typ_x = min(int(lam_val + config.eb), curr_x_max)
                    typ_w = min(int(m_val * config.eb), curr_w_max)
                    ci_f_str = f"[{lf_lut[typ_x, typ_w]:.2f}, {uf_lut[typ_x, typ_w]:.2f}]"
                    ci_0_str = f"[{l0_lut[typ_x, typ_w]:.2f}, {u0_lut[typ_x, typ_w]:.2f}]"
                    ci_1_str = f"[{l1_lut[typ_x, typ_w]:.2f}, {u1_lut[typ_x, typ_w]:.2f}]"
                    
                    res = SimulationResult(
                        method=current_method,
                        lambda_val=lam_val, m_param=m_val, eb=config.eb,
                        target_ci=target_ci, beta=config.berger_boos_beta,
                        cp_0=cp_0, el_0=el_0, ci_0_example=ci_0_str,
                        cp_1=cp_1, el_1=el_1, ci_1_example=ci_1_str,
                        cp_final=cp_f, el_final=el_f, ci_final_example=ci_f_str,
                        iterations=iter_count, time_elapsed=time.time()-t0,
                        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    )
                    current_results.append(res)
                    all_results.append(res)

                cp_el_elapsed = time.perf_counter() - cp_el_t0
                combination_elapsed = time.perf_counter() - combination_t0
                for result in current_results:
                    result.lut_time_elapsed = lut_elapsed
                    result.combination_time_elapsed = combination_elapsed

                lambda_step = (
                    config.lambda_vals[1] - config.lambda_vals[0]
                    if len(config.lambda_vals) > 1 else None
                )
                timing_rows.append({
                    "Method": current_method,
                    "m": m_val,
                    "b": config.eb,
                    "Alpha": alpha,
                    "Target_CI": target_ci,
                    "Beta": config.berger_boos_beta,
                    "Nesting_Tolerance": config.nesting_tolerance,
                    "N": config.N if current_method in ("IM", "NIM") else None,
                    "Lambda_Start": config.lambda_vals[0] if config.lambda_vals else None,
                    "Lambda_Stop": config.lambda_vals[-1] if config.lambda_vals else None,
                    "Lambda_Step": lambda_step,
                    "X_Max": x_max,
                    "W_Max": w_max,
                    "Iterations": iter_count,
                    "LUT_Bounds_Time(s)": lut_bounds_elapsed,
                    "C0_Initial_LUT_Time(s)": c0_lut_elapsed,
                    "C1_One_Time_Improvement_Time(s)": c1_elapsed,
                    "CF_Additional_Refinement_Time(s)": cf_refinement_elapsed,
                    "LUT_Comparison_Export_Time(s)": lut_export_elapsed,
                    "CP_EL_Sweep_Time(s)": cp_el_elapsed,
                    "Scenario_Total_Time(s)": combination_elapsed,
                    "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                })
                
                # Report ICP for the current parameter combination.
                print(f"    [ICP report] Method={current_method}, CI={target_ci}")
                print(f"       ICP(C0):    {icp_0:.6f}")
                print(f"       ICP(C1):    {icp_1:.6f}")
                print(f"       ICP(F): {icp_final:.6f}")
                
                # Save the results for the current parameter combination.
                beta_tag = format_beta_tag(config.berger_boos_beta)
                cp_el_filename = f"{current_method}-{target_ci:.2f}-{int(m_val)}-{config.eb}-beta-{beta_tag}.csv"
                save_results_to_csv(current_results, cp_el_filename)
                save_timing_summary_to_csv(
                    timing_rows,
                    filename=f"Scenario_Runtime_Summary_beta-{beta_tag}.csv",
                )
                save_nesting_diagnostics_to_csv(
                    nesting_rows,
                    filename=f"Nesting_Diagnostics_beta-{beta_tag}.csv",
                )

                # Release GPU memory after each combination to reduce OOM risk.
                del l0_lut, u0_lut, l1_lut, u1_lut, lf_lut, uf_lut, current_results, h_calc_shared
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()

    return all_results

def save_results_to_csv(results: List[SimulationResult], filename: str = None):
    """Write result records to a structured CSV file."""
    if filename is None and len(results) > 0:
        res0 = results[0]
        filename = (f"{res0.method}-{res0.target_ci:.2f}-{int(res0.m_param)}-"
                    f"{res0.eb}-beta-{format_beta_tag(res0.beta)}.csv")
    elif filename is None:
        filename = "results.csv"
    fieldnames = [
        "Method", "Lambda", "m", "b", "Target_CI", "Beta",
        "CP_0", "EL_0", "CI_0_Ex",
        "CP_1", "EL_1", "CI_1_Ex",
        "CP_Final", "EL_Final", "CI_Final_Ex",
        "Iters", "Time(s)", "LUT_Refinement_Time(s)",
        "Combination_Compute_Time(s)", "Timestamp"
    ]
    with open(filename, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for res in results:
            writer.writerow({
                "Method": res.method, "Lambda": res.lambda_val, "m": res.m_param, "b": res.eb,
                "Target_CI": res.target_ci,
                "Beta": res.beta,
                "CP_0": f"{res.cp_0:.6f}", "EL_0": f"{res.el_0:.6f}", "CI_0_Ex": res.ci_0_example,
                "CP_1": f"{res.cp_1:.6f}", "EL_1": f"{res.el_1:.6f}", "CI_1_Ex": res.ci_1_example,
                "CP_Final": f"{res.cp_final:.6f}", "EL_Final": f"{res.el_final:.6f}", "CI_Final_Ex": res.ci_final_example,
                "Iters": res.iterations,
                "Time(s)": f"{res.time_elapsed:.2f}",
                "LUT_Refinement_Time(s)": f"{res.lut_time_elapsed:.2f}",
                "Combination_Compute_Time(s)": f"{res.combination_time_elapsed:.2f}",
                "Timestamp": res.timestamp
            })
    print(f"\nResults saved to {filename}")


def save_timing_summary_to_csv(timing_rows: List[Dict[str, Any]], filename: str = "Scenario_Runtime_Summary.csv"):
    """Write one timing row per (method, m, alpha, beta) simulation scenario."""
    fieldnames = [
        "Method", "m", "b", "Alpha", "Target_CI", "Beta", "Nesting_Tolerance", "N",
        "Lambda_Start", "Lambda_Stop", "Lambda_Step",
        "X_Max", "W_Max", "Iterations",
        "LUT_Bounds_Time(s)", "C0_Initial_LUT_Time(s)",
        "C1_One_Time_Improvement_Time(s)", "CF_Additional_Refinement_Time(s)",
        "LUT_Comparison_Export_Time(s)", "CP_EL_Sweep_Time(s)",
        "Scenario_Total_Time(s)", "Timestamp",
    ]
    time_fields = {
        "LUT_Bounds_Time(s)", "C0_Initial_LUT_Time(s)",
        "C1_One_Time_Improvement_Time(s)", "CF_Additional_Refinement_Time(s)",
        "LUT_Comparison_Export_Time(s)", "CP_EL_Sweep_Time(s)",
        "Scenario_Total_Time(s)",
    }
    with open(filename, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in timing_rows:
            writer.writerow({
                key: (f"{row[key]:.2f}" if key in time_fields else row[key])
                for key in fieldnames
            })
    print(f"\nScenario timing summary saved to {filename}")


def save_nesting_diagnostics_to_csv(
    diagnostic_rows: List[Dict[str, Any]],
    filename: str = "Nesting_Diagnostics.csv",
):
    """Save one C1 subset C0 diagnostic row per simulation scenario."""
    fieldnames = [
        "Method", "m", "b", "Alpha", "Target_CI", "Beta", "Nesting_Tolerance",
        "Cells_Checked", "C0_Nonempty_Cells", "Nesting_Violations",
        "Nesting_Violation_Rate", "Max_Lower_Endpoint_Excess",
        "Max_Upper_Endpoint_Excess", "Max_Endpoint_Excess", "Timestamp",
    ]
    with open(filename, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in diagnostic_rows:
            writer.writerow({
                key: (f"{row[key]:.8g}" if isinstance(row[key], float) else row[key])
                for key in fieldnames
            })
    print(f"Nesting diagnostic saved to {filename}")

def parse_args(argv=None):
    """Parse the experiment grid and runtime batch-size overrides."""
    parser = argparse.ArgumentParser(
        description="Run the Poisson h-function confidence-interval simulation.",
        epilog=(
            "Example: python code.py --methods IM --m 1 5 10 15 "
            "--alpha 0.05 --beta 1e-3. Set HFUNC_CHUNK_MB to override the automatic "
            "GPU workspace budget."
        ),
    )
    parser.add_argument("--methods", nargs="+", choices=("Bayes", "NIM", "IM"),
                        default=["Bayes"], help="initial interval methods to run")
    parser.add_argument("--m", dest="m_params", nargs="+", type=float, default=[20.0],
                        metavar="M", help="one or more background-rate constants")
    parser.add_argument("--alpha", nargs="+", type=float, default=[0.05],
                        metavar="ALPHA", help="one or more significance levels")
    parser.add_argument("--beta", type=float, default=1e-4,
                        help="Berger-Boos penalty parameter (default: 1e-4)")
    parser.add_argument("--nesting-tolerance", type=float, default=1e-4,
                        help="endpoint tolerance for the C1 subset C0 diagnostic (default: 1e-4)")
    parser.add_argument("--lambda-start", type=float, default=0.0,
                        help="first signal parameter value (default: 0)")
    parser.add_argument("--lambda-stop", type=float, default=10.0,
                        help="largest signal parameter value (default: 10)")
    parser.add_argument("--lambda-step", type=float, default=0.1,
                        help="signal parameter grid spacing (default: 0.1)")
    parser.add_argument("--eb", type=float, default=3.0,
                        help="background parameter used for evaluation (default: 3)")
    parser.add_argument("--N", type=int, default=10000,
                        help="auxiliary draws for IM/NIM initial intervals")
    parser.add_argument("--max-iter", type=int, default=10,
                        help="maximum interval-refinement iterations")
    parser.add_argument("--chunk-size", type=int, default=None,
                        help="override the device-tuned Bayes batch size")
    parser.add_argument("--h-chunk-size", type=int, default=None,
                        help="override the device-tuned h-refinement batch size")
    parser.add_argument("--h-lambda-batch-size", type=int, default=None,
                        help="number of lambda grid values evaluated together")
    parser.add_argument("--h-w-chunk-size", type=int, default=None,
                        help="number of w slices processed together during refinement")
    parser.add_argument("--h-b-block-size", type=int, default=None,
                        help="maximum Berger-Boos grid points processed per h-function block")
    parser.add_argument("--lut-chunk-size", type=int, default=None,
                        help="override the device-tuned initial-LUT batch size")
    args = parser.parse_args(argv)

    if any(not 0.0 < alpha < 1.0 for alpha in args.alpha):
        parser.error("each --alpha value must be strictly between 0 and 1")
    if not 0.0 < args.beta < 1.0:
        parser.error("--beta must be strictly between 0 and 1")
    if args.nesting_tolerance < 0.0:
        parser.error("--nesting-tolerance must be nonnegative")
    if any(m_value <= 0.0 for m_value in args.m_params) or args.eb <= 0.0:
        parser.error("each --m value and --eb must be positive")
    if args.lambda_start < 0.0 or args.lambda_step <= 0.0 or args.lambda_stop < args.lambda_start:
        parser.error("lambda grid requires start >= 0, a positive step, and stop >= start")
    lambda_steps = (args.lambda_stop - args.lambda_start) / args.lambda_step
    if not math.isclose(lambda_steps, round(lambda_steps), rel_tol=0.0, abs_tol=1e-9):
        parser.error("lambda range must be evenly divisible by --lambda-step")
    if args.N <= 0 or args.max_iter <= 0:
        parser.error("--N and --max-iter must be positive")
    for option_name in ("chunk_size", "h_chunk_size", "h_lambda_batch_size",
                        "h_w_chunk_size", "h_b_block_size", "lut_chunk_size"):
        value = getattr(args, option_name)
        if value is not None and value <= 0:
            parser.error(f"--{option_name.replace('_', '-')} must be positive")
    return args


def main(argv=None):
    args = parse_args(argv)
    num_lambda_steps = int(math.floor(
        (args.lambda_stop - args.lambda_start) / args.lambda_step + 1e-10
    ))
    lambda_vals = [args.lambda_start + i * args.lambda_step
                   for i in range(num_lambda_steps + 1)]
    target_cis = [1.0 - alpha for alpha in args.alpha]

    cfg = SimulationConfig(
        lambda_vals=lambda_vals,
        m_params=args.m_params,
        methods_to_run=args.methods,
        target_cis=target_cis,
        N=args.N,
        eb=args.eb,
        use_iterative_refinement=True,
        max_iter=args.max_iter,
        convergence_tol=1e-4,
        berger_boos_beta=args.beta,
        nesting_tolerance=args.nesting_tolerance,
        chunk_size=args.chunk_size,
        h_chunk_size=args.h_chunk_size,
        h_lambda_batch_size=args.h_lambda_batch_size,
        h_w_chunk_size=args.h_w_chunk_size,
        h_b_block_size=args.h_b_block_size,
        lut_chunk_size=args.lut_chunk_size,
        num_b_grid=1000,
        num_lambda_grid=500,
    )

    print("\n[Configuration]")
    print(f"  methods={cfg.methods_to_run}")
    print(f"  m={cfg.m_params}")
    print(f"  alpha={args.alpha} (target CI={cfg.target_cis})")
    print(f"  beta={cfg.berger_boos_beta} (nesting tolerance={cfg.nesting_tolerance})")
    print(f"  lambda={lambda_vals[0]}..{lambda_vals[-1]} (step={args.lambda_step})")
    print(f"  batch sizes: Bayes={cfg.chunk_size}, h={cfg.h_chunk_size}, "
          f"h-lambda={cfg.h_lambda_batch_size}, h-w={cfg.h_w_chunk_size}, "
          f"h-b-block={cfg.h_b_block_size}, "
          f"LUT={cfg.lut_chunk_size}")
    print("\n>>> Simulation started <<<")
    run_full_simulation("Bayesian", cfg)
    print("\n>>> All simulations completed <<<")


if __name__ == "__main__":
    main()
