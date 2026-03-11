# -----------------------------------------------------------------------------
# Copyright (c) 2026 Dragos-Patru Covei
#
# All rights reserved.
#
# This software is provided for academic, research, and personal use only.
# Redistribution, modification, and use in source or binary forms are permitted
# for non-commercial purposes, provided that proper credit is given to the author.
#
# Commercial use of this software, in whole or in part, is strictly prohibited
# without prior written permission from the author. Any use of this code or its
# derivatives for financial gain, commercial products, paid services, or
# industrial applications requires an explicit licensing agreement.
#
# For commercial licensing inquiries, please contact the author directly.
#
# This implementation extends and improves the model introduced in:
#   Covei D. (2025), "coveidragos@yahoo.com" 
#
# The model was created solely by the author, and the development of this implementation was supported with assistance from 
# Microsoft Copilot in Edge.
# -----------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator
from scipy.sparse import csr_matrix, linalg
from PIL import Image
from skimage.restoration import denoise_tv_chambolle
from skimage import exposure
import os
import time


# -----------------------------
# Safe power (radial-inspired)
# -----------------------------
def safe_power(x, exponent, lower_bound=1e-8, upper_bound=1e2):
    """
    Clamp x to [lower_bound, upper_bound] and raise to 'exponent'.
    This avoids numerical overflow/underflow.
    """
    x_safe = np.clip(x, lower_bound, upper_bound)
    return np.power(x_safe, exponent)


# -----------------------------
# h(r): user-controllable cost
# -----------------------------
def h_radial(r, R_max, alpha):
    """
    Cost function h(y) = |y|^p, where r = |y|.
    Here p = alpha / (alpha - 1), as in the PDE.
    """
    p = alpha / (alpha - 1.0)
    r_safe = np.maximum(r, 0.0)
    return np.power(r_safe, p)


# -----------------------------
# 2D HJB solver
# -----------------------------
def solve_hjb_2d_final_clarity(R_max=60, alpha=2.0, sigma=0.05, h_grid=0.5):
    """
    2D HJB solver for non-radial contrast, with h handled in a radial-inspired way.

    PDE (discretized):
        -(sigma^2/2) ΔV + C_alpha |∇V|^p - h(y) = 0,  V = g on boundary

    We use:
        p       = alpha / (alpha - 1)
        C_alpha = (alpha - 1) / alpha^p
        h(y)    = h_radial(|y|)  (user-controllable, like in the radial case)
    """
    p = alpha / (alpha - 1.0)
    C_alpha = (alpha - 1.0) / (alpha ** p)

    # 2D grid in (y1, y2)
    ticks = np.arange(-R_max, R_max + h_grid, h_grid)
    Y1, Y2 = np.meshgrid(ticks, ticks, indexing='ij')
    Nx, Ny = Y1.shape

    # Radius in the (y1, y2) plane
    r_val = np.sqrt(Y1**2 + Y2**2)

    # Interior mask (all neighbors inside)
    inside = r_val < (R_max - 2.0 * h_grid)
    interior = np.zeros_like(inside, dtype=bool)
    for i in range(1, Nx - 1):
        for j in range(1, Ny - 1):
            if (inside[i, j] and inside[i+1, j] and inside[i-1, j]
                    and inside[i, j+1] and inside[i, j-1]):
                interior[i, j] = True

    interior_indices = np.where(interior)
    idx_map = -np.ones((Nx, Ny), dtype=int)
    for k, (i, j) in enumerate(zip(*interior_indices)):
        idx_map[i, j] = k
    N_int = len(interior_indices[0])

    # Laplacian matrix (5-point stencil)
    data, rows, cols = [], [], []
    for k, (i, j) in enumerate(zip(*interior_indices)):
        data.append(4.0); rows.append(k); cols.append(k)
        for ni, nj in [(i+1, j), (i-1, j), (i, j+1), (i, j-1)]:
            if interior[ni, nj]:
                data.append(-1.0)
                rows.append(k)
                cols.append(idx_map[ni, nj])
    A = csr_matrix((data, (rows, cols)), shape=(N_int, N_int))

    # h(y) defined via h_radial(|y|)
    h_y = h_radial(r_val, R_max, alpha)

    # Boundary condition
    V_boundary = 0.5
    rhs_scale = (2.0 * h_grid**2) / (sigma**2)

    # Initial guess for V
    V = V_boundary * np.ones_like(Y1)

    print(f"Solving HJB 2D (Grid {Nx}x{Ny}), alpha = {alpha}...")
    st = time.time()

    G_MAX_NONLIN = 3.0

    for it in range(150):
        V_old = V.copy()

        # Gradient of V
        dV1, dV2 = np.gradient(V_old, h_grid, h_grid, edge_order=2)
        grad_mag = np.sqrt(dV1**2 + dV2**2 + 1e-12)

        # Clip gradient magnitude to avoid overflow
        grad_mag_clipped = np.clip(grad_mag, 0.0, G_MAX_NONLIN)

        # Nonlinear term: C_alpha |∇V|^p, using safe_power
        grad_term = C_alpha * safe_power(grad_mag_clipped, p)

        # f = h - C_alpha |∇V|^p
        f = h_y - grad_term

        # Right-hand side
        b_vec = np.zeros(N_int)
        for k, (i, j) in enumerate(zip(*interior_indices)):
            rhs = rhs_scale * f[i, j]
            # boundary correction
            for ni, nj in [(i+1, j), (i-1, j), (i, j+1), (i, j-1)]:
                if not interior[ni, nj]:
                    rhs += V_boundary
            b_vec[k] = rhs

        # Solve linear system
        V_int = linalg.spsolve(A, b_vec)
        V_next = V.copy()
        V_next[interior] = V_int

        # Relaxation for stability
        V = 0.35 * V_next + 0.65 * V_old

        if it % 20 == 0:
            diff = np.max(np.abs(V - V_old))
            print(f"Iteration {it}, diff = {diff:.3e}")
            if not np.isfinite(diff):
                print("Non-finite diff detected, stopping iterations.")
                break
            if diff < 1e-6:
                break

    print(f"Done in {time.time() - st:.2f} s")

    # Drift magnitude (optimal control), radial-inspired
    dV1, dV2 = np.gradient(V, h_grid, h_grid, edge_order=2)
    grad_mag = np.sqrt(dV1**2 + dV2**2 + 1e-12)

    G_MAX_DRIFT = 2.0
    grad_mag_for_drift = np.clip(grad_mag, 0.0, G_MAX_DRIFT)

    exponent = 1.0 / (alpha - 1.0)
    A_drift = (1.0 / alpha) ** exponent
    drift_scalar = A_drift * safe_power(grad_mag_for_drift, exponent)
    drift_scalar = np.clip(drift_scalar, 0.0, 80.0)
    drift_scalar = np.nan_to_num(drift_scalar, nan=0.0, posinf=80.0, neginf=0.0)

    return ticks, ticks, drift_scalar


# -----------------------------
# Apply enhancement using drift
# -----------------------------
def apply_enhancement(img_path, t1, t2, drift_field, T=0.2, dt=0.01):
    img = Image.open(img_path).convert("RGB")
    np_img = np.array(img).astype(np.float64)
    H, W, C = np_img.shape

    # Center and scale in RGB space
    mean_val = np.mean(np_img, axis=(0, 1), keepdims=True)
    centered = np_img - mean_val
    max_val = np.max(np.abs(centered))
    scale = 40.0 / max_val if max_val > 0 else 1.0
    x = centered * scale

    # Interpolator for drift field in (R,G) plane
    interp = RegularGridInterpolator(
        (t1, t2), drift_field, method='cubic',
        bounds_error=False, fill_value=0.0
    )

    print("Applying PDE drift...")
    n_steps = int(T / dt)
    for _ in range(n_steps):
        r = np.sqrt(np.sum(x**2, axis=-1) + 1e-12)

        # Coordinates in (R,G) plane for lookup
        coords = x[:, :, :2].reshape(-1, 2)
        drift_mag = interp(coords).reshape(H, W)

        # Radial drift in RGB space (contrast-enhancing)
        drift = drift_mag[..., None] * (x / r[..., None])

        # Small noise for sharpness
        noise = 0.004 * np.random.randn(*x.shape) * np.sqrt(dt)

        x = x + drift * dt + noise
        x = np.clip(x, -80, 80)

    final = x / scale + mean_val
    return np.clip(final, 0, 255).astype(np.uint8)


# -----------------------------
# Contrast metrics
# -----------------------------
def compute_contrast_metrics(img):
    if len(img.shape) == 3:
        gray = np.dot(img[..., :3], [0.2989, 0.5870, 0.1140])
    else:
        gray = img

    block_size = 8
    h, w = gray.shape
    eme = 0.0
    blocks_h, blocks_w = h // block_size, w // block_size

    for i in range(blocks_h):
        for j in range(blocks_w):
            block = gray[i*block_size:(i+1)*block_size,
                         j*block_size:(j+1)*block_size]
            b_min, b_max = np.min(block), np.max(block)
            if b_min > 0 and b_max > 0:
                eme += 20.0 * np.log(b_max / b_min)
    eme /= (blocks_h * blocks_w)

    gx, gy = np.gradient(gray)
    tenengrad = np.mean(gx**2 + gy**2)
    std_dev = np.std(gray)

    return {"EME": eme, "Tenengrad": tenengrad, "StdDev": std_dev}


# -----------------------------
# Main script with alpha sweep
# -----------------------------
if __name__ == "__main__":
    img_filepath = "springer4.png"
    if not os.path.exists(img_filepath):
        img_filepath = "../springer4.png"

    if not os.path.exists(img_filepath):
        print("Error: springer4.png not found.")
    else:
        # Original image
        orig_img = Image.open(img_filepath).convert("RGB")
        orig_np = np.array(orig_img)

        # Histogram equalization
        he_img = (exposure.equalize_hist(orig_np) * 255).astype(np.uint8)

        # Baseline metrics
        m_orig = compute_contrast_metrics(orig_np)
        m_he = compute_contrast_metrics(he_img)

        print("\nBaseline metrics:")
        print(f"Original     | EME {m_orig['EME']:.2f} | Ten {m_orig['Tenengrad']:.2f} | Std {m_orig['StdDev']:.2f}")
        print(f"Hist. Equal. | EME {m_he['EME']:.2f} | Ten {m_he['Tenengrad']:.2f} | Std {m_he['StdDev']:.2f}")

        # Sweep over alpha values
        alpha_values = [1.02, 1.04, 1.06, 1.08, 1.10, 1.15, 1.20, 1.30, 1.50, 2.0]
        results_alpha = []

        for alpha in alpha_values:
            print(f"\n=== Alpha = {alpha} ===")
            t1, t2, field = solve_hjb_2d_final_clarity(alpha=alpha)

            enhanced = apply_enhancement(img_filepath, t1, t2, field)
            denoised = (denoise_tv_chambolle(
                enhanced.astype(float) / 255.0,
                weight=0.04,
                channel_axis=-1
            ) * 255).astype(np.uint8)

            m_hjb = compute_contrast_metrics(enhanced)
            m_denoised = compute_contrast_metrics(denoised)

            results_alpha.append({
                "alpha": alpha,
                "HJB": m_hjb,
                "HJB_TV": m_denoised
            })

            print(f"HJB 2D PDE  | EME {m_hjb['EME']:.2f} | Ten {m_hjb['Tenengrad']:.2f} | Std {m_hjb['StdDev']:.2f}")
            print(f"HJB + TV    | EME {m_denoised['EME']:.2f} | Ten {m_denoised['Tenengrad']:.2f} | Std {m_denoised['StdDev']:.2f}")

        # Summary: best alpha by EME and Tenengrad (HJB only)
        best_by_eme = max(results_alpha, key=lambda r: r["HJB"]["EME"])
        best_by_ten = max(results_alpha, key=lambda r: r["HJB"]["Tenengrad"])

        print("\n=== Summary vs baselines (HJB only) ===")
        print(f"Original     | EME {m_orig['EME']:.2f} | Ten {m_orig['Tenengrad']:.2f} | Std {m_orig['StdDev']:.2f}")
        print(f"Hist. Equal. | EME {m_he['EME']:.2f} | Ten {m_he['Tenengrad']:.2f} | Std {m_he['StdDev']:.2f}")
        print(f"\nBest alpha by EME (HJB): {best_by_eme['alpha']}, "
              f"EME = {best_by_eme['HJB']['EME']:.2f}, "
              f"Ten = {best_by_eme['HJB']['Tenengrad']:.2f}, "
              f"Std = {best_by_eme['HJB']['StdDev']:.2f}")
        print(f"Best alpha by Tenengrad (HJB): {best_by_ten['alpha']}, "
              f"EME = {best_by_ten['HJB']['EME']:.2f}, "
              f"Ten = {best_by_ten['HJB']['Tenengrad']:.2f}, "
              f"Std = {best_by_ten['HJB']['StdDev']:.2f}")

        # Optionally, show one representative result (e.g., best by EME)
        alpha_star = best_by_eme["alpha"]
        print(f"\nRecomputing and plotting for alpha* = {alpha_star} (best by EME)...")
        t1, t2, field = solve_hjb_2d_final_clarity(alpha=alpha_star)
        enhanced = apply_enhancement(img_filepath, t1, t2, field)
        denoised = (denoise_tv_chambolle(
            enhanced.astype(float) / 255.0,
            weight=0.04,
            channel_axis=-1
        ) * 255).astype(np.uint8)

        plt.figure(figsize=(18, 6))
        plt.subplot(1, 4, 1); plt.imshow(orig_img); plt.title("Original"); plt.axis("off")
        plt.subplot(1, 4, 2); plt.imshow(he_img); plt.title("Hist. Equalization"); plt.axis("off")
        plt.subplot(1, 4, 3); plt.imshow(enhanced); plt.title(f"HJB 2D PDE (alpha={alpha_star})"); plt.axis("off")
        plt.subplot(1, 4, 4); plt.imshow(denoised); plt.title(f"HJB 2D + TV (alpha={alpha_star})"); plt.axis("off")

        plt.tight_layout()
        plt.savefig("hjb_alpha_sweep_best.png", dpi=300)
        plt.show()
