import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator
from scipy.sparse import csr_matrix, linalg
from PIL import Image
from skimage.restoration import denoise_tv_chambolle
import os
import time

def solve_hjb_2d_final_clarity(R_max=60, alpha=1.04, sigma=0.05, h_grid=0.5):
    """
    Robust 2D HJB solver for high-clarity image contrast.
    """
    p = alpha / (alpha - 1)
    C_alpha = (alpha - 1) / (alpha ** p)
    
    ticks = np.arange(-R_max, R_max + h_grid, h_grid)
    Y1, Y2 = np.meshgrid(ticks, ticks, indexing='ij')
    Nx, Ny = Y1.shape
    
    r_val = np.sqrt(Y1**2 + Y2**2)
    inside = r_val < (R_max - 2 * h_grid)
    interior = np.zeros_like(inside, dtype=bool)
    for i in range(1, Nx - 1):
        for j in range(1, Ny - 1):
            if inside[i, j] and inside[i+1, j] and inside[i-1, j] and inside[i, j+1] and inside[i, j-1]:
                interior[i, j] = True
    
    interior_indices = np.where(interior)
    idx_map = -np.ones((Nx, Ny), dtype=int)
    for k, (i, j) in enumerate(zip(*interior_indices)):
        idx_map[i, j] = k
    N_int = len(interior_indices[0])
    
    data, rows, cols = [], [], []
    for k, (i, j) in enumerate(zip(*interior_indices)):
        data.append(4.0); rows.append(k); cols.append(k)
        for ni, nj in [(i+1, j), (i-1, j), (i, j+1), (i, j-1)]:
            if interior[ni, nj]:
                data.append(-1.0); rows.append(k); cols.append(idx_map[ni, nj])
    A = csr_matrix((data, (rows, cols)), shape=(N_int, N_int))
    
    # Boundary Potential - Matching g in radial code
    V_boundary = 0.5 
    h_y = np.log(r_val + 1.0)
    rhs_scale = (2.0 * h_grid**2) / (sigma**2)
    
    V = V_boundary * np.ones_like(Y1)
    print(f"Solving Precise HJB 2D (Grid: {Nx}x{Ny})...")
    st = time.time()
    
    # Use more iterations and a smaller relaxation for alpha=1.04
    for it in range(150):
        V_old = V.copy()
        dV1, dV2 = np.gradient(V_old, h_grid, h_grid, edge_order=2)
        grad_mag = np.sqrt(dV1**2 + dV2**2 + 1e-12)
        
        # Stability: The quasilinear term C_alpha * |grad V|^p can be huge.
        # We cap the normalized gradient mag to ensure it stays within a stable range.
        stable_grad = np.clip(grad_mag, 0, 1.2)
        grad_term = C_alpha * (stable_grad ** p)
        
        f = h_y - grad_term
        b_vec = np.zeros(N_int)
        for k, (i, j) in enumerate(zip(*interior_indices)):
            rhs = rhs_scale * f[i, j]
            # Neighbors for boundary
            for ni, nj in [(i+1, j), (i-1, j), (i, j+1), (i, j-1)]:
                if not interior[ni, nj]:
                    rhs += V_boundary
            b_vec[k] = rhs
            
        V_int = linalg.spsolve(A, b_vec)
        V_next = V.copy()
        V_next[interior] = V_int
        
        # Very smooth update for stability
        V = 0.3 * V_next + 0.7 * V_old
        
        if it % 20 == 0:
            diff = np.max(np.abs(V - V_old))
            print(f"Iteration {it}, diff: {diff:.2e}")
            if diff < 1e-6: break
            
    # Final Drift Lookup
    dV1, dV2 = np.gradient(V, h_grid, h_grid, edge_order=2)
    grad_mag = np.sqrt(dV1**2 + dV2**2 + 1e-12)
    p_mag = (grad_mag / alpha) ** (1.0/(alpha-1))
    p_mag = np.clip(p_mag, 0, 50.0)
    
    # To get perfect radial clarity, we return the scalar drift intensity
    drift_scalar_field = p_mag 
    return ticks, ticks, drift_scalar_field

def apply_enhancement(img_path, t1, t2, drift_field, T=0.2, dt=0.01):
    img = Image.open(img_path).convert("RGB")
    np_img = np.array(img).astype(np.float64)
    H, W, C = np_img.shape
    
    mean_val = np.mean(np_img, axis=(0,1), keepdims=True)
    centered = np_img - mean_val
    max_val = np.max(np.abs(centered))
    scale = 40.0 / max_val if max_val > 0 else 1.0
    x = centered * scale
    
    # Interp for the drift magnitude
    interp = RegularGridInterpolator((t1, t2), drift_field, method='cubic', bounds_error=False, fill_value=0.0)
    
    print("Processing pixels with PDE drift...")
    for _ in range(int(T/dt)):
        # Normalize radius in local coordinates
        r_sq = np.sum(x**2, axis=-1)
        r = np.sqrt(r_sq + 1e-12)
        
        # Map to 2D potential (using r as one axis for consistency)
        # Even if the PDE is solved on (y1, y2), for radial parity 
        # we can use any 2D projection or just the norm if it's radial.
        # But we'll follow the 2D PDE logic: lookup at (x, y) 
        coords = x[:, :, :2].reshape(-1, 2)
        drift_mag = interp(coords).reshape(H, W)
        
        # Consistent Radial Displacement (Repulsive for contrast)
        # x_new = x + dt * (drift_mag) * (x / r)
        # matches the logic u' / r in the radial code.
        drift = drift_mag[..., np.newaxis] * (x / r[..., np.newaxis])
        
        # Minimal stochasticity for sharpness
        noise = 0.005 * np.random.randn(*x.shape) * np.sqrt(dt)
        
        x = x + drift * dt + noise
        x = np.clip(x, -80, 80)
        
    final = x / scale + mean_val
    return np.clip(final, 0, 255).astype(np.uint8)

from skimage import exposure, img_as_ubyte

def compute_contrast_metrics(img):
    """
    Computes several metrics for image contrast and quality.
    - EME (Measure of Enhancment)
    - Tenengrad (Gradient-based sharpness)
    - Gray-Level Variance (General contrast)
    """
    if len(img.shape) == 3:
        gray = np.dot(img[...,:3], [0.2989, 0.5870, 0.1140])
    else:
        gray = img
        
    # 1. EME (Logarithmic Measure of Enhancement)
    # Splits image into 8x8 blocks and calculates contrast
    block_size = 8
    h, w = gray.shape
    eme = 0
    blocks_h, blocks_w = h // block_size, w // block_size
    for i in range(blocks_h):
        for j in range(blocks_w):
            block = gray[i*block_size:(i+1)*block_size, j*block_size:(j+1)*block_size]
            b_min, b_max = np.min(block), np.max(block)
            if b_min > 0 and b_max > 0:
                eme += 20 * np.log(b_max / b_min)
    eme /= (blocks_h * blocks_w)
    
    # 2. Tenengrad (Sharpness/Gradient)
    gx, gy = np.gradient(gray)
    tenengrad = np.mean(gx**2 + gy**2)
    
    # 3. Variance
    std_dev = np.std(gray)
    
    return {"EME": eme, "Tenengrad": tenengrad, "StdDev": std_dev}

if __name__ == "__main__":
    img_filepath = "c:/2026/Articole_2026/Trimis_23_November_2025_rendmat@lincei.it/23_Noiembrie_2025_rendmat@lincei.it/lenna.png"
    
    t1, t2, field = solve_hjb_2d_final_clarity()
    enhanced = apply_enhancement(img_filepath, t1, t2, field)
    
    # Baseline comparison: Histogram Equalization (HE)
    orig_img = Image.open(img_filepath).convert("RGB")
    orig_np = np.array(orig_img)
    he_img = exposure.equalize_hist(orig_np)
    he_img = (he_img * 255).astype(np.uint8)
    
    # Optimized TV denoise for HJB
    denoised = (denoise_tv_chambolle(enhanced.astype(float)/255.0, weight=0.04, channel_axis=-1)*255).astype(np.uint8)
    
    # Metrics Calculation
    m_orig = compute_contrast_metrics(orig_np)
    m_he = compute_contrast_metrics(he_img)
    m_hjb = compute_contrast_metrics(enhanced)
    m_denoised = compute_contrast_metrics(denoised)
    
    print("\nContrast Metrics Comparison:")
    print(f"{'Method':<15} | {'EME':<10} | {'Tenengrad':<10} | {'StdDev':<10}")
    print("-" * 55)
    print(f"{'Original':<15} | {m_orig['EME']:<10.2f} | {m_orig['Tenengrad']:<10.2f} | {m_orig['StdDev']:<10.2f}")
    print(f"{'Hist. Equal.':<15} | {m_he['EME']:<10.2f} | {m_he['Tenengrad']:<10.2f} | {m_he['StdDev']:<10.2f}")
    print(f"{'HJB 2D PDE':<15} | {m_hjb['EME']:<10.2f} | {m_hjb['Tenengrad']:<10.2f} | {m_hjb['StdDev']:<10.2f}")
    print(f"{'HJB + TV':<15} | {m_denoised['EME']:<10.2f} | {m_denoised['Tenengrad']:<10.2f} | {m_denoised['StdDev']:<10.2f}")

    # Visual Comparison
    plt.figure(figsize=(18, 6))
    plt.subplot(1, 4, 1); plt.imshow(orig_img); plt.title("Original")
    plt.subplot(1, 4, 2); plt.imshow(he_img); plt.title("Hist. Equalization")
    plt.subplot(1, 4, 3); plt.imshow(enhanced); plt.title("HJB 2D PDE Contrast")
    plt.subplot(1, 4, 4); plt.imshow(denoised); plt.title("HJB 2D + TV")
    
    plt.tight_layout()
    plt.savefig("hjb_comparison_metrics.png")
    plt.show()
