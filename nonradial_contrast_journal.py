import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator
from scipy.sparse import csr_matrix, linalg
from PIL import Image
from skimage.restoration import denoise_tv_chambolle
from skimage import exposure
import os
import time


def solve_hjb_2d_final_clarity(R_max=60, alpha=1.04, sigma=0.05, h_grid=0.5):
    """
    2D HJB solver stabil numeric pentru contrast non-radial.

    Ecuația (în formă discretizată):
        -(sigma^2/2) ΔV + C_alpha |∇V|^p - h(y) = 0,  V = g pe frontieră

    Alegem:
        p       = alpha / (alpha - 1)
        C_alpha = (alpha - 1) / alpha^p
        h(y)    = (|y| / R_max)^p  (normalizat în [0,1])
    """
    p = alpha / (alpha - 1.0)
    C_alpha = (alpha - 1.0) / (alpha ** p)

    # Grid 2D
    ticks = np.arange(-R_max, R_max + h_grid, h_grid)
    Y1, Y2 = np.meshgrid(ticks, ticks, indexing='ij')
    Nx, Ny = Y1.shape

    # Radial
    r_val = np.sqrt(Y1**2 + Y2**2)
    r_norm = r_val / R_max  # în [0,1]
    inside = r_val < (R_max - 2.0 * h_grid)

    # Puncte interioare (toți vecinii în interior)
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

    # Matricea Laplacianului (5-puncte)
    data, rows, cols = [], [], []
    for k, (i, j) in enumerate(zip(*interior_indices)):
        data.append(4.0); rows.append(k); cols.append(k)
        for ni, nj in [(i+1, j), (i-1, j), (i, j+1), (i, j-1)]:
            if interior[ni, nj]:
                data.append(-1.0)
                rows.append(k)
                cols.append(idx_map[ni, nj])
    A = csr_matrix((data, (rows, cols)), shape=(N_int, N_int))

    # h(y) = (|y|/R_max)^p  => valori în [0,1]
    # h_y = (r_norm ** 2) * (r_norm ** (p - 2))
    # h_y = r_norm ** p
    # h_y = (r_norm ** p) / (1 + r_norm ** p)
    h_y = (r_norm ** p) / (1 + r_norm ** p)


    # Condiție de frontieră simplă
    V_boundary = 0.5
    rhs_scale = (2.0 * h_grid**2) / (sigma**2)

    # Inițializare V
    V = V_boundary * np.ones_like(Y1)

    print(f"Solving HJB 2D (Grid {Nx}x{Ny})...")
    st = time.time()

    # Limită pentru |∇V| în termenul neliniar
    G_MAX_NONLIN = 3.0

    for it in range(150):
        V_old = V.copy()

        # Gradient numeric
        dV1, dV2 = np.gradient(V_old, h_grid, h_grid, edge_order=2)
        grad_mag = np.sqrt(dV1**2 + dV2**2 + 1e-12)

        # Tăiem gradientul pentru a evita overflow la grad_mag**p
        grad_mag_clipped = np.clip(grad_mag, 0.0, G_MAX_NONLIN)
        grad_term = C_alpha * (grad_mag_clipped ** p)

        # f = h - C_alpha |∇V|^p
        f = h_y - grad_term

        # Termul liber
        b_vec = np.zeros(N_int)
        for k, (i, j) in enumerate(zip(*interior_indices)):
            rhs = rhs_scale * f[i, j]
            # corecție de frontieră
            for ni, nj in [(i+1, j), (i-1, j), (i, j+1), (i, j-1)]:
                if not interior[ni, nj]:
                    rhs += V_boundary
            b_vec[k] = rhs

        # Rezolvăm sistemul liniar
        V_int = linalg.spsolve(A, b_vec)
        V_next = V.copy()
        V_next[interior] = V_int

        # Relaxare pentru stabilitate
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

    # Drift optim (magnitudine)
    dV1, dV2 = np.gradient(V, h_grid, h_grid, edge_order=2)
    grad_mag = np.sqrt(dV1**2 + dV2**2 + 1e-12)

    # Limităm gradientul și aici, ca să nu mai apară overflow în power
    G_MAX_DRIFT = 2.0
    grad_mag_for_drift = np.clip(grad_mag, 0.0, G_MAX_DRIFT)

    drift_scalar = (grad_mag_for_drift / alpha) ** (1.0 / (alpha - 1.0))
    drift_scalar = np.clip(drift_scalar, 0, 80.0)

    # Curățăm orice non-finite (de siguranță)
    drift_scalar = np.nan_to_num(drift_scalar, nan=0.0, posinf=80.0, neginf=0.0)

    return ticks, ticks, drift_scalar


def apply_enhancement(img_path, t1, t2, drift_field, T=0.2, dt=0.01):
    img = Image.open(img_path).convert("RGB")
    np_img = np.array(img).astype(np.float64)
    H, W, C = np_img.shape

    # Centrare și scalare în spațiul RGB
    mean_val = np.mean(np_img, axis=(0, 1), keepdims=True)
    centered = np_img - mean_val
    max_val = np.max(np.abs(centered))
    scale = 40.0 / max_val if max_val > 0 else 1.0
    x = centered * scale

    # Interpolator pentru câmpul de drift
    interp = RegularGridInterpolator(
        (t1, t2), drift_field, method='cubic',
        bounds_error=False, fill_value=0.0
    )

    print("Applying PDE drift...")
    n_steps = int(T / dt)
    for _ in range(n_steps):
        r = np.sqrt(np.sum(x**2, axis=-1) + 1e-12)

        # Coordonate în planul (R,G) pentru lookup
        coords = x[:, :, :2].reshape(-1, 2)
        drift_mag = interp(coords).reshape(H, W)

        # Drift radial (repulsiv pentru contrast)
        drift = drift_mag[..., None] * (x / r[..., None])

        # Zgomot mic pentru „sharpness”
        noise = 0.004 * np.random.randn(*x.shape) * np.sqrt(dt)

        x = x + drift * dt + noise
        x = np.clip(x, -80, 80)

    final = x / scale + mean_val
    return np.clip(final, 0, 255).astype(np.uint8)


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


if __name__ == "__main__":
    # Căutăm imaginea în directorul curent sau părinte
    img_filepath = "lincei.png"
    if not os.path.exists(img_filepath):
        img_filepath = "../lincei.png"

    if not os.path.exists(img_filepath):
        print("Error: lincei.png not found.")
    else:
        # Rezolvăm HJB și obținem câmpul de drift
        t1, t2, field = solve_hjb_2d_final_clarity()

        # Aplicăm îmbunătățirea de contrast
        enhanced = apply_enhancement(img_filepath, t1, t2, field)

        # Imagine originală
        orig_img = Image.open(img_filepath).convert("RGB")
        orig_np = np.array(orig_img)

        # Histograma egalizată
        he_img = (exposure.equalize_hist(orig_np) * 255).astype(np.uint8)

        # Denoising TV peste rezultatul HJB
        denoised = (denoise_tv_chambolle(
            enhanced.astype(float) / 255.0,
            weight=0.04,
            channel_axis=-1
        ) * 255).astype(np.uint8)

        # Metrici
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

        # Figură comparativă
        plt.figure(figsize=(18, 6))
        plt.subplot(1, 4, 1); plt.imshow(orig_img); plt.title("Original"); plt.axis("off")
        plt.subplot(1, 4, 2); plt.imshow(he_img); plt.title("Hist. Equalization"); plt.axis("off")
        plt.subplot(1, 4, 3); plt.imshow(enhanced); plt.title("HJB 2D PDE Contrast"); plt.axis("off")
        plt.subplot(1, 4, 4); plt.imshow(denoised); plt.title("HJB 2D + TV"); plt.axis("off")

        plt.tight_layout()
        plt.savefig("hjb_comparison_metrics.png", dpi=300)
        plt.show()
