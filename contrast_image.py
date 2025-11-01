"""
Image Restoration using HJB Equations (Color, no added noise)

- Processes lenna.png as a color image (RGB) in [0,1]
- Restores each channel via HJB PDE iteration
- Uses source term h(x,y) = ln(sqrt(x^2 + y^2) + 1.0)
- No synthetic noise added

Based on:
D.-P. Covei, "Image Restoration via the Integration of Optimal Control Techniques
and the Hamilton-Jacobi-Bellman Equation", Mathematics, 2025.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from skimage.restoration import denoise_tv_chambolle
import itertools
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (needed for 3D plots)

# -----------------------------
# Model parameters
# -----------------------------
N = 3
sigma_default = 0.05
R = 100.0
u0 = 297.79
alpha = 1.04
r0 = 0.01
r_shoot_end = R
rInc = 0.1

# -----------------------------
# Cost function
# -----------------------------
def h(r):
    return np.log(r+1)

# -----------------------------
# Safe power helper
# -----------------------------
def safe_power(u_prime, exponent, lower_bound=1e-8, upper_bound=1e2):
    safe_val = np.clip(-u_prime, lower_bound, upper_bound)
    return np.power(safe_val, exponent)

# -----------------------------
# Radial ODE
# -----------------------------
def value_function_ode(r, u):
    if abs(r) < 1e-6:
        r = 1e-6
    u_prime = u[1]
    if u_prime >= 0:
        u_prime = -1e-8
    A = (1/alpha) ** (1/(alpha - 1)) * ((alpha - 1) / alpha)
    exponent = alpha / (alpha - 1)
    term = A * safe_power(u_prime, exponent)
    du1 = u_prime
    du2 = - ((N - 1) / r) * u_prime + (2 / (sigma_default ** 2)) * (term - h(r))
    return [du1, du2]

# -----------------------------
# Solve ODE (value function, radial)
# -----------------------------
print("Starting ODE integration...")
r_values = np.arange(r0, r_shoot_end + rInc*0.1, rInc)
u_initial = [u0, -1e-6]
sol = solve_ivp(value_function_ode, [r0, r_shoot_end], u_initial,
                t_eval=r_values, method='BDF', rtol=1e-10, atol=1e-10)
if sol.success:
    print("ODE integration successful!")
else:
    raise RuntimeError("ODE integration failed")

# Boundary condition check (np.trapezoid instead of np.trapz)
v_values = -sol.y[1]
integral_v = np.trapezoid(v_values, sol.t)
g_boundary = u0 - integral_v
g_from_solution = sol.y[0][-1]
print("Computed boundary condition g (from integral):", g_boundary)
print("Computed boundary condition g (from ODE solution):", g_from_solution)

# -----------------------------
# Restoration dynamics (spatially varying noise)
# -----------------------------
def restore_image(image_initial, dt, T, sol, alpha, sigma):
    timesteps = int(np.ceil(T / dt))
    x = image_initial.copy()
    for _ in range(timesteps):
        r = np.linalg.norm(x, axis=-1)
        r_safe = np.where(r > 1e-6, r, 1e-6)
        u_prime_val = np.interp(r_safe, sol.t, sol.y[1])
        u_prime_val = np.where(u_prime_val >= 0, -1e-8, u_prime_val)
        restoration_rate_unit = ((1/alpha) ** (1/(alpha - 1))) * \
                                (1.0 / r_safe) * \
                                safe_power(u_prime_val, 1/(alpha - 1))
        drift = restoration_rate_unit[..., np.newaxis] * x
        # i.i.d. Gaussian noise per pixel and channel (spatial variation)
        noise = sigma * np.random.normal(0, np.sqrt(dt), x.shape)
        x = x + drift * dt + noise
        x = np.clip(x, -R, R)
    return x

# -----------------------------
# Load image
# -----------------------------
img = Image.open("lenna.png").convert("RGB")
img_array = np.array(img).astype(np.float64)

# -----------------------------
# Stage 1: TV denoising (on [0,1] range)
# -----------------------------
img_norm = img_array / 255.0
tv_weight = 0.089
img_denoised_norm = denoise_tv_chambolle(img_norm, weight=tv_weight, channel_axis=-1)
img_denoised = np.clip(img_denoised_norm * 255, 0, 255).astype(np.float64)

# -----------------------------
# Centering and scaling based on the denoised image
# -----------------------------
den_mean = np.mean(img_denoised, axis=(0, 1), keepdims=True)
den_centered = img_denoised - den_mean
max_abs = np.max(np.abs(den_centered))
scale_factor = (R/2) / max_abs if max_abs != 0 else 1.0
x_initial = den_centered * scale_factor  # initial condition is the denoised image

# -----------------------------
# Parameter tuning for restoration (starting from denoised)
# -----------------------------
sigma_values = [0.002, 0.007, 0.0189, 0.05]
T_values = [0.197, 1.0]
dt_values = [0.01, 0.17]
results = []

print("\nParameter Tuning Results (restoration after denoising):")
for sigma_param, T_param, dt_param in itertools.product(sigma_values, T_values, dt_values):
    restored_state = restore_image(x_initial, dt_param, T_param, sol, alpha, sigma_param)
    restored_from_denoised = restored_state / scale_factor + den_mean
    restored_from_denoised = np.clip(restored_from_denoised, 0, 255)
    # Metrics vs original degraded image
    mse_val = np.mean((img_array - restored_from_denoised) ** 2)
    psnr_val = peak_signal_noise_ratio(img_array.astype(np.uint8),
                                       restored_from_denoised.astype(np.uint8),
                                       data_range=255)
    ssim_val = structural_similarity(img_array.astype(np.uint8),
                                     restored_from_denoised.astype(np.uint8),
                                     channel_axis=-1, data_range=255)
    results.append({'sigma': sigma_param, 'T': T_param, 'dt': dt_param,
                    'MSE': mse_val, 'PSNR': psnr_val, 'SSIM': ssim_val})
    print(f"sigma={sigma_param}, T={T_param}, dt={dt_param} | "
          f"MSE: {mse_val:.4f}, PSNR: {psnr_val:.4f} dB, SSIM: {ssim_val:.4f}")

best_config = max(results, key=lambda r: r['PSNR'])
print("\nBest configuration based on PSNR (restoration after denoising):")
print(best_config)

# -----------------------------
# Final restoration starting from the denoised image
# -----------------------------
restored_state_best = restore_image(x_initial, best_config['dt'], best_config['T'],
                                    sol, alpha, best_config['sigma'])
restored_from_denoised_best = restored_state_best / scale_factor + den_mean
restored_from_denoised_best = np.clip(restored_from_denoised_best, 0, 255).astype(np.uint8)

# -----------------------------
# Display 2D images (original, denoised, restored-from-denoised)
# -----------------------------
plt.figure(figsize=(18, 6))
plt.subplot(1, 3, 1)
plt.imshow(img_array.astype(np.uint8))
plt.title("(i) Original Degraded Image")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(img_denoised.astype(np.uint8))
plt.title("(ii) TV Denoised Image (Stage 1)")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(restored_from_denoised_best)
plt.title("(iii) Restored From Denoised (Stage 2)")
plt.axis("off")
plt.tight_layout()
plt.show()

# -----------------------------
# 3D difference surfaces (diagnostics)
# -----------------------------
def rgb2gray(img):
    return np.dot(img[..., :3], [0.2989, 0.5870, 0.1140])

original_gray = rgb2gray(img_array.astype(np.float64))
denoised_gray = rgb2gray(img_denoised.astype(np.float64))
restored_gray = rgb2gray(restored_from_denoised_best.astype(np.float64))

diff_denoised_restored = np.abs(denoised_gray - restored_gray)
diff_original_denoised = np.abs(original_gray - denoised_gray)
diff_original_restored = np.abs(original_gray - restored_gray)

rows, cols = diff_denoised_restored.shape
X, Y = np.meshgrid(np.arange(cols), np.arange(rows))

fig3 = plt.figure(figsize=(24, 8))

ax1 = fig3.add_subplot(1, 3, 1, projection='3d')
surf1 = ax1.plot_surface(X, Y, diff_denoised_restored, cmap='plasma', alpha=0.85, linewidth=0)
ax1.set_title("3D Surface: Diff (i) Denoised vs Restored")
ax1.set_xlabel("Pixel Column")
ax1.set_ylabel("Pixel Row")
ax1.set_zlabel("Difference Intensity")
fig3.colorbar(surf1, ax=ax1, shrink=0.6, aspect=12)

ax2 = fig3.add_subplot(1, 3, 2, projection='3d')
surf2 = ax2.plot_surface(X, Y, diff_original_denoised, cmap='coolwarm', alpha=0.85, linewidth=0)
ax2.set_title("3D Surface: Diff (ii) Original vs Denoised")
ax2.set_xlabel("Pixel Column")
ax2.set_ylabel("Pixel Row")
ax2.set_zlabel("Difference Intensity")
fig3.colorbar(surf2, ax=ax2, shrink=0.6, aspect=12)

ax3 = fig3.add_subplot(1, 3, 3, projection='3d')
surf3 = ax3.plot_surface(X, Y, diff_original_restored, cmap='inferno', alpha=0.85, linewidth=0)
ax3.set_title("3D Surface: Diff (iii) Original vs Restored From Denoised")
ax3.set_xlabel("Pixel Column")
ax3.set_ylabel("Pixel Row")
ax3.set_zlabel("Difference Intensity")
fig3.colorbar(surf3, ax=ax3, shrink=0.6, aspect=12)

plt.tight_layout()
plt.show()
