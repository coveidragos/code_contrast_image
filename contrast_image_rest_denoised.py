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
sigma = 0.05
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
    du2 = - ((N - 1) / r) * u_prime + (2 / (sigma ** 2)) * (term - h(r))
    return [du1, du2]

# -----------------------------
# Solve ODE
# -----------------------------
print("Starting ODE integration\ldots")
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
# Restore image (spatially varying noise)
# -----------------------------
def restore_image(image_initial, dt, T, sol, alpha, sigma):
    timesteps = int(T / dt)
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
# Load and normalize image
# -----------------------------
img = Image.open("lenna.png").convert("RGB")
img_array = np.array(img).astype(np.float64)
img_mean = np.mean(img_array, axis=(0, 1), keepdims=True)
img_centered = img_array - img_mean
max_abs = np.max(np.abs(img_centered))
scale_factor = (R/2) / max_abs if max_abs != 0 else 1
x_initial = img_centered * scale_factor

# -----------------------------
# Parameter tuning
# -----------------------------
sigma_values = [0.002, 0.007, 0.0189, 0.05]
T_values = [0.197, 1.0]
dt_values = [0.01, 0.17]
results = []

print("\nParameter Tuning Results:")
for sigma_param, T_param, dt_param in itertools.product(sigma_values, T_values, dt_values):
    restored_state = restore_image(x_initial, dt_param, T_param, sol, alpha, sigma_param)
    restored_image = restored_state / scale_factor + img_mean
    restored_image = np.clip(restored_image, 0, 255)
    mse_val = np.mean((img_array - restored_image) ** 2)
    psnr_val = peak_signal_noise_ratio(img_array.astype(np.uint8),
                                       restored_image.astype(np.uint8),
                                       data_range=255)
    ssim_val = structural_similarity(img_array.astype(np.uint8),
                                     restored_image.astype(np.uint8),
                                     channel_axis=-1, data_range=255)
    results.append({'sigma': sigma_param, 'T': T_param, 'dt': dt_param,
                    'MSE': mse_val, 'PSNR': psnr_val, 'SSIM': ssim_val})
    print(f"MSE: {mse_val:.4f}, PSNR: {psnr_val:.4f} dB, SSIM: {ssim_val:.4f}")

best_config = max(results, key=lambda r: r['PSNR'])
print("\nBest configuration based on PSNR:")
print(best_config)

# -----------------------------
# Final restoration and TV denoising
# -----------------------------
restored_state_best = restore_image(x_initial, best_config['dt'], best_config['T'],
                                    sol, alpha, best_config['sigma'])
restored_image_best = restored_state_best / scale_factor + img_mean
restored_image_best = np.clip(restored_image_best, 0, 255).astype(np.uint8)

restored_image_norm = restored_image_best.astype(np.float64) / 255.0
tv_weight = 0.089
restored_image_denoised_norm = denoise_tv_chambolle(restored_image_norm,
                                                    weight=tv_weight,
                                                    channel_axis=-1)
restored_image_denoised = np.clip(restored_image_denoised_norm * 255, 0, 255).astype(np.uint8)

# -----------------------------
# Display 2D images
# -----------------------------
plt.figure(figsize=(18, 6))
plt.subplot(1, 3, 1)
plt.imshow(img_array.astype(np.uint8))
plt.title("(i) Original Degraded Image")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(restored_image_best)
plt.title("(ii) Restored Image")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(restored_image_denoised)
plt.title("(iii) Restored and Denoised Image")
plt.axis("off")
plt.tight_layout()
plt.show()

# -----------------------------
# 3D difference surfaces (as in the paper)
# -----------------------------
def rgb2gray(img):
    return np.dot(img[..., :3], [0.2989, 0.5870, 0.1140])


original_gray = rgb2gray(img_array.astype(np.float64))
restored_gray = rgb2gray(restored_image_best.astype(np.float64))
denoised_gray = rgb2gray(restored_image_denoised.astype(np.float64))

diff_restored_denoised = np.abs(restored_gray - denoised_gray)
diff_original_restored = np.abs(original_gray - restored_gray)
diff_original_denoised = np.abs(original_gray - denoised_gray)

rows, cols = diff_restored_denoised.shape
X, Y = np.meshgrid(np.arange(cols), np.arange(rows))

fig3 = plt.figure(figsize=(24, 8))

ax1 = fig3.add_subplot(1, 3, 1, projection='3d')
surf1 = ax1.plot_surface(X, Y, diff_restored_denoised, cmap='plasma', alpha=0.85, linewidth=0)
ax1.set_title("3D Surface: Diff (i) Restored vs Denoised")
ax1.set_xlabel("Pixel Column")
ax1.set_ylabel("Pixel Row")
ax1.set_zlabel("Difference Intensity")
fig3.colorbar(surf1, ax=ax1, shrink=0.6, aspect=12)

ax2 = fig3.add_subplot(1, 3, 2, projection='3d')
surf2 = ax2.plot_surface(X, Y, diff_original_restored, cmap='coolwarm', alpha=0.85, linewidth=0)
ax2.set_title("3D Surface: Diff (ii) Original vs Restored")
ax2.set_xlabel("Pixel Column")
ax2.set_ylabel("Pixel Row")
ax2.set_zlabel("Difference Intensity")
fig3.colorbar(surf2, ax=ax2, shrink=0.6, aspect=12)

ax3 = fig3.add_subplot(1, 3, 3, projection='3d')
surf3 = ax3.plot_surface(X, Y, diff_original_denoised, cmap='inferno', alpha=0.85, linewidth=0)
ax3.set_title("3D Surface: Diff (iii) Original vs Restored & Denoised")
ax3.set_xlabel("Pixel Column")
ax3.set_ylabel("Pixel Row")
ax3.set_zlabel("Difference Intensity")
fig3.colorbar(surf3, ax=ax3, shrink=0.6, aspect=12)

plt.tight_layout()
plt.show()