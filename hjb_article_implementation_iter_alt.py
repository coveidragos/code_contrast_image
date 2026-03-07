import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator
from scipy.sparse import csr_matrix, linalg
import time


# ----------------------------------------------------------------------
# Geometry helpers
# ----------------------------------------------------------------------

def build_interior_mask(inside):
    Nx, Ny = inside.shape
    interior = np.zeros_like(inside, dtype=bool)
    for i in range(1, Nx - 1):
        for j in range(1, Ny - 1):
            if (inside[i, j] and inside[i+1, j] and inside[i-1, j] and
                inside[i, j+1] and inside[i, j-1]):
                interior[i, j] = True
    return interior


def compute_torsion_function(X, Y, inside, h_grid):
    """
    Solve -Δφ = 1 in Ω, φ = 0 on ∂Ω using a 5-point stencil.
    """
    Nx, Ny = X.shape
    interior = build_interior_mask(inside)
    interior_indices = np.where(interior)
    idx_map = -np.ones((Nx, Ny), dtype=int)
    for k, (i, j) in enumerate(zip(*interior_indices)):
        idx_map[i, j] = k
    N_int = len(interior_indices[0])

    inv_h2 = 1.0 / (h_grid ** 2)
    data, rows, cols = [], [], []
    b = np.ones(N_int)

    for k, (i, j) in enumerate(zip(*interior_indices)):
        diag = 4.0 * inv_h2
        neighbors = [
            (i+1, j, -inv_h2),
            (i-1, j, -inv_h2),
            (i, j+1, -inv_h2),
            (i, j-1, -inv_h2)
        ]
        for ni, nj, coeff in neighbors:
            if interior[ni, nj]:
                data.append(coeff)
                rows.append(k)
                cols.append(idx_map[ni, nj])
        data.append(diag)
        rows.append(k)
        cols.append(k)

    L = csr_matrix((data, (rows, cols)), shape=(N_int, N_int))
    phi_int = linalg.spsolve(L, b)

    phi = np.zeros_like(X)
    phi[interior] = phi_int
    phi[~inside] = 0.0
    return phi


# ----------------------------------------------------------------------
# Main solver: weighted monotone iteration (correct Lasry–Lions scheme)
# ----------------------------------------------------------------------

def solve_hjb_policy_iteration(a=1.5, b_ellipse=1.0, sigma=0.3, alpha=1.5,
                               g=0.5, h_grid=0.03, max_iter=200, tol=5e-3):
    """
    Solves the quasilinear HJB equation
        - (sigma^2/2) ΔV + C_alpha |∇V|^p - h(y) = 0  in Ω,
        V = g on ∂Ω,
    using the weighted monotone iteration from the article.

    ***Correct scheme (Lasry–Lions):***
    - (σ²/2) Δ V^{k+1} + Λ₀ V^{k+1}
        = h + Λ₀ V^{k} - Cα |∇V^{k}|^p

    Interface identical cu codul tău original.
    """
    # Conjugate parameters
    p = alpha / (alpha - 1.0)
    C_alpha = (alpha - 1.0) / (alpha ** p)

    # Grid
    y1_vals = np.arange(-a - h_grid, a + h_grid + h_grid, h_grid)
    y2_vals = np.arange(-b_ellipse - h_grid, b_ellipse + h_grid + h_grid, h_grid)
    X, Y = np.meshgrid(y1_vals, y2_vals, indexing='ij')
    Nx, Ny = X.shape

    # Elliptical domain
    inside = ((X / a) ** 2 + (Y / b_ellipse) ** 2) < 1.0

    # Interior mask
    interior = build_interior_mask(inside)
    interior_indices = np.where(interior)
    idx_map = -np.ones((Nx, Ny), dtype=int)
    for k, (i, j) in enumerate(zip(*interior_indices)):
        idx_map[i, j] = k
    N_int = len(interior_indices[0])

    # Source term h(y)
    h_y = X**2 + Y**2
    H = np.max(h_y)

    # Torsion function φ
    phi = compute_torsion_function(X, Y, inside, h_grid)

    # Barriers: V_- = g, V_+ = g + B φ
    B = 2.0 * H / (sigma**2)
    V_minus = g * np.ones_like(X)
    V_plus = g + B * phi

    # Initial iterate
    V = V_plus.copy()

    # Gradient bound M from barriers
    dphi_x, dphi_y = np.gradient(phi, h_grid, h_grid)
    M = np.max(B * np.sqrt(dphi_x**2 + dphi_y**2))
    M = max(M, 1e-6)

    # Poincaré constant (upper bound)
    diam = 2.0 * max(a, b_ellipse)
    C_omega = (diam**2) / (np.pi**2)

    # Λ0 from article
    Lambda0 = C_alpha * p * (M**(p-1)) * C_omega
    Lambda0 = max(Lambda0, 10.0)

    print(f"Solving HJB with correct +Lambda0, alpha={alpha:.2f} on {X.shape} grid...")
    print(f"H = {H:.4e}, B = {B:.4e}, M = {M:.4e}, Lambda0 = {Lambda0:.4e}")
    st = time.time()

    # Discretization constants
    s2 = sigma**2 / 2.0
    inv_h2 = 1.0 / (h_grid**2)

    for it in range(max_iter):
        V_old = V.copy()

        # Projection on barriers
        V = np.minimum(V, V_plus)
        V = np.maximum(V, V_minus)

        # Gradient and nonlinear term
        dV_dy1, dV_dy2 = np.gradient(V, h_grid, h_grid, edge_order=2)
        grad_sq = np.minimum(dV_dy1**2 + dV_dy2**2, 1e4)
        grad_p = grad_sq**(p/2.0)

        # *** Correct weighted iteration ***
        # - (σ²/2) Δ V^{k+1} + Λ₀ V^{k+1}
        #     = h + Λ₀ V^{k} - Cα |∇V^{k}|^p
        rhs_f = h_y + Lambda0 * V - C_alpha * grad_p

        data, rows, col = [], [], []
        b_vec = np.zeros(N_int)

        # Linear operator: - (σ²/2) Δ + Λ₀ I
        for k, (i, j) in enumerate(zip(*interior_indices)):
            rhs_val = rhs_f[i, j]
            diag = 4.0 * s2 * inv_h2 + Lambda0

            neighbors = [
                (i+1, j, -s2*inv_h2),
                (i-1, j, -s2*inv_h2),
                (i, j+1, -s2*inv_h2),
                (i, j-1, -s2*inv_h2)
            ]

            for ni, nj, coeff in neighbors:
                if interior[ni, nj]:
                    data.append(coeff)
                    rows.append(k)
                    col.append(idx_map[ni, nj])
                else:
                    rhs_val -= coeff * g  # boundary V = g

            data.append(diag)
            rows.append(k)
            col.append(k)
            b_vec[k] = rhs_val

        L = csr_matrix((data, (rows, col)), shape=(N_int, N_int))
        V_int = linalg.spsolve(L, b_vec)

        V[interior] = V_int
        V[~inside] = g

        diff = np.max(np.abs(V - V_old))
        print(f"Iter {it+1:3d}: diff = {diff:.3e}")
        if diff < tol:
            print(f"Weighted iteration converged after {it+1} steps (diff={diff:.2e})")
            break
    else:
        print("Warning: weighted iteration did not reach tol within max_iter.")

    print(f"Solver finished in {time.time() - st:.2f} seconds.")

    # Final gradients
    dV_dy1, dV_dy2 = np.gradient(V, h_grid, h_grid, edge_order=2)
    grad_norm = np.sqrt(dV_dy1**2 + dV_dy2**2)

    # Optimal control (same formula as în codul vechi)
    with np.errstate(divide='ignore', invalid='ignore'):
        factor = - (1.0 / (alpha ** (p-1))) * (grad_norm ** (p-2))
        factor[np.isinf(factor)] = 0
        v1 = factor * dV_dy1
        v2 = factor * dV_dy2

    v1[np.isnan(v1)] = 0
    v2[np.isnan(v2)] = 0
    v1[~inside] = 0
    v2[~inside] = 0

    return X, Y, V, grad_norm, v1, v2, inside, y1_vals, y2_vals


# ----------------------------------------------------------------------
# Dynamics simulation (unchanged)
# ----------------------------------------------------------------------

def simulate_dynamics(v1, v2, y1_vals, y2_vals, a, b, sigma,
                      dt=0.01, T=10.0, x0=[0.0, 0.0]):
    """Simulates dX = v*(X) dt + sigma dW."""
    v1_interp = RegularGridInterpolator((y1_vals, y2_vals), v1,
                                        bounds_error=False, fill_value=0.0)
    v2_interp = RegularGridInterpolator((y1_vals, y2_vals), v2,
                                        bounds_error=False, fill_value=0.0)

    traj = [np.array(x0)]
    ts = [0.0]
    curr_t = 0.0
    margin = 0.05

    while curr_t < T:
        pos = traj[-1]
        if (pos[0]/a)**2 + (pos[1]/b)**2 >= 1.0 - margin:
            break
        c1 = v1_interp(pos).item()
        c2 = v2_interp(pos).item()
        noise = sigma * np.random.randn(2) * np.sqrt(dt)
        next_pos = pos + np.array([c1, c2]) * dt + noise
        traj.append(next_pos)
        curr_t += dt
        ts.append(curr_t)

    return np.array(traj), np.array(ts)


# ----------------------------------------------------------------------
# Main script (same I/O and plots as original)
# ----------------------------------------------------------------------

if __name__ == "__main__":
    a, b = 1.5, 1.0
    sigma = 0.3
    alpha = 2.0   # pentru α=2 vei obține soluția concavă corectă
    g = 0.5
    h_grid = 0.03

    X, Y, V, Magn, v1, v2, inside, y1_vals, y2_vals = solve_hjb_policy_iteration(
        a, b, sigma, alpha, g, h_grid
    )

    plt.rcParams.update({'font.size': 10, 'font.family': 'sans-serif'})

    fig = plt.figure(figsize=(18, 10))

    # 1. Value Function
    ax1 = fig.add_subplot(221, projection='3d')
    V_plot = np.where(inside, V, np.nan)
    surf1 = ax1.plot_surface(X, Y, V_plot, cmap='magma',
                             edgecolor='none', alpha=0.9, antialiased=True)
    ax1.set_title(r"Value Function $V(y)$", fontsize=14)
    ax1.set_xlabel(r"$y_1$")
    ax1.set_ylabel(r"$y_2$")
    fig.colorbar(surf1, ax=ax1, shrink=0.5, aspect=10)

    # 2. Optimal Control v1*
    ax2 = fig.add_subplot(223, projection='3d')
    v1_plot = np.where(inside, v1, np.nan)
    surf2 = ax2.plot_surface(X, Y, v1_plot, cmap='coolwarm',
                             edgecolor='none', alpha=0.9)
    ax2.set_title(r"Optimal Control $v_1^*(y)$", fontsize=14)
    ax2.set_xlabel(r"$y_1$")
    ax2.set_ylabel(r"$y_2$")
    fig.colorbar(surf2, ax=ax2, shrink=0.5, aspect=10)

    # 3. Optimal Control v2*
    ax3 = fig.add_subplot(224, projection='3d')
    v2_plot = np.where(inside, v2, np.nan)
    surf3 = ax3.plot_surface(X, Y, v2_plot, cmap='coolwarm',
                             edgecolor='none', alpha=0.9)
    ax3.set_title(r"Optimal Control $v_2^*(y)$", fontsize=14)
    ax3.set_xlabel(r"$y_1$")
    ax3.set_ylabel(r"$y_2$")
    fig.colorbar(surf3, ax=ax3, shrink=0.5, aspect=10)

    # 4. Stochastic Trajectory
    traj, times = simulate_dynamics(v1, v2, y1_vals, y2_vals, a, b, sigma,
                                    x0=[0.5, 0.5])
    ax4 = fig.add_subplot(222)
    theta = np.linspace(0, 2*np.pi, 200)
    ax4.plot(a*np.cos(theta), b*np.sin(theta), 'k--', alpha=0.5, label="Boundary")
    ax4.plot(traj[:, 0], traj[:, 1], 'C0-', lw=1.5, label="Trajectory")
    ax4.scatter(traj[0, 0], traj[0, 1], c='green', s=40, label="Start")
    ax4.scatter(traj[-1, 0], traj[-1, 1], c='red', s=40, label="End/Exit")
    ax4.set_title("Simulated Inventory Trajectory", fontsize=14)
    ax4.set_xlabel(r"$y_1$")
    ax4.set_ylabel(r"$y_2$")
    ax4.grid(True, linestyle=':', alpha=0.6)
    ax4.legend()

    plt.tight_layout()
    plt.savefig("hjb_production_planning.png", dpi=300)
    plt.show()

    print(f"Value range: {np.nanmin(V_plot):.3f} to {np.nanmax(V_plot):.3f}")
    print("Main production planning figure saved as 'hjb_production_planning.png'.")
