import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator
from scipy.sparse import csr_matrix, linalg
import time
from matplotlib.lines import Line2D


# ----------------------------------------------------------------------
# Geometry helpers
# ----------------------------------------------------------------------

def build_interior_mask(inside):
    """
    Identify interior points that have all four neighbors inside the domain.
    This avoids including boundary-adjacent points in the PDE discretization.
    """
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
    Compute the torsion function φ solving:
        -Δφ = 1 in Ω
         φ = 0 on ∂Ω
    using a finite-difference discretization on the interior points.
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
    b = np.ones(N_int)  # RHS = 1

    # Build Laplacian matrix
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
# Main solver
# ----------------------------------------------------------------------

def solve_hjb_policy_iteration(a=1.5, b_ellipse=1.0, sigma=0.3, alpha=1.5,
                               g=0.5, h_grid=0.03, max_iter=200, tol=5e-3):
    """
    Solve the stationary HJB equation using a monotone policy iteration scheme.
    The PDE is:
        -(σ²/2) ΔV + Cα |∇V|^p - h = 0
        V = g on ∂Ω
    """

    # Exponent p and constant C_alpha
    p = alpha / (alpha - 1.0)
    C_alpha = (alpha - 1.0) / (alpha ** p)

    # Build computational grid
    y1_vals = np.arange(-a - h_grid, a + h_grid + h_grid, h_grid)
    y2_vals = np.arange(-b_ellipse - h_grid, b_ellipse + h_grid + h_grid, h_grid)
    X, Y = np.meshgrid(y1_vals, y2_vals, indexing='ij')
    Nx, Ny = X.shape

    # Elliptical domain indicator
    inside = ((X / a) ** 2 + (Y / b_ellipse) ** 2) < 1.0

    # Interior mask for PDE discretization
    interior = build_interior_mask(inside)
    interior_indices = np.where(interior)
    idx_map = -np.ones((Nx, Ny), dtype=int)

    for k, (i, j) in enumerate(zip(*interior_indices)):
        idx_map[i, j] = k

    N_int = len(interior_indices[0])

    # Running cost h(y) = |y|^p
    r = np.sqrt(X**2 + Y**2)
    h_y = r**p
    H = np.max(h_y)

    # Compute torsion function φ
    phi = compute_torsion_function(X, Y, inside, h_grid)

    # Compute gradient of φ
    dphi_x, dphi_y = np.gradient(phi, h_grid, h_grid)
    grad_phi_norm = np.sqrt(dphi_x**2 + dphi_y**2)
    M_phi = np.max(grad_phi_norm)

    # Constants B and C1
    B = 2.0 * H / (sigma**2)
    C1 = B * M_phi

    # Lower and upper barriers
    V_minus = g * np.ones_like(X)
    V_plus = g + B * phi

    # Initialize V with the upper barrier
    V = V_plus.copy()

    # Bound on |∇V|
    M = np.max(B * grad_phi_norm)
    M = max(M, 1e-6)

    # Domain diameter constant
    diam = 2.0 * max(a, b_ellipse)
    C_omega = (diam**2) / (np.pi**2)

    # Stabilization parameter Λ₀
    Lambda0 = C_alpha * p * (M**(p-1)) * C_omega
    Lambda0 = max(Lambda0, 10.0)

    print(f"H = {H:.4e}, B = {B:.4e}, M_phi = {M_phi:.4e}, C1 = {C1:.4e}")

    s2 = sigma**2 / 2.0
    inv_h2 = 1.0 / (h_grid**2)

    # Policy iteration loop
    for it in range(max_iter):
        V_old = V.copy()

        # Enforce barriers
        V = np.minimum(V, V_plus)
        V = np.maximum(V, V_minus)

        # Compute gradient of V
        dV_dy1, dV_dy2 = np.gradient(V, h_grid, h_grid, edge_order=2)
        grad_sq = np.minimum(dV_dy1**2 + dV_dy2**2, 1e4)
        grad_p = grad_sq**(p/2.0)

        # RHS of linearized PDE
        rhs_f = h_y + Lambda0 * V - C_alpha * grad_p

        data, rows, col = [], [], []
        b_vec = np.zeros(N_int)

        # Build linear system
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
                    rhs_val -= coeff * g  # boundary condition

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
            break

    # Compute gradient norm
    dV_dy1, dV_dy2 = np.gradient(V, h_grid, h_grid, edge_order=2)
    grad_norm = np.sqrt(dV_dy1**2 + dV_dy2**2)

    # Optimal control
    with np.errstate(divide='ignore', invalid='ignore'):
        factor = - (1.0 / (alpha ** (p-1))) * (grad_norm ** (p-2))
        factor[np.isinf(factor)] = 0
        v1 = factor * dV_dy1
        v2 = factor * dV_dy2

    v1[np.isnan(v1)] = 0
    v2[np.isnan(v2)] = 0
    v1[~inside] = 0
    v2[~inside] = 0

    return X, Y, V, grad_norm, v1, v2, inside, y1_vals, y2_vals, phi, V_plus, V_minus, C1


# ----------------------------------------------------------------------
# Main script with asymptotic plots
# ----------------------------------------------------------------------

if __name__ == "__main__":
    # Domain and PDE parameters
    a, b = 1.5, 1.0
    sigma = 0.3
    alpha = 2.0
    g = 0.5
    h_grid = 0.03

    # Solve HJB
    X, Y, V, grad_norm, v1, v2, inside, y1_vals, y2_vals, phi, V_plus, V_minus, C1 = \
        solve_hjb_policy_iteration(a, b, sigma, alpha, g, h_grid)

    plt.rcParams.update({'font.size': 10, 'font.family': 'sans-serif'})

    # ---------------------------------------------------------------
    # Compute asymptotic diagnostics
    # ---------------------------------------------------------------
    t = np.sqrt((X/a)**2 + (Y/b)**2)
    t[t == 0] = 1
    Xb = X / t
    Yb = Y / t
    dist = np.sqrt((Xb - X)**2 + (Yb - Y)**2)
    dist[~inside] = np.nan

    V_minus_g = V - g
    ratio_V = V_minus_g / dist
    ratio_grad = grad_norm / C1

    print("\nAsymptotic diagnostics:")
    print("max (V - g) / dist =", np.nanmax(ratio_V))
    print("max |∇V| / C1      =", np.nanmax(ratio_grad))

    # ---------------------------------------------------------------
    # FIGURE 1 — Solution + bounds (wireframe) + 3D view + 2D view
    # ---------------------------------------------------------------
    fig = plt.figure(figsize=(18, 6))

    eps = 1e-6
    V_minus_plot = V_minus + eps

    ax = fig.add_subplot(131, projection='3d')

    ax.plot_surface(X, Y, np.where(inside, V, np.nan),
                    color='limegreen', alpha=0.75, edgecolor='none')

    ax.plot_wireframe(X, Y, np.where(inside, V_minus_plot, np.nan),
                      color='blue', linewidth=1.5)

    ax.plot_wireframe(X, Y, np.where(inside, V_plus, np.nan),
                      color='red', linewidth=1.5)

    ax.set_title(r"Solution $V$ and bounds $V_{-} \leq V \leq V_{+}$", fontsize=14)
    ax.set_xlabel("y1")
    ax.set_ylabel("y2")

    legend_elements_V = [
        Line2D([0], [0], color='blue', lw=3, label=r"$V_{-}=g$"),
        Line2D([0], [0], color='limegreen', lw=6, label=r"$V$"),
        Line2D([0], [0], color='red', lw=3, label=r"$V_{+}=g+B\phi$")
    ]
    ax.legend(handles=legend_elements_V, loc='upper left')

    ax2 = fig.add_subplot(132, projection='3d')
    ax2.plot_surface(X, Y, np.where(inside, V, np.nan),
                     cmap='viridis', edgecolor='none')
    ax2.set_title(r"Solution $V$ — separate 3D view")

    ax3 = fig.add_subplot(133)
    im = ax3.imshow(np.where(inside, V, np.nan), cmap='viridis')
    plt.colorbar(im, ax=ax3)
    ax3.set_title(r"Solution $V$ — 2D view")

    plt.tight_layout()
    plt.savefig("solution_with_bounds.png", dpi=300)

    # ---------------------------------------------------------------
    # FIGURE 2 — Gradient + bounds (wireframe) + 3D view + 2D view
    # ---------------------------------------------------------------
    fig2 = plt.figure(figsize=(18, 6))

    ax4 = fig2.add_subplot(131, projection='3d')

    ax4.plot_surface(X, Y, np.where(inside, grad_norm, np.nan),
                     color='limegreen', alpha=0.75, edgecolor='none')

    ax4.plot_wireframe(X, Y, np.where(inside, np.zeros_like(V), np.nan),
                       color='blue', linewidth=1.5)

    ax4.plot_wireframe(X, Y, np.where(inside, C1*np.ones_like(V), np.nan),
                       color='red', linewidth=1.5)

    ax4.set_title(r"$|\nabla V|$ and bounds $0 \leq |\nabla V| \leq C_1$", fontsize=14)
    ax4.set_xlabel("y1")
    ax4.set_ylabel("y2")

    legend_elements_grad = [
        Line2D([0], [0], color='blue', lw=3, label=r"$0$"),
        Line2D([0], [0], color='limegreen', lw=6, label=r"$|\nabla V|$"),
        Line2D([0], [0], color='red', lw=3, label=r"$C_1$")
    ]
    ax4.legend(handles=legend_elements_grad, loc='upper left')

    ax5 = fig2.add_subplot(132, projection='3d')
    ax5.plot_surface(X, Y, np.where(inside, grad_norm, np.nan),
                     cmap='inferno', edgecolor='none')
    ax5.set_title(r"$|\nabla V|$ — separate 3D view")

    ax6 = fig2.add_subplot(133)
    im2 = ax6.imshow(np.where(inside, grad_norm, np.nan), cmap='inferno')
    plt.colorbar(im2, ax=ax6)
    ax6.set_title(r"$|\nabla V|$ — 2D view")

    plt.tight_layout()
    plt.savefig("gradient_with_bounds.png", dpi=300)

    plt.show()

    print("Figures solution_with_bounds.png and gradient_with_bounds.png have been generated.")
