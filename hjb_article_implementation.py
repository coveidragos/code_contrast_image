import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.interpolate import RegularGridInterpolator
from scipy.sparse import csr_matrix, linalg
import time

def solve_hjb_policy_iteration(a=1.5, b_ellipse=1.0, sigma=0.3, alpha=1.5, g=0.5, h_grid=0.03, max_iter=50, tol=1e-5):
    """
    Solves the quasilinear HJB equation:
    - (sigma^2/2) * Delta V + C_alpha * |grad V|^p - h(y) = 0
    using Policy Iteration (Howard's Algorithm).
    """
    # Conjugate parameters
    p = alpha / (alpha - 1)
    C_alpha = (alpha - 1) / (alpha ** p)
    
    # Grid setup
    y1_vals = np.arange(-a - h_grid, a + h_grid + h_grid, h_grid)
    y2_vals = np.arange(-b_ellipse - h_grid, b_ellipse + h_grid + h_grid, h_grid)
    X, Y = np.meshgrid(y1_vals, y2_vals, indexing='ij')
    Nx, Ny = X.shape
    
    # Domain mask: (y1/a)^2 + (y2/b)^2 < 1
    inside = ((X / a) ** 2 + (Y / b_ellipse) ** 2) < 1.0
    
    # Interior points are those whose neighbors are all inside
    interior = np.zeros_like(inside, dtype=bool)
    for i in range(1, Nx - 1):
        for j in range(1, Ny - 1):
            if inside[i, j] and inside[i+1, j] and inside[i-1, j] and inside[i, j+1] and inside[i, j-1]:
                interior[i, j] = True
    
    # Map interior points to a linear system index
    interior_indices = np.where(interior)
    idx_map = -np.ones((Nx, Ny), dtype=int)
    for k, (i, j) in enumerate(zip(*interior_indices)):
        idx_map[i, j] = k
    N_int = len(interior_indices[0])
    
    # Source term h(y)
    h_y = X**2 + Y**2
    
    # Initialize V with boundary condition g
    V = g * np.ones_like(X)
    
    print(f"Solving HJB for alpha={alpha:.2f} on {X.shape} grid using Policy Iteration...")
    st = time.time()
    
    # Constants for discretization
    s2 = sigma**2 / 2.0
    inv_h2 = 1.0 / (h_grid**2)
    inv_2h = 1.0 / (2.0 * h_grid)
    
    for it in range(max_iter):
        V_old = V.copy()
        
        # 1. Policy Improvement: Find optimal control v* given current V
        dV_dy1, dV_dy2 = np.gradient(V, h_grid, h_grid, edge_order=2)
        grad_norm = np.sqrt(dV_dy1**2 + dV_dy2**2)
        
        # v* magnitude: |v*| = (|grad V| / alpha)^(1/(alpha-1))
        # p-1 = 1/(alpha-1)
        with np.errstate(divide='ignore', invalid='ignore'):
            v_mag = (grad_norm / alpha) ** (1.0 / (alpha - 1.0))
            v_mag = np.nan_to_num(v_mag, nan=0.0, posinf=1e5, neginf=0.0)
            v_mag = np.clip(v_mag, 0, 1e5)
            # Optimal direction is opposite to gradient
            unit_v1 = -dV_dy1 / (grad_norm + 1e-12)
            unit_v2 = -dV_dy2 / (grad_norm + 1e-12)
            v1_opt = v_mag * unit_v1
            v2_opt = v_mag * unit_v2
        
        v1_opt[np.isnan(v1_opt)] = 0
        v2_opt[np.isnan(v2_opt)] = 0
        v1_opt[~inside] = 0
        v2_opt[~inside] = 0
        
        # 2. Policy Evaluation: Solve for V:
        # -(sigma^2/2) Delta V - v1_opt * dV/dy1 - v2_opt * dV/dy2 = h + |v_opt|^alpha
        cost_control = (v_mag)**alpha
        rhs_f = h_y + cost_control
        
        data, rows, col = [], [], []
        b_vec = np.zeros(N_int)
        
        # Build Sparse Matrix L
        for k, (i, j) in enumerate(zip(*interior_indices)):
            rhs_val = rhs_f[i, j]
            # Diagonal: 4*s2/h^2
            diag = 4.0 * s2 * inv_h2
            
            # Neighbors contributions -s2/h2 - v/(2h)
            neighbors = [
                (i+1, j, -s2*inv_h2 - v1_opt[i, j]*inv_2h), # E
                (i-1, j, -s2*inv_h2 + v1_opt[i, j]*inv_2h), # W
                (i, j+1, -s2*inv_h2 - v2_opt[i, j]*inv_2h), # N
                (i, j-1, -s2*inv_h2 + v2_opt[i, j]*inv_2h)  # S
            ]
            
            for ni, nj, coeff in neighbors:
                if interior[ni, nj]:
                    data.append(coeff); rows.append(k); col.append(idx_map[ni, nj])
                else:
                    rhs_val -= coeff * g # Boundary contribution
            
            data.append(diag); rows.append(k); col.append(k)
            b_vec[k] = rhs_val
            
        L = csr_matrix((data, (rows, col)), shape=(N_int, N_int))
        V_int = linalg.spsolve(L, b_vec)
        V[interior] = V_int
        
        diff = np.max(np.abs(V - V_old))
        if diff < tol:
            print(f"Policy iteration converged after {it+1} steps (diff={diff:.2e})")
            break
    else:
        print(f"Warning: Policy iteration did not converge.")

    print(f"Solver finished in {time.time() - st:.2f} seconds.")
    
    # Final Gradients
    dV_dy1, dV_dy2 = np.gradient(V, h_grid, h_grid, edge_order=2)
    grad_norm = np.sqrt(dV_dy1**2 + dV_dy2**2)
    
    # Recalculate optimal control for output
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

def simulate_dynamics(v1, v2, y1_vals, y2_vals, a, b, sigma, dt=0.01, T=10.0, x0=[0.0, 0.0]):
    """Simulates dX = v*(X) dt + sigma dW."""
    v1_interp = RegularGridInterpolator((y1_vals, y2_vals), v1, bounds_error=False, fill_value=0.0)
    v2_interp = RegularGridInterpolator((y1_vals, y2_vals), v2, bounds_error=False, fill_value=0.0)
    
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

if __name__ == "__main__":
    a, b = 1.5, 1.0
    sigma = 0.3
    alpha = 1.5
    g = 0.5
    h_grid = 0.03
    
    X, Y, V, Magn, v1, v2, inside, y1_vals, y2_vals = solve_hjb_policy_iteration(a, b, sigma, alpha, g, h_grid)
    
    plt.rcParams.update({'font.size': 10, 'font.family': 'sans-serif'})
    
    # Combined Production Planning Figure (matching LaTeX caption)
    fig = plt.figure(figsize=(18, 10))
    
    # 1. Value Function (Left)
    ax1 = fig.add_subplot(221, projection='3d')
    V_plot = np.where(inside, V, np.nan)
    surf1 = ax1.plot_surface(X, Y, V_plot, cmap='magma', edgecolor='none', alpha=0.9, antialiased=True)
    ax1.set_title(f"Value Function $V(y)$", fontsize=14)
    ax1.set_xlabel("$y_1$")
    ax1.set_ylabel("$y_2$")
    fig.colorbar(surf1, ax=ax1, shrink=0.5, aspect=10)
    
    # 2. Optimal Control v1* (Center-ish)
    ax2 = fig.add_subplot(223, projection='3d')
    v1_plot = np.where(inside, v1, np.nan)
    surf2 = ax2.plot_surface(X, Y, v1_plot, cmap='coolwarm', edgecolor='none', alpha=0.9)
    ax2.set_title("Optimal Control $v_1^*(y)$", fontsize=14)
    ax2.set_xlabel("$y_1$")
    ax2.set_ylabel("$y_2$")
    fig.colorbar(surf2, ax=ax2, shrink=0.5, aspect=10)
    
    # 3. Optimal Control v2* (Right-ish)
    ax3 = fig.add_subplot(224, projection='3d')
    v2_plot = np.where(inside, v2, np.nan)
    surf3 = ax3.plot_surface(X, Y, v2_plot, cmap='coolwarm', edgecolor='none', alpha=0.9)
    ax3.set_title("Optimal Control $v_2^*(y)$", fontsize=14)
    ax3.set_xlabel("$y_1$")
    ax3.set_ylabel("$y_2$")
    fig.colorbar(surf3, ax=ax3, shrink=0.5, aspect=10)
    
    # 4. Stochastic Trajectory (Top Right)
    traj, times = simulate_dynamics(v1, v2, y1_vals, y2_vals, a, b, sigma, x0=[0.5, 0.5])
    ax4 = fig.add_subplot(222)
    theta = np.linspace(0, 2*np.pi, 200)
    ax4.plot(a*np.cos(theta), b*np.sin(theta), 'k--', alpha=0.5, label="Boundary")
    ax4.plot(traj[:, 0], traj[:, 1], 'C0-', lw=1.5, label="Trajectory")
    ax4.scatter(traj[0, 0], traj[0, 1], c='green', s=40, label="Start")
    ax4.scatter(traj[-1, 0], traj[-1, 1], c='red', s=40, label="End/Exit")
    ax4.set_title("Simulated Inventory Trajectory", fontsize=14)
    ax4.set_xlabel("$y_1$")
    ax4.set_ylabel("$y_2$")
    ax4.grid(True, linestyle=':', alpha=0.6)
    ax4.legend()
    
    plt.tight_layout()
    plt.savefig("hjb_production_planning.png", dpi=300)
    plt.show()
    
    print(f"Value range: {np.nanmin(V_plot):.3f} to {np.nanmax(V_plot):.3f}")
    print("Main production planning figure saved as 'hjb_production_planning.png'.")
