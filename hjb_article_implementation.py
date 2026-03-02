import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.interpolate import RegularGridInterpolator
from scipy.sparse import csr_matrix, linalg
import time

def solve_hjb_monotone(a=1.0, b_ellipse=1.0, sigma=0.3, alpha=1.5, g=0.2, h_grid=0.03, max_iter=100, tol=1e-5):
    """
    Solves the quasilinear HJB equation from the article:
    - (sigma^2/2) * Delta V + C_alpha * |grad V|^p - h(y) = 0
    using the monotone iteration scheme:
    - (sigma^2/2) * Delta V_{k+1} = C_alpha * |grad V_k|^p - h(y)
    valid for any alpha in (1, 2].
    """
    # Conjugate parameters
    p = alpha / (alpha - 1)
    # C_alpha = (alpha - 1) / alpha^(alpha/(alpha-1))
    C_alpha = (alpha - 1) / (alpha ** p)
    
    print(f"Solving general HJB: alpha={alpha:.2f}, p={p:.2f}, C_alpha={C_alpha:.4f}")
    
    # Grid setup
    y1_vals = np.arange(-a - h_grid, a + h_grid, h_grid)
    y2_vals = np.arange(-b_ellipse - h_grid, b_ellipse + h_grid, h_grid)
    X, Y = np.meshgrid(y1_vals, y2_vals, indexing='ij')
    Nx, Ny = X.shape
    
    # Domain mask: (y1/a)^2 + (y2/b)^2 < 1
    inside = ((X / a) ** 2 + (Y / b_ellipse) ** 2) < 1.0
    
    # Identify interior points
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
    
    # 1. Build Laplacian Matrix A (once)
    # Solve - (sigma^2/2) Delta V = RHS => (standard laplacian)
    # - (sigma^2/(2h^2)) * (V_r + V_l + V_u + V_d - 4V_c) = RHS
    # Let L be the matrix such that (LV)_c = 4V_c - V_r - V_l - V_u - V_d
    # Then (sigma^2/(2h^2)) * LV = RHS
    data, rows, cols = [], [], []
    for k, (i, j) in enumerate(zip(*interior_indices)):
        data.append(4.0); rows.append(k); cols.append(k)
        for ni, nj in [(i+1, j), (i-1, j), (i, j+1), (i, j-1)]:
            if interior[ni, nj]:
                data.append(-1.0); rows.append(k); cols.append(idx_map[ni, nj])
    A = csr_matrix((data, (rows, cols)), shape=(N_int, N_int))
    
    # Scale matrix for the solver: A_scaled * V = RHS_scaled
    # (sigma^2 / (2 * h^2)) * A * V = RHS  => A * V = (2 * h^2 / sigma^2) * RHS
    rhs_scale = (2.0 * h_grid**2) / (sigma**2)

    # 2. Initial Guess: g + B * phi (Super-solution as per article)
    # Solve -Delta phi = 1
    b_phi = np.ones(N_int) * (h_grid**2)
    phi_int = linalg.spsolve(A, b_phi)
    phi = np.zeros_like(X)
    phi[interior] = phi_int
    
    h_y = X**2 + Y**2
    B = 2.0 * np.max(h_y) / (sigma**2)
    V = g + B * phi
    
    # Outer monotone iteration
    st = time.time()
    for it in range(max_iter):
        V_old = V.copy()
        
        # Compute gradient magnitude |grad V|
        dV_dy1, dV_dy2 = np.gradient(V_old, h_grid, h_grid, edge_order=2)
        grad_mag_sq = dV_dy1**2 + dV_dy2**2
        # Clip to prevent overflow (especially for p > 2)
        grad_mag_sq = np.clip(grad_mag_sq, 0, 1e4)
        grad_mag = np.sqrt(grad_mag_sq)
        
        # New RHS: f = h(y) - C_alpha * |grad V|^p
        f = h_y - C_alpha * (grad_mag ** p)
        
        b_vec = np.zeros(N_int)
        for k, (i, j) in enumerate(zip(*interior_indices)):
            rhs_val = rhs_scale * f[i, j]
            # Boundary contributions
            for ni, nj in [(i+1, j), (i-1, j), (i, j+1), (i, j-1)]:
                if not interior[ni, nj]:
                    rhs_val += g
            b_vec[k] = rhs_val
            
        V_int_new = linalg.spsolve(A, b_vec)
        
        # Stability: damping (relaxation factor)
        V_new = V.copy()
        V_new[interior] = V_int_new
        V_new[~inside] = g
        
        # Damping factor
        V = 0.5 * V_new + 0.5 * V_old
        
        diff = np.max(np.abs(V - V_old))
        if diff < tol:
            print(f"Converged after {it+1} iterations (diff={diff:.2e})")
            break
    else:
        print(f"Warning: Did not converge (diff={diff:.2e})")
        
    print(f"Solver finished in {time.time() - st:.2f}s")
    
    # Optimal control p*(y) = - (1/alpha^(1/(alpha-1))) * |grad V|^((2-alpha)/(alpha-1)) * grad V
    # Which simplifies to: p* = - (|grad V|/alpha)^(1/(alpha-1)) * (grad V / |grad V|)
    dV_dy1, dV_dy2 = np.gradient(V, h_grid, h_grid, edge_order=2)
    grad_mag = np.sqrt(dV_dy1**2 + dV_dy2**2)
    
    with np.errstate(divide='ignore', invalid='ignore'):
        # Robust clipping of the gradient before the power law
        grad_mag_safe = np.clip(grad_mag, 1e-6, 30.0)
        p_star_mag = (grad_mag_safe / alpha) ** (1.0 / (alpha - 1))
        
        # Guard against excessive drift
        p_star_mag = np.clip(p_star_mag, 0, 10.0)
        
        # Handle direction
        dir1 = np.where(grad_mag > 1e-9, dV_dy1 / grad_mag, 0)
        dir2 = np.where(grad_mag > 1e-9, dV_dy2 / grad_mag, 0)
        p1 = -p_star_mag * dir1
        p2 = -p_star_mag * dir2

    p1[~inside] = 0
    p2[~inside] = 0
    
    return X, Y, V, grad_mag, p1, p2, inside, y1_vals, y2_vals

def simulate_inventory(p1, p2, y1_vals, y2_vals, a, b, sigma, dt=0.01, T=10.0, x0=[0.0, 0.0]):
    p1_interp = RegularGridInterpolator((y1_vals, y2_vals), p1, bounds_error=False, fill_value=0.0)
    p2_interp = RegularGridInterpolator((y1_vals, y2_vals), p2, bounds_error=False, fill_value=0.0)
    
    traj = [np.array(x0)]
    ts = [0.0]
    curr_t = 0.0
    
    while curr_t < T:
        pos = traj[-1]
        if (pos[0]/a)**2 + (pos[1]/b)**2 >= 1.0:
            break
            
        c1 = p1_interp(pos).item()
        c2 = p2_interp(pos).item()
        
        noise = sigma * np.random.randn(2) * np.sqrt(dt)
        next_pos = pos + np.array([c1, c2]) * dt + noise
        
        traj.append(next_pos)
        curr_t += dt
        ts.append(curr_t)
        
    return np.array(traj), np.array(ts)

if __name__ == "__main__":
    # Parameters for testing (General alpha)
    a, b = 1.0, 1.0
    sigma = 0.3
    alpha = 1.5      # Can be any value in (1, 2]
    g = 0.2
    h_grid = 0.02
    
    X, Y, V, Magn, p1, p2, inside, y1_vals, y2_vals = solve_hjb_monotone(a, b, sigma, alpha, g, h_grid)
    
    # Plotting
    plt.rcParams.update({'font.size': 10})
    
    # Figure: Unified Numerical Results
    # 0. Run Simulation first
    traj, times = simulate_inventory(p1, p2, y1_vals, y2_vals, a, b, sigma, x0=[0.1, -0.1])
    
    fig = plt.figure(figsize=(18, 11))
    p1_p = np.where(inside, p1, np.nan)
    p2_p = np.where(inside, p2, np.nan)
    V_plot = np.where(inside, V, np.nan)
    
    # 1. Value Function (3D)
    ax1 = fig.add_subplot(2, 3, 1, projection='3d')
    surf1 = ax1.plot_surface(X, Y, V_plot, cmap='viridis', edgecolor='none', alpha=0.9)
    ax1.set_title(f"Value Function $V(y)$ ($\\alpha={alpha}$)")
    ax1.set_xlabel("$y_1$") ; ax1.set_ylabel("$y_2$")
    fig.colorbar(surf1, ax=ax1, shrink=0.5, aspect=10)
    
    # 2. Optimal Control p1* (3D)
    ax2 = fig.add_subplot(2, 3, 2, projection='3d')
    surf2 = ax2.plot_surface(X, Y, p1_p, cmap='RdBu_r', edgecolor='none', alpha=0.9)
    ax2.set_title("Optimal Control $p_1^*(y)$")
    ax2.set_xlabel("$y_1$") ; ax2.set_ylabel("$y_2$")
    fig.colorbar(surf2, ax=ax2, shrink=0.5, aspect=10)
    
    # 3. Optimal Control p2* (3D)
    ax3 = fig.add_subplot(2, 3, 3, projection='3d')
    surf3 = ax3.plot_surface(X, Y, p2_p, cmap='RdBu_r', edgecolor='none', alpha=0.9)
    ax3.set_title("Optimal Control $p_2^*(y)$")
    ax3.set_xlabel("$y_1$") ; ax3.set_ylabel("$y_2$")
    fig.colorbar(surf3, ax=ax3, shrink=0.5, aspect=10)
    
    # 4. Stochastic Trajectory (Phase Plane)
    ax4 = fig.add_subplot(2, 3, 4)
    theta = np.linspace(0, 2*np.pi, 200)
    ax4.plot(a*np.cos(theta), b*np.sin(theta), 'k--', alpha=0.5, label="Boundary")
    ax4.plot(traj[:, 0], traj[:, 1], 'b-', lw=1.5, label="Path")
    ax4.scatter(traj[0, 0], traj[0, 1], color='green', s=40, label="Start")
    ax4.scatter(traj[-1, 0], traj[-1, 1], color='red', s=40, label="End")
    ax4.set_title("Stochastic Trajectory (Phase Plane)")
    ax4.set_xlabel("$y_1$") ; ax4.set_ylabel("$y_2$")
    ax4.legend(fontsize='small')
    ax4.grid(True, alpha=0.3)
    
    # 5. Inventory Levels vs Time (y1, y2)
    ax5 = fig.add_subplot(2, 3, 5)
    ax5.plot(times, traj[:, 0], 'r-', lw=2, label="$y_1(t)$ (red)", alpha=0.8)
    ax5.plot(times, traj[:, 1], 'g--', lw=2, label="$y_2(t)$ (green)", alpha=0.8)
    ax5.set_title("Inventory Levels vs Time")
    ax5.set_xlabel("Time") ; ax5.set_ylabel("Quantity")
    ax5.grid(True, alpha=0.3)
    ax5.legend(fontsize='small')
    
    # 6. Gradient Magnitude (Bonus/Verification)
    ax6 = fig.add_subplot(2, 3, 6)
    Magn_p = np.where(inside, Magn, np.nan)
    im = ax6.imshow(Magn_p.T, extent=[-a, a, -b, b], origin='lower', cmap='plasma')
    ax6.set_title("Gradient Magnitude $|\nabla V|$")
    fig.colorbar(im, ax=ax6)
    
    plt.tight_layout()
    plt.savefig("hjb_production_planning.png", dpi=300)
    plt.show()
    
    print("Implementation of quasilinear HJB complete (all plots generated).")
