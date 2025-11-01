import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

#####################################
# Model Parameters 
#####################################
N = 3                            # Dimension of the state space.
sigma = 0.189                    # Diffusion coefficient (image noise).
R = 10.0                         # Threshold for the state norm.
u0 = 97.799                        # Initial condition: u(r0) = u0.
alpha = 2                   # Exponent in the intervention cost function.

#####################################
# Shooting and PDE Parameters 
#####################################
r0 = 0.01                        # Small starting r to avoid singularity at 0.
r_shoot_end = R                  # We integrate up to r = R.
rInc = 0.1                       # Integration step size.

#####################################
# Image Dynamics Simulation Parameters 
#####################################
dt = 0.01                        # Time step for Euler-Maruyama simulation.
T = 10                           # Maximum simulation time.

#####################################
# Image Cost Function 
#####################################
def h(r):
    """
    Image cost function: h(r) = r^2.
    This penalizes larger deviations (i.e., larger image state norms).
    """
    return r**2

#####################################
# Helper Function for Safe Power Computation 
#####################################
def safe_power(u_prime, exponent, lower_bound=1e-8, upper_bound=1e2):
    """
    Compute (-u_prime)^exponent safely by clipping -u_prime between
    lower_bound and upper_bound. We assume u_prime is negative.
    """
    safe_val = np.clip(-u_prime, lower_bound, upper_bound)
    return np.power(safe_val, exponent)

#####################################
# ODE Definition for the Value Function 
#####################################
def value_function_ode(r, u):
    """
    Defines the ODE for u(r) based on the reduced HJB equation:
    
      u''(r) = -((N-1)/r)*u'(r) + (2/sigma^2)*[ A*(-u'(r))^(alpha/(alpha-1)) - h(r) ],
      
    where
      A = (1/alpha)^(1/(alpha-1)) * ((alpha-1)/alpha).
    
    Debug prints are added to alert us to potential instabilities.
    """
    # Ensure r is not too small (avoid singularity)
    if abs(r) < 1e-6:
        r = 1e-6

    u_val = u[0]
    u_prime = u[1]

    # Ensure u_prime is negative. If not, force to a small negative number.
    if u_prime >= 0:
        print(f"[DEBUG] At r = {r:.4f}, u_prime ({u_prime}) is nonnegative. Forcing to -1e-8.")
        u_prime = -1e-8

    A = (1 / alpha) ** (1/(alpha-1)) * ((alpha-1) / alpha)
    exponent = alpha / (alpha - 1)
    term = A * safe_power(u_prime, exponent)

    du1 = u_prime
    du2 = -((N-1) / r) * u_prime + (2/(sigma**2)) * (term - h(r))
    
    # Extra debug output in a region where instability was observed.
    if 0.45 < r < 0.5:
        print(f"[DEBUG] r = {r:.4f}: u = {u_val}, u_prime (used) = {u_prime}")
    
    return [du1, du2]

#####################################
# Solve the Radial ODE via a Shooting Method 
#####################################
print("Starting ODE integration\ldots")
r_values = np.arange(r0, r_shoot_end + rInc*0.1, rInc)
initial_derivative = -1e-6         # A small negative initial derivative.
u_initial = [u0, initial_derivative]

# Using a stiff solver (BDF) with tight tolerances for stability.
sol = solve_ivp(
    value_function_ode,
    [r0, r_shoot_end],
    u_initial,
    t_eval=r_values,
    method='BDF',                # Stiff solver.
    rtol=1e-10,
    atol=1e-10
)

if sol.success:
    print("ODE integration successful!")
    print(f"Number of r-points: {len(sol.t)}")
else:
    print("ODE integration failed!")

#####################################
# Compute the Boundary Condition g 
#####################################
# We have: u(r) = u(r0) - ?[r0 to r] (-u'(s)) ds.
v_values = -sol.y[1]             # Since u'(r) is negative, -u'(r) is positive.
integral_v = np.trapz(v_values, sol.t)
g_boundary = u0 - integral_v
g_from_solution = sol.y[0][-1]

print("Computed boundary condition g (from integral):", g_boundary)
print("Computed boundary condition g (from ODE solution):", g_from_solution)

#####################################
# Plot Value Function and Its Derivative 
#####################################
plt.figure(figsize=(10, 5))
plt.plot(sol.t, sol.y[0], label="Value Function u(r)")
plt.plot(sol.t, sol.y[1], label="Derivative u'(r)")
plt.axhline(y=g_from_solution, color='r', linestyle='--',
            label=f"Boundary: u(R) = {g_from_solution:.4f}")
plt.xlabel("r (Image State Norm)")
plt.ylabel("u(r) and u'(r)")
plt.title("Shooting Method: u(r) and u'(r)")
plt.legend()
plt.show()
print("Displayed Value Function plot.")

#####################################
# Image Dynamics Simulation
#####################################
def simulate_image_dynamics(x_init, dt, T):
    """
    Simulates the dynamics of image states using the Euler-Maruyama method.
    
    The SDE for each component is given by:
      dx_i = [ ((1/alpha)^(1/(alpha-1))*(1/r)*(-u'(r))^(1/(alpha-1)))*x_i ] dt + sigma*dW_i,
    where r = ||x||, and u'(r) is obtained via interpolation from the ODE solution.
    
    The simulation stops if ||x|| >= R.
    """
    timesteps = int(T / dt)
    x = np.zeros((N, timesteps))
    x[:, 0] = x_init
    
    for t in range(1, timesteps):
        r_norm = np.linalg.norm(x[:, t-1])
        r_norm_safe = r_norm if r_norm > 1e-6 else 1e-6
        
        # Interpolate to get u'(r) from the ODE solution.
        u_prime_val = np.interp(r_norm, sol.t, sol.y[1])
        if u_prime_val >= 0:
            print(f"[DEBUG] At simulation step {t}, u_prime_val ({u_prime_val}) was nonnegative. Forcing to -1e-8.")
            u_prime_val = -1e-8
        
        restoration_rate_unit = ((1/alpha)**(1/(alpha-1))) * (1.0 / r_norm_safe) * \
                                safe_power(u_prime_val, 1/(alpha-1))
        
        for i in range(N):
            drift = restoration_rate_unit * x[i, t-1]
            x[i, t] = x[i, t-1] + drift * dt + sigma * np.random.normal(0, np.sqrt(dt))
        
        if np.linalg.norm(x[:, t]) >= R:
            x = x[:, :t+1]
            print(f"[DEBUG] Stopping simulation at step {t} as state norm reached/exceeded R.")
            break

    return x

print("Starting image dynamics simulation\ldots")
x_initial = np.array([1.0] * N)
image_trajectories = simulate_image_dynamics(x_initial, dt, T)
print("Image dynamics simulation complete.")

#####################################
# Plot Image State Trajectories 
#####################################
plt.figure(figsize=(10, 5))
time_axis = np.arange(image_trajectories.shape[1]) * dt
for i in range(N):
    plt.plot(time_axis, image_trajectories[i], label=f"Component {i+1}")
plt.xlabel("Time")
plt.ylabel("Image State Deviation")
plt.title("Image Dynamics Trajectories (Euler-Maruyama Simulation)")
plt.legend()
plt.show()
print("Displayed Image Dynamics plot.")

#####################################
#Plot Net Restoration Rate (Per Unit Deviation) 
#####################################
# Ensure that u'(r) is safe for all r.
u_prime_safe_array = np.where(sol.y[1] >= 0, -1e-8, np.clip(sol.y[1], -100, -1e-8))
net_rest_rate_per_unit = ((1/alpha)**(1/(alpha-1))) * (1/sol.t) * \
                         np.power(-u_prime_safe_array, 1/(alpha-1))

plt.figure(figsize=(10, 5))
plt.plot(sol.t, net_rest_rate_per_unit, label="Net Restoration Rate per Unit Deviation")
plt.xlabel("r (Image State Norm)")
plt.ylabel("Rate")
plt.title("Net Restoration Rate per Unit Deviation vs. r")
plt.legend()
plt.show()
print("Displayed restoration rate (per unit deviation) plot.")

# =================== Plot Magnitude of the Net Restoration Rate =======================
net_rest_rate_magnitude = ((1/alpha)**(1/(alpha-1))) * \
                          np.power(-u_prime_safe_array, 1/(alpha-1))
plt.figure(figsize=(10, 5))
plt.plot(sol.t, net_rest_rate_magnitude, label="Magnitude of Net Restoration Rate")
plt.xlabel("r (Image State Norm)")
plt.ylabel("Magnitude")
plt.title("Magnitude of the Net Restoration Rate vs. r")
plt.legend()
plt.show()
print("Displayed net restoration rate magnitude plot.")

print("All computations and plots have been executed.")