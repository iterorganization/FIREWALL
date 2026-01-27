import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def compute_energy_loss_map(file_path):
    data = sio.loadmat(file_path)
    maps = data['MAPS']

    # Extract metadata
    energies = maps[0, 1:]   # MeV
    angles   = maps[1, 1:]   # degrees
    depths   = maps[2:, 0]   # mm

    unique_energies = np.unique(energies)
    unique_angles   = np.unique(angles)

    # Initialize result grid
    # rows = angles, cols = energies
    energy_loss = np.full((len(unique_angles), len(unique_energies)), np.nan)

    # Loop over all profiles
    for col in range(len(energies)):
        E = energies[col]
        A = angles[col]

        profile = maps[2:, col + 1]  # dE/dx (MeV/mm)

        # Integrate over depth → MeV
        deposited_energy = np.trapezoid(profile, depths)

        # Energy loss fraction (or use deposited_energy directly)
        loss_fraction = deposited_energy / E

        i_angle  = np.where(unique_angles == A)[0][0]
        i_energy = np.where(unique_energies == E)[0][0]

        energy_loss[i_angle, i_energy] = loss_fraction

    return unique_energies, unique_angles, energy_loss


energies, angles, loss_map = compute_energy_loss_map('3deg7T.mat')

plt.figure(figsize=(8, 6))
mesh = plt.pcolormesh(
    energies,
    angles,
    loss_map,
    shading='auto'
)

plt.colorbar(mesh, label='Energy Deposition Fraction')
plt.xlabel('Energy (MeV)')
plt.ylabel('Angle (deg)')
plt.title('Energy Loss vs Energy and Angle')

plt.show()


def compute_depth_diagnostics(file_path):
    data = sio.loadmat(file_path)
    maps = data['MAPS']

    # Extract metadata
    energies = maps[0, 1:]   # MeV
    angles   = maps[1, 1:]   # degrees
    depths   = maps[2:, 0]   # mm

    unique_energies = np.unique(energies)
    unique_angles   = np.unique(angles)

    # Initialize result grids
    mean_depth_map = np.full((len(unique_angles), len(unique_energies)), np.nan)
    r95_depth_map  = np.full((len(unique_angles), len(unique_energies)), np.nan)

    # Loop over all profiles
    for col in range(len(energies)):
        E = energies[col]
        A = angles[col]
        profile = maps[2:, col + 1]  # dE/dx (MeV/mm)

        # --- Diagnostic 1: Mean Depth (Centroid) ---
        total_deposited = np.trapezoid(profile, depths)
        if total_deposited > 0:
            mean_z = np.trapezoid(profile * depths, depths) / total_deposited
        else:
            mean_z = np.nan

        # --- Diagnostic 2: 95% Loading Depth (R95) ---
        # Cumulative integration to find the penetration front
        cumulative_E = np.cumsum(profile) 
        if cumulative_E[-1] > 0:
            fractional_E = cumulative_E / cumulative_E[-1]
            # Interpolate to find the exact depth at 0.95 fraction
            z_95 = np.interp(0.95, fractional_E, depths)
        else:
            z_95 = np.nan

        # Map to grid
        i_angle  = np.where(unique_angles == A)[0][0]
        i_energy = np.where(unique_energies == E)[0][0]

        mean_depth_map[i_angle, i_energy] = mean_z
        r95_depth_map[i_angle, i_energy]  = z_95

    return unique_energies, unique_angles, mean_depth_map, r95_depth_map

# Load and compute
energies, angles, mean_map, r95_map = compute_depth_diagnostics('3deg7T.mat')

# --- Plotting ---
fig, ax = plt.subplots(1, 2, figsize=(16, 6))

# Plot Mean Depth
im1 = ax[0].pcolormesh(energies, angles, mean_map, shading='auto', cmap='viridis')
fig.colorbar(im1, ax=ax[0], label='Mean Depth <z> (mm)')
ax[0].set_title('Mean Energy Deposition Depth')
ax[0].set_xlabel('Energy (MeV)')
ax[0].set_ylabel('Angle (deg)')

# Plot 95% Depth
im2 = ax[1].pcolormesh(energies, angles, r95_map, shading='auto', cmap='magma')
fig.colorbar(im2, ax=ax[1], label='95% Range Depth (mm)')
ax[1].set_title('95% Energy Loading Depth (Range)')
ax[1].set_xlabel('Energy (MeV)')
ax[1].set_ylabel('Angle (deg)')

plt.tight_layout()
plt.show()


def fit_penetration_model(unique_energies, unique_angles, depth_map):
    """
    Fits z = alpha * E^p * cos(theta) and returns parameters + prediction function.
    """
    ee, aa = np.meshgrid(unique_energies, unique_angles)
    
    # Flatten and mask NaNs
    energies_flat = ee.ravel()
    angles_flat = aa.ravel()
    depths_flat = depth_map.ravel()
    mask = ~np.isnan(depths_flat)
    
    def range_model(x_data, alpha, p):
        energy, angle_deg = x_data
        return alpha * (energy**p) * np.cos(np.radians(angle_deg))

    # Fit the model (p0=[initial alpha, initial p])
    popt, _ = curve_fit(range_model, (energies_flat[mask], angles_flat[mask]), depths_flat[mask], p0=[0.05, 1.7])
    
    def predict(E, A):
        return popt[0] * (E**popt[1]) * np.cos(np.radians(A))

    return popt, predict

# --- 2. Main Execution Block ---
# (Assuming 'energies', 'angles', and 'r95_map' are already computed from previous steps)
params, predict_func = fit_penetration_model(energies, angles, r95_map)
alpha_fit, p_fit = params

print("-" * 30)
print(f"FIT RESULTS:")
print(f"Alpha (Material Const): {alpha_fit:.5f}")
print(f"p (Power Index): {p_fit:.3f}")
print(f"Equation: Depth = {alpha_fit:.4f} * E^{p_fit:.2f} * cos(theta)")
print("-" * 30)

# --- 3. Visualization: Data vs. Fit ---
plt.figure(figsize=(10, 7))

# Plot the simulation data as a heatmap
mesh = plt.pcolormesh(energies, angles, r95_map, shading='auto', cmap='viridis', alpha=0.8)
plt.colorbar(mesh, label='Measured 95% Depth (mm)')

# Create a smooth grid to draw analytical contour lines
e_smooth = np.linspace(energies.min(), energies.max(), 50)
a_smooth = np.linspace(angles.min(), angles.max(), 50)
EE, AA = np.meshgrid(e_smooth, a_smooth)
ZZ_fit = predict_func(EE, AA)

# Overlay the analytical fit as contour lines
contours = plt.contour(e_smooth, a_smooth, ZZ_fit, colors='white', levels=10, linestyles='--')
plt.clabel(contours, inline=True, fontsize=10, fmt='%.1f mm')

plt.title(f'95% Depth: Simulation vs. Analytical Fit\n$z = {alpha_fit:.3f} \cdot E^{{{p_fit:.2f}}} \cdot \cos(\\theta)$')
plt.xlabel('Energy (MeV)')
plt.ylabel('Angle (deg)')
plt.grid(alpha=0.3)
plt.show()
