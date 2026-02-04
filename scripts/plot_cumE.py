import h5py
import matplotlib.pyplot as plt
import numpy as np

def plot_energy_profiles(file_path):
    # Open the HDF5 file
    with h5py.File(file_path, 'r') as f:
        # Read the datasets we created
        depths = f['target_depths'][:]
        energy_matrix = f['cum_energy_dep'][:]  # Shape: (num_walls, num_depths)
        wall_ids = f['wall_ids'][:]

    plt.figure(figsize=(10, 6))

    # Plot each wall's profile
    # Using a colormap to distinguish different walls if needed
    for i in range(energy_matrix.shape[0]):
        plt.plot(depths, energy_matrix[i, :], color='tab:blue', alpha=0.3, linewidth=0.5)

    # Plot the mean profile for better visibility of the trend
    mean_profile = np.mean(energy_matrix, axis=0)
    plt.plot(depths, mean_profile, color='red', linewidth=2, label='Mean Profile')

    # Formatting
    plt.title(f"Cumulative Energy Deposition Profiles ({len(wall_ids)} Walls)")
    plt.xlabel("Depth [m]")
    plt.ylabel("Energy Deposition [J/m]") # Units depend on your PhysConst conversions
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.yscale('log')  # Energy deposition often spans several orders of magnitude
    plt.legend()
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Replace with your actual output path
    plot_energy_profiles('../build/results.h5')
