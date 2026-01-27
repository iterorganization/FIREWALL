import scipy.io as sio
import numpy as np

def compute_profile_energy(file_path, target_mev, target_deg, weight):
    # Load the MATLAB file
    data = sio.loadmat(file_path)
    maps = data['MAPS']

    # Extract Metadata
    energies = maps[0, 1:]
    angles = maps[1, 1:]
    depths = maps[2:, 0]  # Depth in mm
    
    # Find the column index for the target Energy and Angle
    # Search for pairs (energy, angle) matching our targets
    profile_idx = -1
    for i in range(len(energies)):
        if np.isclose(energies[i], target_mev) and np.isclose(angles[i], target_deg):
            profile_idx = i + 1 # Offset by 1 because col 0 is depth
            break
            
    if profile_idx == -1:
        print(f"Error: Profile for {target_mev} MeV, {target_deg} deg not found.")
        return

    # Extract the dE/dx profile (MeV/mm)
    profile = maps[2:, profile_idx]
    
    # 1. Integrate the profile over depth (MeV/mm * mm = MeV)
    # This gives the total energy deposited by a single particle in MeV
    deposited_energy_mev = np.trapz(profile, depths)
    
    # 2. Conversion Constants
    mev_to_joule = 1.6022e-13
    
    # 3. Calculate Energy in Joules for the specified particle weight
    # Energy (J) = Energy (MeV) * Weight * (Joules per MeV)
    #total_energy_joules = deposited_energy_mev * weight * mev_to_joule
    total_energy_joules = deposited_energy_mev * mev_to_joule
    print(f"--- Results for {target_mev} MeV, {target_deg} deg ---")
    print(f"Energy Deposited per Particle: {deposited_energy_mev} MeV")
    #print(f"Total Energy (Weight {weight:.2e}): {total_energy_joules:.6f} J")
    
    # Diagnostic: Ideal energy (assuming 100% deposition)
    #ideal_joules = target_mev * weight * mev_to_joule
    ideal_joules = target_mev * mev_to_joule
    print(f"Theoretical Max (100% Deposition): {ideal_joules} J")
    print(f"Deposition Efficiency: {(deposited_energy_mev / target_mev)*100}%")

# Parameters
file_name = '3deg7T.mat'
weight = 6.24e11
target_mev = 25.0
target_deg = 10.0

compute_profile_energy(file_name, target_mev, target_deg, weight)
