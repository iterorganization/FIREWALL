import numpy as np
import scipy.io
import h5py
import os
import sys

def convert_mat_to_h5(mat_path, h5_path):
    if not os.path.exists(mat_path):
        print(f"Error: Could not find {mat_path}")
        return

    print(f"Reading {mat_path}...")
    data = scipy.io.loadmat(mat_path)

    # Extract data based on energy_dep_interpolator.py logic
    energies_train = np.array([0.5,1,3,6,10,15,25,30,40,50], dtype=np.float64)
    angles_train = np.array([1,2,5,10,20,30,40,55,70,90], dtype=np.float64)

    # data['MAPS'] shape seems to be (N_depth + 2, N_profiles + 1)
    # The first 2 rows seem to be headers or something based on [2:] slicing
    # Column 0 is depth (starting from row 2)
    depths = data['MAPS'][:,0][2:].astype(np.float64)

    # Profiles
    # Y = data['MAPS'][2:, 1:]
    # Reshape logic from python code:
    # temp = np.empty((10,10,12000))
    # for i in range(10): (Energy)
    #     for j in range(10): (Angle)
    #         temp[i,j,:] = Y[:,i+10*j]

    Y = data['MAPS'][2:,1:]
    profiles = np.empty((10, 10, len(depths)), dtype=np.float64)
    for i in range(10):
        for j in range(10):
            profiles[i, j, :] = Y[:, i + 10*j]

    print(f"Writing to {h5_path}...")
    with h5py.File(h5_path, 'w') as f:
        f.create_dataset('energies', data=energies_train)
        f.create_dataset('angles', data=angles_train)
        f.create_dataset('depths', data=depths)
        f.create_dataset('profiles', data=profiles) # Shape (10, 10, 12000)

    print("Conversion complete.")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python prepare_data.py <mat_path> <h5_path>")
        sys.exit(1)
    
    mat_path = sys.argv[1]
    h5_path = sys.argv[2]
    convert_mat_to_h5(mat_path, h5_path)
