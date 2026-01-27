import numpy as np
import h5py

N_list = [500]
load_time = 1e-4

for N in N_list:
    # 1. Generate Random Impact Times in {0, load_time}
    time = np.random.uniform(0, load_time, N)
    
    # 2. Generate Random Energies (0.5 to 50 MeV)
    energy = np.random.uniform(0.5, 50.0, N)
    
    # 3. Generate Random Angles (1 to 90 deg)
    angle = np.random.uniform(1.0, 90.0, N)
    
    # 4. Generate Random Weights (1e24 to 1e25)
    weight = np.random.uniform(1e12, 1e13, N)
    
    i_elm = np.full(N, -1)

    # Write to HDF5
    file_name = f"random_impacts_{N}.h5"
    with h5py.File(file_name, "w") as f:
        grp = f.create_group("groups")
        subgrp = grp.create_group("001")
        
        subgrp.create_dataset("energy", data=energy)
        subgrp.create_dataset("angle", data=angle)
        subgrp.create_dataset("i_elm", data=i_elm)
        subgrp.create_dataset("t_loss", data=time)
        subgrp.create_dataset("weight", data=weight)
        
        # Storing the total load_time as a global attribute/dataset
        f.create_dataset("time", data=load_time)

    print(f"Successfully created {file_name} with {N} random samples.")

