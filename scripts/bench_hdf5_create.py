import numpy as np
import h5py
import scipy
from scipy.io import loadmat
import os

folder_1 = "1MeV_files"
folder_2 = "25MeV_files"

for folder in [folder_1, folder_2]:
    if not os.path.exists(folder):
        os.makedirs(folder)

N_list = [10, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000]


load_time = 1e-3
E_part_1 = 1
E_part_2 = 25
angle_1 = 2
angle_2 = 10
conv_factor = 6.24150907e12 #Jul to MeV
E_loaded_1 = 10*1.0587*conv_factor
E_loaded_2 = 40*1.1371*conv_factor

for N in N_list:
    time = np.linspace(0, load_time, N)
    weight1 = np.ones(N)*(E_loaded_1/(N*E_part_1))
    weight2 = np.ones(N)*(E_loaded_2/(N*E_part_2))
    i_elm = np.ones(N)*(-1)

    file_path_1 = os.path.join(folder_1, f"bench_1MeV_{N}.h5")
    with h5py.File(file_path_1, "w") as f:
        grp = f.create_group("groups")
        subgrp = grp.create_group("001")
        subgrp.create_dataset("energy", data = E_part_1)
        subgrp.create_dataset("angle", data = angle_1)
        subgrp.create_dataset("i_elm", data = i_elm)
        subgrp.create_dataset("t_loss", data = time)
        subgrp.create_dataset("weight", data = weight1)
        f.create_dataset("time", data = time[-1])

    file_path_2 = os.path.join(folder_2, f"bench_25MeV_{N}.h5")
    with h5py.File(file_path_2, "w") as f:
        grp = f.create_group("groups")
        subgrp = grp.create_group("001")
        subgrp.create_dataset("energy", data = E_part_2)
        subgrp.create_dataset("angle", data = angle_2)
        subgrp.create_dataset("i_elm", data = i_elm)
        subgrp.create_dataset("t_loss", data = time)
        subgrp.create_dataset("weight", data = weight2)
        f.create_dataset("time", data = time[-1])
