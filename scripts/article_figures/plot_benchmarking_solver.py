
import h5py
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 15,
    "legend.fontsize": 12  # Slightly smaller to fit more items
})

# Load Benchmark Data
def load_bench(path):
    mat = loadmat(path, squeeze_me=True, struct_as_record=False)
    data = mat["data"]
    return data.time, data.arc_length, data.temperature

# --- DATA LOADING ---
t_bench1, d_bench1, temp_bench1 = load_bench('../../data/bench_data/true_sol/t_in_depth_results_1MeV.mat')
t_bench2, d_bench2, temp_bench2 = load_bench('../../data/bench_data/true_sol/t_in_depth_results_25MeV.mat')

t_melt_1 = 1.035e-4
t_melt_2 = 2.5e-4

# Initialize Figure
fig, axs = plt.subplots(1, 2, figsize=(14, 7))

for ax in axs:
    ax.set_box_aspect(1)



# --- PLOT 1: 1 MeV Profiles (Left) ---
ax1 = axs[0]
ax1.plot(d_bench1*1e3, temp_bench1[23][::-1], lw=2, color='tab:blue', label='MEMENTO at t = 0.1035 ms')
#ax1.axhline(3695, lw=2, color='red', ls='--', label='Melting point W')
with h5py.File(f'../../build/bench_solver.h5', 'r') as f:
    ax1.plot(f['depths'][:]*1e3, f['temperature_1'][:, 1035], color='tab:red', lw=2, label=f'FIREWALL at t = 0.1035 ms')
ax1.set_xlim(0, 1)
ax1.set_xlabel('Depth [mm]')
ax1.set_ylabel('Temperature [K]')

ax1_ins = inset_axes(ax1, width="35%", height="35%", loc="center right")
ax1_ins.plot(d_bench1*1e3, temp_bench1[23][::-1], lw=2, color='tab:blue')
with h5py.File(f'../../build/bench_solver.h5', 'r') as f:
    ax1_ins.plot(f['depths'][:]*1e3, f['temperature_1'][:, 1035], lw=2, color='tab:red')
ax1_ins.set_xlim(0.00, 0.05)
ax1_ins.set_ylim(3450, 3750)
mark_inset(ax1, ax1_ins, loc1=1, loc2=3, fc="none", ec="0.5")

# --- PLOT 2: 25 MeV Profiles (Right) ---
ax2 = axs[1]
ax2.plot(d_bench2*1e3, temp_bench2[26][::-1], lw=2, color='tab:blue', label='MEMENTO at t = 0.25 ms')
#ax2.axhline(3695, lw=2, color='red', ls='--')
with h5py.File(f'../../build/bench_solver.h5', 'r') as f:
    ax2.plot(f['depths'][:]*1e3, f['temperature_2'][:, 2500], lw=2, color='tab:red', label=f'FIREWALL at t = 0.25 ms')
ax2.set_xlim(0, 10)
ax2.set_xlabel('Depth [mm]')
ax2.set_ylabel('Temperature [K]')

ax2_ins = inset_axes(ax2, width="35%", height="35%", loc="center right")
ax2_ins.plot(d_bench2*1e3, temp_bench2[26][::-1], lw=2, color='tab:blue')
with h5py.File(f'../../build/bench_solver.h5', 'r') as f:
    ax2_ins.plot(f['depths'][:]*1e3, f['temperature_2'][:, 2500], lw=2, color='tab:red')
ax2_ins.set_xlim(0.00, 0.05)
ax2_ins.set_ylim(3640, 3725)
mark_inset(ax2, ax2_ins, loc1=1, loc2=3, fc="none", ec="0.5")


# --- GLOBAL LEGEND CONSTRUCTION ---
# --- INDIVIDUAL LEGENDS ---
# For the first plot
ax1.legend(loc='upper right', frameon=True)

# For the second plot 
# Note: We manually add the 'Melting point' label here since it wasn't labeled in ax2 originally
h2, l2 = ax2.get_legend_handles_labels()
ax2.legend(h2, l2, loc='upper right', frameon=True)



plt.show()


def get_sim_melt_time(time_array, surface_temp_array, threshold=3695):
    """Finds the first time the simulation surface temperature crosses the threshold."""
    above_threshold = np.where(surface_temp_array >= threshold)[0]
    if len(above_threshold) == 0:
        return np.nan
    
    idx = above_threshold[0]
    if idx == 0: return time_array[0]
        
    # Linear interpolation for sub-timestep precision
    t1, t2 = time_array[idx-1], time_array[idx]
    temp1, temp2 = surface_temp_array[idx-1], surface_temp_array[idx]
    return t1 + (threshold - temp1) * (t2 - t1) / (temp2 - temp1)

# --- Error Calculations ---

# 1. MeV Case Calculations
with h5py.File('../../build/bench_solver.h5', 'r') as f:
    # Generate simulation time vector (adjust dt if different)
    sim_time_1 = np.arange(f['temperature_1'].shape[1]) * 1e-7 
    
    # Time Error
    t_sim_melt1 = get_sim_melt_time(sim_time_1, f['temperature_1'][0, :])
    rel_time_err1 = abs(t_sim_melt1 - t_melt_1) / t_melt_1
    
    # L2 Spatial Error (interpolating benchmark to simulation grid)
    bench_interp1 = np.interp(f['depths'][:], d_bench1, temp_bench1[23][::-1])
    rel_l2_1 = np.linalg.norm(f['temperature_1'][:, 1035] - bench_interp1) / np.linalg.norm(bench_interp1)

# 25 MeV Case Calculations
with h5py.File('../../build/bench_solver.h5', 'r') as f:
    sim_time_2 = np.arange(f['temperature_2'].shape[1]) * 1e-7
    
    # Time Error
    t_sim_melt2 = get_sim_melt_time(sim_time_2, f['temperature_2'][0, :])
    rel_time_err2 = abs(t_sim_melt2 - t_melt_2) / t_melt_2
    
    # L2 Spatial Error
    bench_interp2 = np.interp(f['depths'][:], d_bench2, temp_bench2[26][::-1])
    rel_l2_2 = np.linalg.norm(f['temperature_2'][:, 2500] - bench_interp2) / np.linalg.norm(bench_interp2)

# --- Print Summary ---
print(f"{'Case':<10} | {'Rel. Time Error (%)':<20} | {'Rel. L2 Error (%)':<20}")
print("-" * 60)
print(f"{'1 MeV':<10} | {rel_time_err1*100:<20.4f} | {rel_l2_1*100:<20.4f}")
print(f"{'25 MeV':<10} | {rel_time_err2*100:<20.4f} | {rel_l2_2*100:<20.4f}")