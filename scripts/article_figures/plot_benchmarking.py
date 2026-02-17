
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
t_bench1, d_bench1, temp_bench1 = load_bench('../data/bench_data/true_sol/t_in_depth_results_1MeV.mat')
t_bench2, d_bench2, temp_bench2 = load_bench('../data/bench_data/true_sol/t_in_depth_results_25MeV.mat')

t_melt_1 = 1.035e-4
t_melt_2 = 2.5e-4
N_list = [100, 200, 500, 1000, 2000, 5000, 10000, 20000]

# Initialize Figure
fig, axs = plt.subplots(2, 2, figsize=(14, 10))

# --- PLOT 1: 1 MeV Profiles (Top Left) ---
ax1 = axs[0, 0]
ax1.plot(d_bench1*1e3, temp_bench1[23][::-1], lw=2, ls='--', color='blue', label='Benchmark 1 MeV')
ax1.axhline(3695, lw=2, color='red', ls='--', label='Melting point W')
for N in N_list:
    with h5py.File(f'sim_res_1MeV/results_bench_1MeV_{N}.h5', 'r') as f:
        ax1.plot(f['depths'][:]*1e3, f['temperature'][:, 1035], lw=1.5, label=f'{N} particles')
ax1.set_xlim(0, 1)
ax1.set_xlabel('Depth [mm]')
ax1.set_ylabel('Temperature [K]')

ax1_ins = inset_axes(ax1, width="35%", height="35%", loc="center right")
ax1_ins.plot(d_bench1*1e3, temp_bench1[23][::-1], lw=1.5, ls='--', color='blue')
for N in N_list:
    with h5py.File(f'sim_res_1MeV/results_bench_1MeV_{N}.h5', 'r') as f:
        ax1_ins.plot(f['depths'][:]*1e3, f['temperature'][:, 1035], lw=1)
ax1_ins.set_xlim(0.00, 0.05)
ax1_ins.set_ylim(3400, 3800)
mark_inset(ax1, ax1_ins, loc1=1, loc2=3, fc="none", ec="0.5")

# --- PLOT 2: 25 MeV Profiles (Top Right) ---
ax2 = axs[0, 1]
ax2.plot(d_bench2*1e3, temp_bench2[26][::-1], lw=2, ls='--', color='green', label='Benchmark 25 MeV')
ax2.axhline(3695, lw=2, color='red', ls='--')
for N in N_list:
    with h5py.File(f'sim_res_25MeV/results_bench_25MeV_{N}.h5', 'r') as f:
        ax2.plot(f['depths'][:]*1e3, f['temperature'][:, 2500], lw=1.5)
ax2.set_xlim(0, 10)
ax2.set_xlabel('Depth [mm]')
ax2.set_ylabel('Temperature [K]')

ax2_ins = inset_axes(ax2, width="35%", height="35%", loc="center right")
ax2_ins.plot(d_bench2*1e3, temp_bench2[26][::-1], lw=1.5, ls='--', color='green')
for N in N_list:
    with h5py.File(f'sim_res_25MeV/results_bench_25MeV_{N}.h5', 'r') as f:
        ax2_ins.plot(f['depths'][:]*1e3, f['temperature'][:, 2500], lw=1)
ax2_ins.set_xlim(0.00, 0.05)
ax2_ins.set_ylim(3640, 3725)
mark_inset(ax2, ax2_ins, loc1=1, loc2=3, fc="none", ec="0.5")

# --- DATA PROCESSING FOR ERROR PLOTS ---
res_L2_1, res_dt_1 = [], []
res_L2_25, res_dt_25 = [], []

for N in N_list:
    # 1MeV Error
    with h5py.File(f'sim_res_1MeV/results_bench_1MeV_{N}.h5', 'r') as f:
        sim_d, sim_t = f['depths'][:], f['temperature'][:, 1035]
        b_interp = np.interp(sim_d, d_bench1, temp_bench1[23][::-1])
        res_L2_1.append(np.linalg.norm(sim_t - b_interp) / np.linalg.norm(b_interp))
        surf_t, times = f['temperature'][0], f['times'][:]
        t_obs = times[np.argmax(surf_t >= 3695)]
        res_dt_1.append(np.abs(t_obs - t_melt_1) / 1e-3)

    # 25MeV Error
    with h5py.File(f'sim_res_25MeV/results_bench_25MeV_{N}.h5', 'r') as f:
        sim_d, sim_t = f['depths'][:], f['temperature'][:, 2500]
        b_interp = np.interp(sim_d, d_bench2, temp_bench2[26][::-1])
        res_L2_25.append(np.linalg.norm(sim_t - b_interp) / np.linalg.norm(b_interp))
        surf_t, times = f['temperature'][0], f['times'][:]
        t_obs = times[np.argmax(surf_t >= 3695)]
        res_dt_25.append(np.abs(t_obs - t_melt_2) / 1e-3)

# --- PLOT 3: L2 Error ---
ax3 = axs[1, 0]
ax3.semilogx(N_list, res_L2_1, '-x', color='blue', label='1 MeV')
ax3.semilogx(N_list, res_L2_25, '-x', color='black', label='25 MeV')
ax3.set_xlabel('Number of particles ($N$)')
ax3.set_ylabel(r'$L_2$ relative Error')
ax3.grid(True, which="both", ls="-", alpha=0.2)

# --- PLOT 4: Melt Time Error ---
ax4 = axs[1, 1]
ax4.semilogx(N_list, res_dt_1, '-x', color='blue', label='1 MeV')
ax4.semilogx(N_list, res_dt_25, '-x', color='black', label='25 MeV')
ax4.set_xlabel('Number of particles ($N$)')
ax4.set_ylabel('Melt time relative error')
ax4.grid(True, which="both", ls="-", alpha=0.2)

# --- GLOBAL LEGEND CONSTRUCTION ---
# Legend 1: Profile information (Top half of margin)
h1, l1 = ax1.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
all_h1 = [h1[0], h2[0], h1[1]] + h1[2:]
all_l1 = [l1[0], l2[0], l1[1]] + l1[2:]

fig.legend(all_h1, all_l1, 
           loc='center right', 
           bbox_to_anchor=(1, 0.72), 
           title="Simulation Parameters",
           frameon=True)

# Legend 2: Error series (Bottom half of margin)
h_err, l_err = ax3.get_legend_handles_labels()
fig.legend(h_err, l_err, 
           loc='center right', 
           bbox_to_anchor=(0.95, 0.28), 
           title="Error Comparison",
           frameon=True)

# Layout adjustments for full-page look
plt.subplots_adjust(right=0.82, top=0.92, bottom=0.08, hspace=0.3, wspace=0.25)

# Plot Labeling (a, b, c, d)
plot_labels = ['a)', 'b)', 'c)', 'd)']
all_axs = [ax1, ax2, ax3, ax4]
for ax, label in zip(all_axs, plot_labels):
    ax.text(0.95, 0.95, label, transform=ax.transAxes,
            fontsize=12, va='top', ha='right', fontweight='bold')

plt.show()
