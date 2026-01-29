import sys
import h5py
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from scipy.io import loadmat

# 1. Load Simulation Data (HDF5)
input_file = sys.argv[1] if len(sys.argv) > 1 else 'results_bench.h5'

with h5py.File(input_file, 'r') as f:
  sol_sim = f['temperature'][:]  # Shape: (Depth, Time)
  times_sim = f['times'][:]
  depths_sim = f['depths'][:]


# 2. Load Benchmark Data (MAT)
def load_bench(path):
  mat = loadmat(path, squeeze_me=True, struct_as_record=False)
  data = mat["data"]
  # temperature is often (Time, Depth) in these mat files
  return data.time, data.arc_length, data.temperature

t_bench1, d_bench1, temp_bench1 = load_bench('t_in_depth_results_1MeV.mat')

# 3. Setup Figure
fig, ax = plt.subplots(figsize=(10, 6))

# Simulation Line
line_sim, = ax.plot(depths_sim, sol_sim[:, 0], lw=3, color='blue', label='sim')

# Benchmark Lines (Live updating)
line_bench1, = ax.plot(d_bench1, temp_bench1[0][::-1], lw=1.5, ls='--', color='blue', label='Bench 1MeV')

# 4. Formatting
ax.axhline(3695)
ax.set_ylim(300, 5000)
ax.set_xlim(0, 0.01)
ax.set_xlabel("Depth (m)")
ax.set_ylabel("Temperature [K]")
ax.grid(True, linestyle=':', alpha=0.7)
ax.legend()

# 5. Animation Update Function
def update(frame):
  current_t = times_sim[frame]
  
  # Update Simulation Line
  line_sim.set_ydata(sol_sim[:, frame])
  
  # Update Benchmark 1 (Find closest time index)
  idx1 = np.argmin(np.abs(t_bench1 - current_t))
  line_bench1.set_ydata(temp_bench1[idx1][::-1])
  
  ax.set_title(f"Comparison at t = {current_t:.4e} s")
  return line_sim, line_bench1

# 6. Create Animation
ani = animation.FuncAnimation(
  fig, update, frames=len(times_sim), interval=50, blit=False
)

plt.show()
