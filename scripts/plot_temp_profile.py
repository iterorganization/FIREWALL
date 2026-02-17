import argparse
import h5py
import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.animation import FuncAnimation

def run_live_animation(ids, times, depths, data_list, split=False):
    """
    Handles live animation with dynamic time display and optimized Y-limits.
    """
    depth_plot = depths * 1e3
    # Find the global minimum across all data for the bottom of the axis
    global_min = 0
    
    # Store animations to prevent garbage collection
    anis = [] 

    if split:
        for wid, data in zip(ids, data_list):
            fig, ax = plt.subplots(figsize=(8, 5))
            line, = ax.plot(depth_plot, data[0, :], lw=2, label=f'Wall {wid}')
            
            # Set Y-limit based on this specific Wall ID's max temperature
            local_max = np.max(data)
            ax.set_ylim(global_min, local_max * 1.05) # 5% headroom
            
            time_text = ax.text(0.95, 0.95, '', transform=ax.transAxes, 
                                ha='right', va='top', fontweight='bold')
            
            ax.set_title(f"Live Profile - Wall {wid}")
            ax.set_xlabel("Depth [mm]")
            ax.set_ylabel("Temperature [K]")
            ax.legend(loc='upper left')
            ax.grid(True, alpha=0.3)
            
            def update(frame, l=line, d=data, txt=time_text):
                l.set_ydata(d[frame, :])
                txt.set_text(f"Time: {times[frame]:.6e} s")
                return l, txt

            ani = FuncAnimation(fig, update, frames=len(times), blit= False, interval=1)
            anis.append(ani)
        plt.show()

    else:
        # Overlay all IDs: Set Y-limit based on the highest temperature reached by ANY Wall ID
        global_max = max(np.max(d) for d in data_list)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.set_ylim(global_min, global_max * 1.05)
        
        lines = []
        for wid, data in zip(ids, data_list):
            ln, = ax.plot(depth_plot, data[0, :], lw=2, label=f'Wall {wid}')
            lines.append(ln)
        
        time_text = ax.text(0.95, 0.95, '', transform=ax.transAxes, 
                            ha='right', va='top', fontweight='bold', fontsize=12)

        ax.set_title("Live Temperature Profiles Comparison")
        ax.set_xlabel("Depth [mm]")
        ax.set_ylabel("Temperature [K]")
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)

        def update(frame):
            for ln, d in zip(lines, data_list):
                ln.set_ydata(d[frame, :])
            time_text.set_text(f"Time: {times[frame]:.6e} s")
            # Return list of artists for blitting
            return lines + [time_text]

        ani = FuncAnimation(fig, update, frames=len(times), blit=False, interval=1)
        plt.show()

def plot_snapshots(ids, time_val, depths, profiles, split=False):
    """Static plotting logic for specific time indices."""
    depth_plot = depths * 1e3
    if split:
        for wid, profile in zip(ids, profiles):
            plt.figure(figsize=(8, 5))
            plt.plot(depth_plot, profile, lw=2, label=f'Wall {wid}')
            plt.title(f"Wall {wid} at t = {time_val:.3e}s")
            plt.xlabel("Depth [mm]"); plt.ylabel("Temperature [K]")
            plt.legend(); plt.grid(True)
    else:
        plt.figure(figsize=(10, 6))
        for wid, profile in zip(ids, profiles):
            plt.plot(depth_plot, profile, lw=2, label=f'Wall {wid}')
        plt.title(f"Comparison at t = {time_val:.3e}s")
        plt.xlabel("Depth [mm]"); plt.ylabel("Temperature [K]")
        plt.legend(); plt.grid(True)
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_file", required=True, help="Path to HDF5 file containing full in-depth temperature profiles for selected wall IDs")
    parser.add_argument("--wall_ids", help="Comma-separated IDs or .txt file")
    parser.add_argument("--time_idx", type=int, default=0, help="Time index to plot for static snapshot (ignored if --live is set)")
    parser.add_argument("--live", action="store_true", help="Animate temperature profiles over time")
    parser.add_argument("--split", action="store_true", help="Create separate subplots for each Wall ID instead of overlaying, can be used with --live or static")
    args = parser.parse_args()

    # --- ID Parsing ---
    target_ids = []
    if args.wall_ids:
        if os.path.isfile(args.wall_ids):
            with open(args.wall_ids, 'r') as f:
                target_ids = [s.strip() for s in f.read().replace('\n', ',').split(',') if s.strip()]
        else:
            target_ids = [s.strip() for s in args.wall_ids.split(',') if s.strip()]

    with h5py.File(args.results_file, 'r') as f:
        times = f['times'][:]
        depths = f['full_profiles/depths'][:]
        available_ids = [k for k in f['full_profiles'].keys() if k != 'depths']
        if not target_ids: target_ids = available_ids

        valid_ids = []
        all_data = [] # Stores (time, depth) arrays

        for wid in target_ids:
            path = f'full_profiles/{wid}'
            if path in f:
                valid_ids.append(wid)
                all_data.append(f[path][:])
            else:
                print(f"Warning: ID {wid} not found.")

        if not valid_ids:
            print("No valid Wall IDs to plot."); exit()

        if args.live:
            run_live_animation(valid_ids, times, depths, all_data, split=args.split)
        else:
            profiles_at_t = [d[args.time_idx, :] for d in all_data]
            plot_snapshots(valid_ids, times[args.time_idx], depths, profiles_at_t, split=args.split)
