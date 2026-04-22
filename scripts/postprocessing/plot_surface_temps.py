#!/usr/bin/env python3

import argparse
import sys
import os
import numpy as np
import h5py
import pyvista as pv
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 18,
    "axes.labelsize": 24
})

# Fixed Parameters
R0, Z0 = 5.2, 0.1
VMIN, VMAX_DEFAULT = 299, 1500 # Adjust VMAX_DEFAULT as needed

def read_wallinput(wallin):
    if not os.path.exists(wallin):
        sys.exit(f"Error: Wall geometry file '{wallin}' not found.")
    with h5py.File(wallin, 'r') as h5:
        data = h5["nodes"][:]
        ntri = int(h5["ntriangle"][0])
    verts = []
    faces = []
    for i in range(ntri):
        p1, p2, p3 = data[i*9:i*9+3], data[i*9+3:i*9+6], data[i*9+6:i*9+9]
        verts += [p1, p2, p3]
        faces += [[3, i*3, i*3+1, i*3+2]]
    return pv.PolyData(np.array(verts), faces)

def read_temps(results, mesh, option):
    m = mesh.copy()
    m.cell_data['temps'] = np.zeros(mesh.n_cells)
    with h5py.File(results, 'r') as h5:
        wallid = (h5['wall_ids'][:] - 1).astype(int)
        if option == "last_time":
            temps = h5['surf_temp'][:,-1]
        elif option == "max_temperature" or option == "binary":
            temps = np.max(h5['surf_temp'][:], axis=1)
        elif option == "first_melt":
            raw_data = h5['surf_temp'][:]
            # Find the first time index where ANY cell > 3695
            melt_timeline = np.any(raw_data > 3695, axis=0)
            if np.any(melt_timeline):
                idx = np.argmax(melt_timeline)
                print(f"File {os.path.basename(results)} reached melt at index {idx}")
                temps = raw_data[:, idx]
            else:
                print(f"Warning: No melt detected in {results}. Using last time step.")
                temps = raw_data[:, -1]

        valid = (wallid >= 0) & (wallid < mesh.n_cells)
        m.cell_data['temps'][wallid[valid]] = temps[valid]
    return m

def plot_3d_subplot(ax, mesh, r0, z0, phicam, title, vmin, vmax, option):
    p = pv.Plotter(off_screen=True)
    m_plot = mesh.copy()
    
    if option == "binary":
        # Values > 3695 are 1 (Red), everything else is NaN (Light Grey)
        raw_temps = m_plot.cell_data['temps']
        binary_data = np.where(raw_temps > 3695, 1.0, np.nan)
        m_plot.cell_data['plot_scalars'] = binary_data
        cmap_binary = mpl.colors.ListedColormap(['red'])
        p.add_mesh(m_plot, scalars='plot_scalars', cmap=cmap_binary, 
                   show_scalar_bar=False, nan_color='lightgrey', clim=[1, 1])
    else:
        # Standard logic for other modes
        temps = m_plot.cell_data['temps'].copy()
        temps[temps <= vmin] = np.nan
        m_plot.cell_data['plot_scalars'] = temps
        p.add_mesh(m_plot, scalars='plot_scalars', cmap='jet', 
                   clim=[vmin, vmax], log_scale=True, show_scalar_bar=False, nan_color='lightgrey')

    p.camera.position = (1.9*r0*np.cos(phicam), 1.9*r0*np.sin(phicam), z0 - 1)
    p.camera.focal_point = (1.87*r0*np.cos(phicam), 1.87*r0*np.sin(phicam), z0 - 0.9)
    p.show(auto_close=False)
    ax.imshow(p.image)
    ax.axis("off")
    p.close()

def plot_dist_subplot(ax, mesh, title, vmin, vmax, option):
    centers = mesh.cell_centers().points
    phi_deg = np.rad2deg(np.arctan2(centers[:, 1], centers[:, 0])) % 360
    z = centers[:, 2]
    temps = mesh.cell_data['temps']

    from scipy.stats import binned_statistic_2d
    
    bin_values, xedges, yedges, _ = binned_statistic_2d(
        phi_deg, z, temps, 
        statistic='max', # Changed from average to max
        bins=[180, 90],
        range=[[0, 360], [1.3, 4.6]] # Explicit range to match your limits
    )

    # Replace NaNs (empty bins) with vmin
    bin_values = np.nan_to_num(bin_values, nan=vmin)

    ax.set_facecolor("black")
    X, Y = np.meshgrid(xedges, yedges)

    if option == "binary":
        cmap_bin = mpl.colors.ListedColormap(['black', 'red'])
        norm_bin = mpl.colors.BoundaryNorm([0, 3695, 10000], cmap_bin.N)
        im = ax.pcolormesh(X, Y, bin_values.T, shading='flat', 
                           cmap=cmap_bin, norm=norm_bin)
    else:   
        im = ax.pcolormesh(X, Y, bin_values.T, shading='flat', cmap='turbo',
                           norm=mpl.colors.LogNorm(vmin=vmin, vmax=vmax))

    ax.set_box_aspect(1)
    ax.set_xlim(0, 360)
    ax.set_ylim(1.3,4.6)
    ax.set_xticks([0, 90, 180, 270, 360])
    ax.set_xlabel("Toroidal angle [deg]")
    if ax.get_subplotspec().is_first_col():
        ax.set_ylabel("Z [m]")
    
    return im


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--wall_file", help = "Path to wall geometry HDF5 file", required=True)
    parser.add_argument("--results_file", help = "Path to the results HDF5 file", required=True)
    parser.add_argument("--option", choices=["last_time", "max_temperature", "first_melt", "binary"], default="last_time", help="Plotting option: 'last_time' for final step, 'max_temperature' for max temperatures reached over simulation, 'first_melt' for first time step where any cell > 3695K, 'binary' for melt/no-melt")
    parser.add_argument("--phi_cam", type=lambda x: np.radians(float(eval(x, {"np": np, "pi": np.pi}))), default=0.0, help="Camera angle in degrees (e.g., 0, 10, 45, 90).")
    parser.add_argument("--live", action="store_true", help="Animate temperature over time")
    args = parser.parse_args()

    base_mesh = read_wallinput(args.wall_file)
    if args.live:
        from matplotlib.animation import FuncAnimation
        # --- Optimized Live Animation ---
        with h5py.File(args.results_file, 'r') as h5:
            all_temps = h5['surf_temp'][:]
            wall_ids = (h5['wall_ids'][:] - 1).astype(int)
            total_steps = all_temps.shape[1]
        
        frame_indices = np.arange(0, total_steps, 10)
        global_vmax = all_temps.max()

        # 1. Persistent PyVista setup
        p = pv.Plotter(off_screen=True)
        base_mesh.cell_data['temps'] = np.full(base_mesh.n_cells, np.nan)
        p.add_mesh(base_mesh, scalars='temps', cmap='jet', clim=[VMIN, global_vmax], 
                   log_scale=True, nan_color='lightgrey', show_scalar_bar=False)
        p.camera.position = (1.9*R0*np.cos(args.phi_cam), 1.9*R0*np.sin(args.phi_cam), Z0 - 1)
        p.camera.focal_point = (1.87*R0*np.cos(args.phi_cam), 1.87*R0*np.sin(args.phi_cam), Z0 - 0.9)
        p.render()
        initial_img = p.screenshot()

        # 2. Matplotlib figure setup
        fig, (ax3d, ax2d) = plt.subplots(1, 2, figsize=(16, 7))
        ax2d.set_facecolor("black")
        im3d = ax3d.imshow(p.screenshot())
        ax3d.axis("off")

        # Pre-calc coordinates for 2D histogram speed
        centers = base_mesh.cell_centers().points
        phi_deg = np.rad2deg(np.arctan2(centers[:, 1], centers[:, 0])) % 360
        z_coords = centers[:, 2]

        # Initial 2D Plot
        im2d = ax2d.imshow(np.full((90, 180), VMIN).T, origin='lower', extent=[0, 360, 1.3, 4.6],
                           cmap='jet', norm=mpl.colors.LogNorm(vmin=VMIN, vmax=global_vmax), aspect='auto')
        ax2d.set_xlabel("Toroidal angle [deg]")
        ax2d.set_ylabel("Z [m]")

        def update(frame_idx):
            current_frame_temps = np.zeros(base_mesh.n_cells)
            valid = (wall_ids >= 0) & (wall_ids < base_mesh.n_cells)
            current_frame_temps[wall_ids[valid]] = all_temps[valid, frame_idx]
            
            # Update 3D
            plot_scalars = current_frame_temps.copy()
            plot_scalars[plot_scalars < VMIN] = np.nan
            base_mesh.cell_data['temps'] = plot_scalars
            p.render()
            im3d.set_data(p.screenshot())
            
            # Update 2D
            counts, _, _ = np.histogram2d(phi_deg, z_coords, bins=(180, 90), range=[[0, 360], [1.3, 4.6]])
            t_sum, _, _ = np.histogram2d(phi_deg, z_coords, bins=(180, 90), range=[[0, 360], [1.3, 4.6]], weights=current_frame_temps)
            avg_temp = np.where(counts > 0, t_sum / counts, VMIN)
            im2d.set_array(avg_temp.T)
            
            fig.suptitle(f"Time Step: {frame_idx}", fontsize=16)
            return im3d, im2d

        ani = FuncAnimation(fig, update, frames=frame_indices, interval=1, blit=False)
        plt.show()
        p.close()

    else:
        mesh = read_temps(args.results_file, base_mesh, args.option)
        temps = mesh.cell_data['temps']
        #global_vmax = temps.max() if args.option != "binary" else 1.0
        global_vmax = 3695
        # --- 3D Plotting (Single Plot) ---
        fig3d, ax3d = plt.subplots(figsize=(10, 10))
        plot_3d_subplot(ax3d, mesh, R0, Z0, args.phi_cam, "3D View", VMIN, global_vmax, args.option)

        if args.option != "binary":
            # Simplified colorbar for a single plot
            norm = mpl.colors.LogNorm(vmin=VMIN, vmax=global_vmax)
            sm = mpl.cm.ScalarMappable(norm=norm, cmap='jet')
            ticks_to_show = [300, 500, 1000, 1500, 2500, 3695]
            cb3d = fig3d.colorbar(
                sm, 
                ax=ax3d, 
                orientation='horizontal', 
                location='top', 
                fraction=0.08, 
                pad=0.04, 
                ticks=ticks_to_show
            )
            # Force the labels to match the ticks exactly
            cb3d.ax.set_xticklabels([str(t) for t in ticks_to_show])
            label = 'Maximal surface temperature [K]' if args.option == "max_temperature" else 'Surface temperature [K]'
            cb3d.set_label(label)
        
        # --- 2D Plotting (Single Plot) ---
        # Changed from (1, 4) to just a single subplot
        fig2d, ax2d = plt.subplots(figsize=(10, 6)) 
        im2d = plot_dist_subplot(ax2d, mesh, "Temperature Distribution", VMIN, global_vmax, args.option)
        
        if args.option != "binary":
            ticks_to_show = [300, 500, 1000, 1500, 2500, 3695]
            # Standard colorbar attachment
            cb2d = fig2d.colorbar(im2d, ax=ax2d, fraction=0.1, aspect=15, ticks=ticks_to_show)
            cb2d.ax.set_yticklabels([str(t) for t in ticks_to_show])
            cb2d.set_label('Surface temperature [K]')
        
        fig2d.tight_layout()
        plt.show()

