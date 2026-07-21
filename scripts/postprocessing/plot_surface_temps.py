#!/usr/bin/env python3

# Post-processing script for visualizing surface temperatures on a wall. It reads wall geometry and surface temperature data from HDF5 files and generates visualizations of the surface temperature distribution on the wall. The script supports both interactive 3D visualization and static 2D/3D plots, with options for different temperature representations.

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

# Fixed Parameters. The actual RO and ZO are for the ITER geometry, to be made generic in the future.
R0, Z0 = 5.2, 0.1
VMIN, VMAX = 299, 3695

# Helper function to read wall geometry from the HDF5 file and create a PyVista mesh
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

# Helper function to read surface temperatures from the results file and map them onto the mesh
def read_temps(results, mesh, option):
    m = mesh.copy()
    m.cell_data['temps'] = np.zeros(mesh.n_cells)
    with h5py.File(results, 'r') as h5:
        wallid = (h5['wall_ids'][:] - 1).astype(int)
        surf_temp = h5['surf_temp'][:]
        is_time_resolved = surf_temp.ndim == 2 and surf_temp.shape[1] > 1

        if option == "last_time":
            temps = surf_temp[:, -1] if surf_temp.ndim == 2 else surf_temp
        elif option in ("max_temperature", "binary"):
            if not is_time_resolved:
                sys.exit(f"Error: option '{option}' requires time-resolved data "
                         f"(re-run FIREWALL with --store_all_times). "
                         f"This file only contains the last timestep.")
            temps = np.max(surf_temp, axis=1)
        elif option == "first_melt":
            if not is_time_resolved:
                sys.exit("Error: option 'first_melt' requires time-resolved data "
                         "(re-run FIREWALL with --store_all_times). "
                         "This file only contains the last timestep.")
            raw_data = surf_temp
            # Find the first time index where ANY cell > VMAX
            melt_timeline = np.any(raw_data > VMAX, axis=0)
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

# Plot the surface temperature distribution in a 3D interactive window
def plot_3d_interactive(mesh, vmin, vmax, option):
    p = pv.Plotter(off_screen=False)
    m_plot = mesh.copy()

    if option == "binary":
        raw_temps = m_plot.cell_data['temps']
        binary_data = np.where(raw_temps > VMAX, 1.0, np.nan)
        m_plot.cell_data['plot_scalars'] = binary_data
        cmap_binary = mpl.colors.ListedColormap(['red'])
        p.add_mesh(m_plot, scalars='plot_scalars', cmap=cmap_binary,
                   show_scalar_bar=False, nan_color='lightgrey', clim=[1, 1])
        p.add_text("Melted (T > 3695 K)", font_size=10)
    else:
        temps = m_plot.cell_data['temps'].copy()
        temps[temps <= vmin] = np.nan
        m_plot.cell_data['plot_scalars'] = temps
        label = 'Maximal surface temperature [K]' if option == "max_temperature" else 'Surface temperature [K]'
        p.add_mesh(m_plot, scalars='plot_scalars', cmap='jet',
                   clim=[vmin, vmax], log_scale=True, nan_color='lightgrey',
                   scalar_bar_args={'title': label})

    p.enable_trackball_style()   # explicit: mouse-drag rotate, right-drag zoom, middle-drag pan
    p.view_isometric()           # sensible default framing, auto-fits clipping range to the mesh
    p.reset_camera()             # ensure clipping planes fit the whole geometry, not a stale fixed view
    p.show()

# Plot the surface temperature distribution in a 3D interactive window with live updates
def plot_3d_interactive_live(results_file, wall_ids, base_mesh, fps=10, frame_stride=10):
    from pyvistaqt import BackgroundPlotter
    from PyQt5 import QtCore  # pyvistaqt depends on PyQt5 (or PySide2, see note below)

    with h5py.File(results_file, 'r') as h5:
        all_temps = h5['surf_temp'][:]
        if all_temps.ndim == 1 or all_temps.shape[1] < 2:
            sys.exit("Error: --interactive --live requires time-resolved surf_temp data "
                     "(re-run FIREWALL with --store_all_times).")
        total_steps = all_temps.shape[1]

    base_mesh.cell_data['temps'] = np.full(base_mesh.n_cells, np.nan)

    p = BackgroundPlotter()  # opens a real Qt window; app.exec_() runs inside show(), managed by pyvistaqt
    p.add_mesh(base_mesh, scalars='temps', cmap='jet', clim=[VMIN, VMAX],
               log_scale=True, nan_color='lightgrey',
               scalar_bar_args={'title': 'Surface temperature [K]'})
    p.enable_trackball_style()
    p.view_isometric()
    p.reset_camera()

    valid = (wall_ids >= 0) & (wall_ids < base_mesh.n_cells)
    frame_indices = np.arange(0, total_steps, frame_stride)
    state = {'i': 0}

    def advance():
        frame_idx = frame_indices[state['i']]
        temps = np.full(base_mesh.n_cells, np.nan)
        frame_data = all_temps[valid, frame_idx]
        frame_data = np.where(frame_data < VMIN, np.nan, frame_data)
        temps[wall_ids[valid]] = frame_data
        base_mesh.cell_data['temps'] = temps
        p.add_text(f"Time Step: {frame_idx}", name="time_label", font_size=12)
        state['i'] = (state['i'] + 1) % len(frame_indices)

    p.add_callback(advance, interval=int(1000 / fps))
    p.app.exec_()

# Plot the surface temperature distribution in a 3D subplot for static plots
def plot_3d_subplot(ax, mesh, r0, z0, phicam, title, vmin, vmax, option):
    p = pv.Plotter(off_screen=True)
    m_plot = mesh.copy()
    
    if option == "binary":
        # Values > VMAX (3695K) are 1 (Red), everything else is NaN (Light Grey)
        raw_temps = m_plot.cell_data['temps']
        binary_data = np.where(raw_temps > VMAX, 1.0, np.nan)
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

# 2D subplot (toroidal projection) for temperature distribution
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
        norm_bin = mpl.colors.BoundaryNorm([0, VMAX, 10000], cmap_bin.N)
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
    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--wall_file", help = "Path to wall geometry HDF5 file", required=True)
    parser.add_argument("--results_file", help = "Path to the results HDF5 file", required=True)
    parser.add_argument("--interactive", action="store_true",
                         help="Open an interactive 3D window with a free-moving camera "
                              "(colored using --option, or animated over time if used with --live)")
    parser.add_argument("--option", choices=["last_time", "max_temperature", "first_melt", "binary"], default="last_time",
                     help="Plotting option: 'last_time' works with any results file; "
                          "'max_temperature', 'first_melt', and 'binary' require the results file "
                          "to have been produced with --store_all_times")
    parser.add_argument("--phi_cam", type=lambda x: np.radians(float(eval(x, {"np": np, "pi": np.pi}))), default=0.0, help="Camera angle in degrees (e.g., 0, 10, 45, 90).")
    parser.add_argument("--live", action="store_true", help="Animate temperature over time")
    args = parser.parse_args()

    base_mesh = read_wallinput(args.wall_file)

    # Plot the wall interactovely and animated
    if args.interactive and args.live:
        with h5py.File(args.results_file, 'r') as h5:
            wall_ids = (h5['wall_ids'][:] - 1).astype(int)
        plot_3d_interactive_live(args.results_file, wall_ids, base_mesh)

    # Plot the wall interactively (single frame)
    elif args.interactive:
        mesh = read_temps(args.results_file, base_mesh, args.option)
        plot_3d_interactive(mesh, VMIN, VMAX, args.option)

    # Plot the wall animated in a 3D and 2D subplot
    elif args.live:
        from matplotlib.animation import FuncAnimation
        # Live Animation
        with h5py.File(args.results_file, 'r') as h5:
            all_temps = h5['surf_temp'][:]
            wall_ids = (h5['wall_ids'][:] - 1).astype(int)
            if all_temps.ndim == 1 or all_temps.shape[1] < 2:
                sys.exit("Error: --live requires time-resolved surf_temp data "
                         "(re-run FIREWALL with --store_all_times). "
                         "This file only contains the last timestep.")
            total_steps = all_temps.shape[1]
        
        frame_indices = np.arange(0, total_steps, 10)

        # PyVista setup
        p = pv.Plotter(off_screen=True)
        base_mesh.cell_data['temps'] = np.full(base_mesh.n_cells, np.nan)
        p.add_mesh(base_mesh, scalars='temps', cmap='jet', clim=[VMIN, VMAX], 
                   log_scale=True, nan_color='lightgrey', show_scalar_bar=False)
        p.camera.position = (1.9*R0*np.cos(args.phi_cam), 1.9*R0*np.sin(args.phi_cam), Z0 - 1)
        p.camera.focal_point = (1.87*R0*np.cos(args.phi_cam), 1.87*R0*np.sin(args.phi_cam), Z0 - 0.9)
        p.render()
        initial_img = p.screenshot()

        # Matplotlib figure setup
        fig, (ax3d, ax2d) = plt.subplots(1, 2, figsize=(16, 7))
        ax2d.set_facecolor("black")
        im3d = ax3d.imshow(p.screenshot())
        ax3d.axis("off")

        centers = base_mesh.cell_centers().points
        phi_deg = np.rad2deg(np.arctan2(centers[:, 1], centers[:, 0])) % 360
        z_coords = centers[:, 2]

        # Precompute bin assignment for each cell
        nx, ny = 180, 90
        xedges = np.linspace(0, 360, nx + 1)
        yedges = np.linspace(1.3, 4.6, ny + 1)
        ix = np.clip(np.digitize(phi_deg, xedges) - 1, 0, nx - 1)
        iy = np.clip(np.digitize(z_coords, yedges) - 1, 0, ny - 1)

        # Initial 2D Plot
        im2d = ax2d.imshow(np.full((90, 180), VMIN).T, origin='lower', extent=[0, 360, 1.3, 4.6],
                           cmap='jet', norm=mpl.colors.LogNorm(vmin=VMIN, vmax=VMAX), aspect='auto')
        ax2d.set_xlabel("Toroidal angle [deg]")
        ax2d.set_ylabel("Z [m]")

        # --- Single shared colorbar for both subplots ---
        ticks_to_show = [t for t in [300, 500, 1000, 1500, 2500, VMAX] if t <= VMAX] or [VMIN, VMAX]
        norm = mpl.colors.LogNorm(vmin=VMIN, vmax=VMAX)
        sm = mpl.cm.ScalarMappable(norm=norm, cmap='jet')

        cbar = fig.colorbar(
            sm,
            ax=[ax3d, ax2d],
            orientation='vertical',
            fraction=0.03,
            pad=0.02,
            ticks=ticks_to_show
        )
        cbar.ax.set_yticklabels([str(t) for t in ticks_to_show])
        cbar.set_label('Surface temperature [K]')

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
            
            grid = np.full((nx, ny), -np.inf)
            np.maximum.at(grid, (ix, iy), current_frame_temps)
            grid[np.isinf(grid)] = VMIN  # empty bins -> vmin, same as static's nan_to_num
            im2d.set_array(grid.T)
            
            fig.suptitle(f"Time Step: {frame_idx}", fontsize=16)
            return im3d, im2d

        ani = FuncAnimation(fig, update, frames=frame_indices, interval=1, blit=False)
        plt.show()
        p.close()

    # Plot the wall in a 3D and 2D subplot (single frame)
    else:
        mesh = read_temps(args.results_file, base_mesh, args.option)
        temps = mesh.cell_data['temps']
        # --- 3D Plotting (Single Plot) ---
        fig3d, ax3d = plt.subplots(figsize=(10, 10))
        plot_3d_subplot(ax3d, mesh, R0, Z0, args.phi_cam, "3D View", VMIN, VMAX, args.option)

        if args.option != "binary":
            # Simplified colorbar for a single plot
            norm = mpl.colors.LogNorm(vmin=VMIN, vmax=VMAX)
            sm = mpl.cm.ScalarMappable(norm=norm, cmap='jet')
            ticks_to_show = [300, 500, 1000, 1500, 2500, VMAX]
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
        fig2d, ax2d = plt.subplots(figsize=(10, 6)) 
        im2d = plot_dist_subplot(ax2d, mesh, "Temperature Distribution", VMIN, VMAX, args.option)
        
        if args.option != "binary":
            ticks_to_show = [300, 500, 1000, 1500, 2500, 3695]
            cb2d = fig2d.colorbar(im2d, ax=ax2d, fraction=0.1, aspect=15, ticks=ticks_to_show)
            cb2d.ax.set_yticklabels([str(t) for t in ticks_to_show])
            cb2d.set_label('Surface temperature [K]')
        
        fig2d.tight_layout()
        plt.show()

