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
    "font.size": 13,
    "axes.labelsize": 15
})

# Fixed Parameters
R0, Z0 = 5.2, 0.1
VMIN, VMAX_DEFAULT = 301, 1500 # Adjust VMAX_DEFAULT as needed

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
    ax.text(0.02, 0.98, title, transform=ax.transAxes, 
            color='white', fontsize=15, fontweight='bold', va='top', ha='left')
    p.close()

def plot_dist_subplot(ax, mesh, title, vmin, vmax, option):
    centers = mesh.cell_centers().points
    phi_deg = np.rad2deg(np.arctan2(centers[:, 1], centers[:, 0])) % 360
    z = centers[:, 2]
    
    counts, xedges, yedges = np.histogram2d(phi_deg, z, bins=(180, 90))
    temp_sum, _, _ = np.histogram2d(phi_deg, z, bins=(180, 90), weights=mesh.cell_data['temps'])
    
    with np.errstate(divide='ignore', invalid='ignore'):
        avg_temp = np.where(counts > 0, temp_sum / counts, vmin)

    ax.set_facecolor("black")
    X, Y = np.meshgrid(xedges, yedges)

    if option == "binary":
        cmap_bin = mpl.colors.ListedColormap(['black', 'red'])
        norm_bin = mpl.colors.BoundaryNorm([0, 3695, 10000], cmap_bin.N)
        
        im = ax.pcolormesh(X, Y, avg_temp.T, shading='auto', 
                           cmap=cmap_bin, norm=norm_bin)
    else:   
        im = ax.pcolormesh(X, Y, avg_temp.T, shading='auto', cmap='jet',
                       norm=mpl.colors.LogNorm(vmin=vmin, vmax=vmax))

    ax.set_box_aspect(1)
    ax.set_title(title, fontsize=15)
    ax.set_xlim(0, 360)
    ax.set_ylim(1.3,4.6)
    ax.set_xticks([0, 90, 180, 270, 360])
    ax.set_xlabel("Toroidal angle [deg]")
    if ax.get_subplotspec().is_first_col():
        ax.set_ylabel("Z [m]")
    
    return im


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--wall_file")
    parser.add_argument("--results_files", nargs=4)
    parser.add_argument("--option", choices=["last_time", "max_temperature", "first_melt", "binary"], default="last_time")
    args = parser.parse_args()

    base_mesh = read_wallinput(args.wall_file)
    fnames = ["D1", "D2", "D3", "D4"]
    phi_cams = [290 * np.pi / 180, 10 * np.pi / 180, 190 * np.pi / 180, 190 * np.pi / 180]
    
    meshes = [read_temps(f, base_mesh, args.option) for f in args.results_files]


    # Find global max across BOTH datasets for a consistent color scale
    all_temps = [m.cell_data['temps'] for m in meshes]
    global_vmax = max([t.max() for t in all_temps])

   # --- 3D Plotting ---
    fig3d, axes3d = plt.subplots(2, 2, figsize=(10, 10))
    for i, ax in enumerate(axes3d.flat):
        plot_3d_subplot(ax, meshes[i], R0, Z0, phi_cams[i], fnames[i], VMIN, global_vmax, args.option)

    if args.option != "binary":
        # Global Horizontal Colorbar
        cax = fig3d.add_axes([0.25, 0.78, 0.5, 0.03])
        norm = mpl.colors.LogNorm(vmin=VMIN, vmax=global_vmax)
        cb = fig3d.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap='jet'), 
                        cax=cax, orientation='horizontal')
        if args.option!="max_temperature":
            cb.set_label('Surface temperature [K]', labelpad=12)
        else:
            cb.set_label('Maximal surface temperature [K]', labelpad=12)
        # Ticks and Labels on Top
        cax.xaxis.set_ticks_position('top')
        cax.xaxis.set_label_position('top')
        fig3d.tight_layout(rect=[0, 0, 1, 0.78], h_pad=6.0)
    else:
        fig3d.tight_layout(h_pad=6.0)

    # --- 2D Plotting ---
    fig2d, axes2d = plt.subplots(1, 4, figsize=(17, 5), sharey=True)
    last_im = None
    for i, ax in enumerate(axes2d):
        last_im = plot_dist_subplot(ax, meshes[i], fnames[i], VMIN, global_vmax, args.option)
    
    if args.option != "binary":
        fig2d.subplots_adjust(right=0.88, top=0.85, wspace=0.1) 
        cax_2d = fig2d.add_axes([0.91, 0.15, 0.02, 0.7]) 
        fig2d.colorbar(last_im, cax=cax_2d, label='Surface temperature [K]')
    else:
        fig2d.tight_layout()

    plt.show()
