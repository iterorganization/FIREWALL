#!/usr/bin/env python3

import argparse
import sys
import os
import numpy as np
import h5py
import pyvista as pv
import matplotlib.pyplot as plt
import matplotlib as mpl

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 13
})

# Fixed Parameters
R0, Z0 = 5.2, 0.1
VMIN = 301
VMAX_TEMP = 3695

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

def get_mesh_with_data(results, base_mesh, option):
    m = base_mesh.copy()
    m.cell_data['temps'] = np.zeros(base_mesh.n_cells)
    with h5py.File(results, 'r') as h5:
        wallid = (h5['wall_ids'][:] - 1).astype(int)
        if option == "last_time":
            data = h5['surf_temp'][:,-1]
        elif option == 'energy_fraction':
            # Convert fraction to percentage
            data = h5['energy_fraction'][:] * 100
        
        valid = (wallid >= 0) & (wallid < base_mesh.n_cells)
        m.cell_data['temps'][wallid[valid]] = data[valid]
    return m

def plot_3d_subplot(ax, mesh, r0, z0, phicam, option):
    p = pv.Plotter(off_screen=True)
    m_plot = mesh.copy()
    
    if option == "energy_fraction":
        vals = m_plot.cell_data['temps'].copy()
        vals[vals <= 0.01] = np.nan # threshold for %
        m_plot.cell_data['plot_scalars'] = vals
        p.add_mesh(m_plot, scalars='plot_scalars', cmap='RdYlGn', 
                   clim=[0, 100], show_scalar_bar=False, nan_color='lightgrey')
    else: 
        temps = m_plot.cell_data['temps'].copy()
        temps[temps <= VMIN] = np.nan
        m_plot.cell_data['plot_scalars'] = temps
        p.add_mesh(m_plot, scalars='plot_scalars', cmap='jet', 
                   clim=[VMIN, VMAX_TEMP], log_scale=True, 
                   show_scalar_bar=False, nan_color='lightgrey')

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

    if option == "energy_fraction":
        # Values <= 0.01 are treated as "no data"
        mask = temps > 0.01
    else:
        # Values <= VMIN (301) are treated as "no data"
        mask = temps > vmin

    phi_valid = phi_deg[mask]
    z_valid = z[mask]
    temps_valid = temps[mask]

    from scipy.stats import binned_statistic_2d

    if option == 'energy_fraction':
        bin_values, xedges, yedges, _ = binned_statistic_2d(
            phi_valid, z_valid, temps_valid, 
            statistic='max', 
            bins=[180, 90],
            range=[[0, 360], [1.3, 4.6]]
        )
    else:
        bin_values, xedges, yedges, _ = binned_statistic_2d(
            phi_valid, z_valid, temps_valid, 
            statistic='max', 
            bins=[180, 90],
            range=[[0, 360], [1.3, 4.6]]
        )

    ax.set_facecolor("black")


    masked_values = np.ma.masked_invalid(bin_values)
    X, Y = np.meshgrid(xedges, yedges)

    if option == "energy_fraction":
        # Pass the masked_values.T instead of bin_values.T
        im = ax.pcolormesh(X, Y, masked_values.T, shading='flat', 
                           cmap='RdYlGn', vmin=vmin, vmax=vmax)
    else:   
        im = ax.pcolormesh(X, Y, masked_values.T, shading='flat', cmap='jet',
                           norm=mpl.colors.LogNorm(vmin=vmin, vmax=vmax))

    ax.set_box_aspect(1)
    #ax.set_title(title, fontsize=15)
    ax.set_xlim(0, 360)
    ax.set_ylim(1.3, 4.6)
    ax.set_xticks([0, 90, 180, 270, 360])
    ax.set_xlabel("Toroidal angle [deg]")
    if ax.get_subplotspec().is_first_col():
        ax.set_ylabel("Z [m]")
    
    return im

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--wall_file", required=True)
    parser.add_argument("--results_files", nargs=2)
    args = parser.parse_args()

    base_mesh = read_wallinput(args.wall_file)
    
    j2_file, j4_file = args.results_files[0], args.results_files[1]
    phi_j2, phi_j4 = 10 * np.pi / 180, 190 * np.pi / 180

    m_j2_t = get_mesh_with_data(j2_file, base_mesh, "last_time")
    m_j2_e = get_mesh_with_data(j2_file, base_mesh, "energy_fraction")
    m_j4_t = get_mesh_with_data(j4_file, base_mesh, "last_time")
    m_j4_e = get_mesh_with_data(j4_file, base_mesh, "energy_fraction")

    fig, axes = plt.subplots(2, 2, figsize=(12, 11))

    plot_3d_subplot(axes[0, 0], m_j2_t, R0, Z0, phi_j2, "last_time")
    plot_3d_subplot(axes[0, 1], m_j2_e, R0, Z0, phi_j2, "energy_fraction")
    plot_3d_subplot(axes[1, 0], m_j4_t, R0, Z0, phi_j4, "last_time")
    plot_3d_subplot(axes[1, 1], m_j4_e, R0, Z0, phi_j4, "energy_fraction")

    plt.tight_layout(rect=[0, 0.0, 1, 0.96], h_pad=0.05)

    def get_col_center_and_width(ax):
        bbox = ax.get_position()
        return bbox.x0, bbox.width

    cbar_width = 0.30
    cbar_height = 0.02
    top_y = 0.92

    # --- Left Column: Temperature ---
    x0_left, w_left = get_col_center_and_width(axes[0, 0])
    cax1_x = x0_left + (w_left - cbar_width) / 2
    cax1 = fig.add_axes([cax1_x, top_y, cbar_width, cbar_height])
    
    # Defining explicit ticks for the log scale
    temp_ticks = [300, 500, 1000, 2000, 3695]
    sm1 = mpl.cm.ScalarMappable(norm=mpl.colors.LogNorm(300, VMAX_TEMP), cmap='jet')
    cb1 = fig.colorbar(sm1, cax=cax1, orientation='horizontal', ticks=temp_ticks)
    cb1.ax.set_xticklabels([str(t) for t in temp_ticks])
    cb1.set_label("Surface Temperature [K]", labelpad=-50)
    cax1.xaxis.set_ticks_position('top')

    # --- Right Column: Energy Fraction (%) ---
    x0_right, w_right = get_col_center_and_width(axes[0, 1])
    cax2_x = x0_right + (w_right - cbar_width) / 2
    cax2 = fig.add_axes([cax2_x, top_y, cbar_width, cbar_height])
    
    sm2 = mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(0, 100), cmap='RdYlGn')
    cb2 = fig.colorbar(sm2, cax=cax2, orientation='horizontal')
    cb2.set_label("Energy Fraction [%]", labelpad=-50)
    cax2.xaxis.set_ticks_position('top')

    plt.show()


    # --- NEW FIGURE: 1x2 Temperature Distribution Plots ---
    fig_dist, axes_dist = plt.subplots(1, 2, figsize=(12, 6))

    # Plot J2 and J4 temperature distributions
    im_j2_t = plot_dist_subplot(axes_dist[0], m_j2_t, "", VMIN, VMAX_TEMP, "last_time")
    im_j4_t = plot_dist_subplot(axes_dist[1], m_j4_t, "", VMIN, VMAX_TEMP, "last_time")

    # Tighten layout, leaving more room at the top (0.80 instead of 0.85)
    plt.tight_layout(rect=[0.02, 0.02, 0.98, 0.85])

    # Shared colorbar for the two plots
    # Increased height from 0.03 to 0.04 to give text more breathing room
    cbar_ax = fig_dist.add_axes([0.25, 0.86, 0.5, 0.04]) 
    
    temp_ticks = [300, 500, 1000, 2000, 3695]
    cb_dist_t = fig_dist.colorbar(im_j2_t, cax=cbar_ax, orientation='horizontal', ticks=temp_ticks)
    
    cb_dist_t.ax.set_xticklabels([str(t) for t in temp_ticks], fontsize=10)
    cbar_ax.set_xlim(300, 3695)
    
    # Adjust labelpad to pull the label away from the ticks (positive values move it up)
    cb_dist_t.set_label("Max Surface Temperature [K]", labelpad=10)
    
    cbar_ax.xaxis.set_ticks_position('top')
    cbar_ax.xaxis.set_label_position('top')

    plt.show()
