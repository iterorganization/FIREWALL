#!/usr/bin/env python3

import argparse
import sys
import os
import numpy as np
import h5py
import pyvista as pv
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy.ma as ma
from matplotlib.colors import LogNorm

##############################
# Fixed Parameters
##############################
# These remain hardcoded as they are physics/geometry constants
# not typically changed per file run.
GROUP    = "001"   # HDF5 load group
R0       = 5.2
Z0       = 0.1

##############################
# Read wall mesh ITER
##############################

def read_wallinput(wallin):
    if not os.path.exists(wallin):
        sys.exit(f"Error: Wall geometry file '{wallin}' not found.")

    print(f"Reading geometry from: {wallin}")
    with h5py.File(wallin,'r') as h5:
        data = h5["nodes"][:]
        ntri = int(h5["ntriangle"][0])

    verts = []
    faces = []
    warea = np.zeros(ntri)

    for i in range(ntri):
        p1 = data[i*9+0:i*9+3]
        p2 = data[i*9+3:i*9+6]
        p3 = data[i*9+6:i*9+9]

        verts += [p1, p2, p3]
        faces += [[3, i*3+0, i*3+1, i*3+2]]

        a = p2 - p1
        b = p3 - p1
        warea[i] = np.linalg.norm(np.cross(a, b)) / 2
    
    print('mean area', np.mean(warea))
    print('mean length', np.sqrt(2*np.mean(warea)))
    verts = np.array(verts)
    mesh = pv.PolyData(verts, faces)

    # Empty cell data fields
    mesh.cell_data['temps'] = np.zeros(ntri)

    return mesh, warea


##############################
# Read temperature data
##############################

def read_temps(results, mesh):
    if not os.path.exists(results):
        sys.exit(f"Error: Results file '{results}' not found.")

    print(f"Reading results from: {results}")
    with h5py.File(results,'r') as h5:
        # Check if wall_ids exists to avoid cryptic errors
        if 'wall_ids' not in h5 or 'surf_temp' not in h5:
             sys.exit("Error: HDF5 file missing 'wall_ids' or 'surf_temp' datasets.")
        
        wallid = h5['wall_ids'][:] - 1
        wallid = wallid.astype(int)
        temps  = h5['surf_temp'][:]
        
        # largest_10 = np.sort(temps)[-50:]
        # print(largest_10)

    # Ensure indices are within bounds
    if np.max(wallid) >= mesh.n_cells:
        print(f"Warning: Max wall_id ({np.max(wallid)}) exceeds mesh cells ({mesh.n_cells}). Clipping.")
        valid_mask = wallid < mesh.n_cells
        wallid = wallid[valid_mask]
        temps = temps[valid_mask]

    mesh.cell_data['temps'][wallid] = temps

    return mesh

##############################
# 3D plot ITER fix
##############################

def plotmesh(mesh, r0, z0, phicam=10*np.pi/180, temps='temps'):
    cmap = mpl.colormaps["plasma"].copy()
    cmap.set_bad(color=[0.9, 0.9, 0.9])

    a = mesh.cell_data[temps]
    
    # Clip for visualization
    a_clipped = np.clip(a, 301, a.max()) 

    log_norm = mpl.colors.LogNorm(vmin=301, vmax=a_clipped.max())

    p = pv.Plotter(off_screen=True)
    p.add_mesh(
        mesh,
        scalars=temps,
        cmap=cmap,
        clim=[301, a_clipped.max()],
        log_scale=True,
        show_scalar_bar=False
    )

    p.camera.position = (1.9*r0*np.cos(phicam), 1.9*r0*np.sin(phicam), z0 - 1)
    p.camera.focal_point = (1.87*r0*np.cos(phicam), 1.87*r0*np.sin(phicam), z0 - 0.9)

    p.show()
    
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(p.image)
    ax.axis("off")

    sm = mpl.cm.ScalarMappable(
        cmap=cmap,
        norm=log_norm
    )
    sm.set_array([])

    cbar = fig.colorbar(
        sm,
        ax=ax,
        orientation="horizontal",
        location="top",
        pad=0.02,
        fraction=0.05
    )

    cbar.set_label(r"Temperature (K)", labelpad=8)
    cbar.ax.tick_params(which="both", direction="in")

##############################
# Distribution plot
##############################

def plotdistribution(mesh, bins=(360, 180), scalar_name='temps'):
    """
    Plots the temperature distribution as a function of Toroidal Angle and Z.
    """
    centers = mesh.cell_centers().points
    x, y, z = centers[:, 0], centers[:, 1], centers[:, 2]

    phi = np.arctan2(y, x)
    phi_deg = np.rad2deg(phi) % 360  # Convert to [0, 360]

    temp_data = mesh.cell_data[scalar_name]

    counts, xedges, yedges = np.histogram2d(phi_deg, z, bins=bins)
    temp_sum, _, _ = np.histogram2d(phi_deg, z, bins=bins, weights=temp_data)

    with np.errstate(divide='ignore', invalid='ignore'):
        avg_temp = temp_sum / counts

    fig, ax = plt.subplots(figsize=(6, 6))

    avg_temp = np.ma.masked_where(counts == 0, avg_temp)

    cmap = mpl.colormaps["plasma"].copy() 
    
    # Fill empty bins with background temp for continuity if desired, or keep masked
    avg_temp_filled = np.where(counts == 0, 301, avg_temp)

    ax.set_facecolor("black")

    X, Y = np.meshgrid(xedges, yedges)

    im = ax.pcolormesh(X, Y, avg_temp_filled.T, 
                       shading='auto', 
                       cmap=cmap,
                       norm=mpl.colors.LogNorm(vmin=301, vmax=np.nanmax(temp_data)))

    ax.set_xlabel("Toroidal angle [deg]")
    ax.set_ylabel("Z [m]")
    
    ax.set_xlim(0, 360)
    ax.set_ylim(0.5, 4.6)
    ax.set_xticks([0, 90, 180, 270, 360])
    
    return fig, ax

##############################
# Main
##############################

if __name__ == "__main__":
    # Setup Argument Parser
    parser = argparse.ArgumentParser(
        description="Visualize ITER wall temperatures from HDF5 data."
    )
    
    parser.add_argument(
        "wall_file", 
        help="Path to the HDF5 wall geometry file (e.g., newiterwall_offset10cm.h5)"
    )
    parser.add_argument(
        "results_file", 
        help="Path to the HDF5 results file (e.g., results.h5)"
    )
    parser.add_argument(
        "--output", "-o", 
        default="output.vtk", 
        help="Filename to save the VTK output (default: output.vtk). Pass 'none' to skip saving."
    )

    args = parser.parse_args()

    # Execution
    mesh, warea = read_wallinput(args.wall_file)
    mesh = read_temps(args.results_file, mesh)
    
    print("Generating plots...")
    plotmesh(mesh, R0, Z0)
    plotdistribution(mesh)

    plt.show()

    # Save logic
    if args.output.lower() != "none":
        print(f"Saving VTK to {args.output}...")
        mesh.save(args.output)
    else:
        print("Skipping VTK save.")