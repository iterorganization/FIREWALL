#include <hdf5.h>
#include <omp.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>
#include <fstream>
#include <iterator>  // for back_inserter

#include "ConfigParser.h"
#include "Interpolator.h"
#include "Solver.h"
#include "Utils.h"
#include "ArgParser.h"
#include "PhysConst.h"

struct Particles {
    std::vector<int> wall_id;
    std::vector<double> t_loss;
    std::vector<double> weight;
    std::vector<double> vx, vy, vz;
    std::vector<double> energy, angle;
    Particles (std::string partPath){
        hid_t partFile = H5Fopen(partPath.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
        if (partFile < 0) throw std::runtime_error("Failed to open particles file: " + partPath);

        hid_t partGroup = H5Gopen2(partFile, "groups/001", H5P_DEFAULT);
        if (partGroup < 0) {
            H5Fclose(partFile);
            throw std::runtime_error("Failed to open group 'groups/001' in " + partPath);
        }

        size_t n_particles = Utils::getNumParticles(partPath);

        printf("Total particles in file: %zu\n", n_particles);

        t_loss = Utils::readH5DoubleDatasetGroup(partGroup, "t_loss");
        weight = Utils::readH5DoubleDatasetGroup(partGroup, "weight");

        wall_id.resize(n_particles);
        vx.resize(n_particles);
        vy.resize(n_particles);
        vz.resize(n_particles);
        energy.resize(n_particles);
        angle.resize(n_particles);

        std::vector<int> i_elm = Utils::readH5IntDatasetGroup(partGroup, "i_elm");
        std::vector<double> v_flat = Utils::readH5DoubleDatasetGroup(partGroup, "v");  // Nx3 flattened

        #pragma omp parallel for
        for (size_t i = 0; i < n_particles; ++i) {
            wall_id[i] = -i_elm[i];  // Python code flips sign: wetted_sorted = -wetted_elements[sort_idx]
            vx[i] = v_flat[i * 3 + 0];
            vy[i] = v_flat[i * 3 + 1];
            vz[i] = v_flat[i * 3 + 2];
            const double p_norm = std::sqrt(vx[i] * vx[i] + vy[i] * vy[i] + vz[i] * vz[i]);
            const double pc = p_norm * PhysConst::c;
            const double m0c2 = PhysConst::m_e_u * PhysConst::c * PhysConst::c;
            const double E = (std::sqrt(pc * pc + m0c2 * m0c2) - m0c2) * PhysConst::J_to_MeV;
            energy[i] = E;
        }

        H5Gclose(partGroup);
        H5Fclose(partFile);

        // for (size_t i = 0; i < n_particles; ++i) std::cout << "Particle " << i << ": E = " << energy[i] << " eV\n";
    }
};

struct Result {
    int wall_id;
    double surf_temp;
    std::vector<double> snaps;
};

struct NormVec {
    double x, y, z, len;

    NormVec() : x(0), y(0), z(0), len(0) {}

    // Add 'inline' to suggest the compiler paste this code directly into the caller
    inline NormVec(const double* wall, int offset) {
        // 1. Point directly to the data. 
        // This creates no new arrays, just looks at existing memory.
        const double* p0 = &wall[offset];     // or wall + offset
        const double* p1 = &wall[offset + 3];
        const double* p2 = &wall[offset + 6];

        // 2. Calculate vector components using scalars (doubles).
        // Compilers will map these directly to CPU registers (XMM/YMM), 
        // avoiding memory writes entirely.
        double ax = p1[0] - p0[0];
        double ay = p1[1] - p0[1];
        double az = p1[2] - p0[2];

        double bx = p2[0] - p0[0];
        double by = p2[1] - p0[1];
        double bz = p2[2] - p0[2];

        // 3. Cross product
        x = ay * bz - az * by;
        y = az * bx - ax * bz;
        z = ax * by - ay * bx;

        // 4. Normalize
        len = std::sqrt(x * x + y * y + z * z);
        
        // Prevent division by zero if points are identical/collinear
        if (len > 0) {
            double invLen = 1.0 / len; // Multiplication is faster than division
            x *= invLen;
            y *= invLen;
            z *= invLen;
        }
    }
};

int main(int argc, char* argv[]) {
    ArgParser::Args args = ArgParser::parse(argc, argv);
    if (args.help) {
        ArgParser::printUsage(argv[0]);
        return 0;
    }

    std::string configPath = args.configPath;
    std::string wallPath = args.wallPath;
    std::string partPath = args.partPath;
    std::string interpPath = args.interpPath;
    std::string outPath = args.outPath;

    std::cout << "Configuration:\n"
                << "  Config File: " << configPath << "\n"
                << "  Wall File:   " << wallPath << "\n"
                << "  Part File:   " << partPath << "\n"
                << "  Interp File: " << interpPath << "\n"
                << "  Output File: " << outPath << "\n";
    
    if (args.wallIds.empty()) {
        std::cout << "  Walls:       All\n";
    } else {
        std::cout << "  Walls:       ";
        for (size_t i = 0; i < args.wallIds.size(); ++i) {
            std::cout << args.wallIds[i] << (i < args.wallIds.size() - 1 ? ", " : "");
        }
        std::cout << "\n";
    }

    // --- Load Data ---
    std::cout << "Loading data..." << std::endl;

    // Wall
    hid_t wallFile = H5Fopen(wallPath.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    if (wallFile < 0) throw std::runtime_error("Failed to open wall file: " + wallPath);

    std::vector<double> wall = Utils::readH5DoubleDataset(wallFile, "nodes");
    H5Fclose(wallFile);

    // Particles
    Particles particles(partPath);

    // --- Sort and Filter ---
    std::cout << "Sorting..." << std::endl;
    size_t n_particles = particles.wall_id.size();
    std::vector<size_t> p_indices(n_particles);
    std::iota(p_indices.begin(), p_indices.end(), 0);

    std::sort(p_indices.begin(), p_indices.end(), [&](size_t i, size_t j) {
        return particles.wall_id[i] < particles.wall_id[j];
    });

    // Find first non-zero ID
    auto it_first_nonzero = std::find_if(p_indices.begin(), p_indices.end(), [&](size_t idx) {
        return particles.wall_id[idx] != 0;
    });

    if (it_first_nonzero == p_indices.end()) {
        std::cerr << "No non-zero particles found." << std::endl;
        return 1;
    }

    // Slice from idx to end-1 (matching previous logic: copy from it_first_nonzero to end-1)
    if (std::distance(it_first_nonzero, p_indices.end()) <= 1) {
        std::cerr << "Not enough particles after filtering." << std::endl;
        return 1;
    }

    // wall id indices
    std::vector<size_t> valid_filtered_indices(it_first_nonzero, p_indices.end() - 1);

    // Unique IDs
    std::vector<int> unique_wall_ids;
    if (!valid_filtered_indices.empty()) {
        unique_wall_ids.push_back(particles.wall_id[valid_filtered_indices[0]]);
        for (size_t i = 1; i < valid_filtered_indices.size(); ++i) {
            if (particles.wall_id[valid_filtered_indices[i]] != particles.wall_id[valid_filtered_indices[i - 1]]) {
                unique_wall_ids.push_back(particles.wall_id[valid_filtered_indices[i]]);
            }
        }
    }

    SimulationParams params(configPath);

    double L_1 = params.L / params.L_sub;
    double L_2 = params.L - L_1;

    int N_x1 = static_cast<int>(L_1 / params.delta_x1);
    int N_x2 = static_cast<int>(L_2 / params.delta_x2);
    // --- Prepare Interpolator and Material ---
    Interpolator interpolator(interpPath);
    Material material;
    Solver solver(material);

    std::vector<int> selected_wall_ids;

    if (args.wallIds.empty()) {
        selected_wall_ids = unique_wall_ids;
    } else {
        std::copy_if(unique_wall_ids.begin(), unique_wall_ids.end(), std::back_inserter(selected_wall_ids),
            [&](int id) { return std::find(args.wallIds.begin(), args.wallIds.end(), id) != args.wallIds.end(); });

        if (selected_wall_ids.empty()) std::cerr << "Warning: None of the requested wall IDs were found.\n";
    }
    

    int N_select = selected_wall_ids.size();

    std::cout << "Processing " << N_select << " elements using OpenMP..." << std::endl;
    std::cout << "Max threads: " << omp_get_max_threads() << std::endl;

    std::vector<Result> results(N_select);

    // Identify ranges for each unique ID in valid_filtered_indices
    std::vector<std::pair<size_t, size_t>> ranges(N_select);

    size_t current_idx = 0;
    for (int i = 0; i < N_select; ++i) {
        int uid = selected_wall_ids[i];
        size_t start = current_idx;
        while (current_idx < valid_filtered_indices.size() && particles.wall_id[valid_filtered_indices[current_idx]] == uid) {
            current_idx++;
        }
        ranges[i] = {start, current_idx};
    }

    std::vector<NormVec> norm_vecs(N_select);
    #pragma omp parallel for 
    for (int i = 0; i < N_select; ++i){
        norm_vecs[i] = NormVec(&wall[0], selected_wall_ids[i] * 9); 
    }

    // Target depths
    std::vector<double> target_depths(N_x1 + N_x2);
    target_depths[0] = 0.0;
    for (int d = 0; d < (N_x1 + N_x2) - 1; ++d) {
        double spacing;
        if (d < N_x1 - 1)
            spacing = params.delta_x1;
        else if (d == N_x1 - 1)
            spacing = 0.5 * (params.delta_x1 + params.delta_x2);
        else
            spacing = params.delta_x2;
        target_depths[d + 1] = target_depths[d] + spacing;
    }

    std::vector<double> target_depths_mm(N_x1 + N_x2);
    double dx1_mm = params.delta_x1 * 1000.0;
    double dx2_mm = params.delta_x2 * 1000.0;

    target_depths_mm[0] = 0.0;
    for (int d = 0; d < (N_x1 + N_x2) - 1; ++d) {
        double spacing;
        if (d < N_x1 - 1)
            spacing = dx1_mm;
        else if (d == N_x1 - 1)
            spacing = 0.5 * (dx1_mm + dx2_mm);
        else
            spacing = dx2_mm;
        target_depths_mm[d + 1] = target_depths_mm[d] + spacing;
    }

// --- Parallel Loop ---
  #pragma omp parallel for 
  for (int i = 0; i < N_select; ++i) {
      int wid = selected_wall_ids[i];
      size_t start = ranges[i].first;
      size_t end = ranges[i].second;
      size_t count = end - start;

      // Offset check was here but NormVec is computed outside now. 
      // We can check validity of wid still if needed, but NormVec construction likely assumed valid.
      
      NormVec norm_vec = norm_vecs[i]; // Use precomputed

      double area = 0.5 * norm_vec.len;
      double coeff = 1.0 / (area * params.t_dep);

      for (size_t k = start; k < end; ++k) {
          const double dot = particles.vx[k] * norm_vec.x + particles.vy[k] * norm_vec.y + particles.vz[k] * norm_vec.z;
          const double p_norm = std::sqrt(particles.vx[k] * particles.vx[k] +
                                          particles.vy[k] * particles.vy[k] +
                                          particles.vz[k] * particles.vz[k]);
          // Angle
          // degrees(arccos(dot/norm)) - 90
          const double ang_rad = std::acos(dot / p_norm);
          const double ang_deg = std::abs(ang_rad * 180.0 / M_PI - 90.0);
          particles.angle[k] = ang_deg;
      }

      // Interpolate Profile
      std::vector<double> dE_dx(target_depths_mm.size() * count);

      for (size_t j = start; j < end; ++j) {
          // Interpolate for this particle
          std::vector<double> prof = interpolator.getProfile(particles.energy[j], particles.angle[j], target_depths_mm);

          // Store in dE_dx (Depth-Major)
          // dE_dx[depth_idx * count + particle_idx]
          for (size_t d = 0; d < prof.size(); ++d) {
              dE_dx[d * count + (j - start)] = prof[d] * PhysConst::MeVmm_to_Jm;  // Apply conv_factor2 here to match
                                                              // python passing `conv_factor2 * dE_dx`
          }
      }

      // Solve Heat Eq
      std::vector<double> out_T, out_times;
      int out_Nx, out_Nt;

      std::vector<double> depths_m(target_depths_mm.size());
      for (size_t d = 0; d < depths_m.size(); ++d) depths_m[d] = target_depths_mm[d] * 1e-3;

      auto weight_view = std::span(particles.weight).subspan(start, count);
      auto t_loss_view = std::span(particles.t_loss).subspan(start, count);

      solver.solve(dE_dx, weight_view, t_loss_view, depths_m, params, coeff, out_T, out_times, out_Nx, out_Nt);

      double max_val = -1e20;
      int max_idx = -1;

      for (size_t idx = 0; idx < out_T.size(); ++idx) {
          if (out_T[idx] > max_val) {
              max_val = out_T[idx];
              max_idx = idx;
          }
      }

      // max_idx = row * Nt + col
      int col = max_idx % out_Nt;
      // row = max_idx / out_Nt;

      // surf_temp = sol[0, col] -> x=0, time=col.
      // index = 0 * Nt + col = col.
      double surf_temp = out_T[col];

      // Snaps
      // snap_idx = linspace(0, Nt-1, 50)
      // snaps = sol[0, snap_idx]
      std::vector<double> snaps(50);
      for (int s = 0; s < 50; ++s) {
          // linspace logic
          // 0 to Nt-1
          int t_idx = static_cast<int>(s * (out_Nt - 1) / 49.0);  // simple linear map
          if (t_idx >= out_Nt) t_idx = out_Nt - 1;
          snaps[s] = out_T[t_idx];  // x=0, time=t_idx -> index=t_idx
      }

      results[i].wall_id = wid;
      results[i].surf_temp = surf_temp;
      results[i].snaps = snaps;
    }

    // --- Write Results ---
    std::cout << "Writing results to " << outPath << "..." << std::endl;
    hid_t resFile = H5Fcreate(outPath.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    if (resFile < 0) throw std::runtime_error("Failed to create result file: " + outPath);

    std::vector<double> wall_ids_out(N_select);
    std::vector<double> surf_temps_out(N_select);
    // temp_snaps is (N_select, 50)
    std::vector<double> temp_snaps_out(N_select * 50);

    for (int i = 0; i < N_select; ++i) {
        wall_ids_out[i] = static_cast<double>(results[i].wall_id);
        surf_temps_out[i] = results[i].surf_temp;
        for (int s = 0; s < 50; ++s) {
            temp_snaps_out[i * 50 + s] = results[i].snaps[s];
        }
    }

    // Write datasets
    hsize_t dims1[1] = {(hsize_t)N_select};
    hid_t space1 = H5Screate_simple(1, dims1, NULL);

    hid_t ds1 = H5Dcreate2(resFile, "wall_ids", H5T_NATIVE_DOUBLE, space1, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    H5Dwrite(ds1, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, wall_ids_out.data());
    H5Dclose(ds1);

    hid_t ds2 = H5Dcreate2(resFile, "surf_temp", H5T_NATIVE_DOUBLE, space1, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    H5Dwrite(ds2, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, surf_temps_out.data());
    H5Dclose(ds2);

    H5Sclose(space1);

    hsize_t dims2[2] = {(hsize_t)N_select, 50};
    hid_t space2 = H5Screate_simple(2, dims2, NULL);
    hid_t ds3 = H5Dcreate2(resFile, "temp_snaps", H5T_NATIVE_DOUBLE, space2, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    H5Dwrite(ds3, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, temp_snaps_out.data());
    H5Dclose(ds3);
    H5Sclose(space2);

    H5Fclose(resFile);

    std::cout << "Done." << std::endl;


    return 0;
}
