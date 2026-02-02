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
#include <span> // Requires C++20. If using C++17, use a polyfill or raw pointers.

#include "ConfigParser.h"
#include "Interpolator.h"
#include "Solver.h"
#include "Utils.h"
#include "ArgParser.h"
#include "PhysConst.h"

using fpType = double;
using v = std::vector<fpType>;
using vv = std::vector<std::vector<fpType>>;
using vvv = std::vector<std::vector<std::vector<fpType>>>;

// --- Helper to physically reorder vectors ---
template <typename T>
void apply_permutation(std::vector<T>& data, const std::vector<size_t>& p_indices) {
    if (data.size() != p_indices.size()) return; // Safety check
    std::vector<T> sorted_data(data.size());
    
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < data.size(); ++i) {
        sorted_data[i] = data[p_indices[i]];
    }
    data.swap(sorted_data);
}

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

        std::vector<int> i_elm = Utils::readH5IntDatasetGroup(partGroup, "i_elm");
        size_t n_particles = i_elm.size();

        printf("Total particles in file: %zu\n", n_particles);

        t_loss = Utils::readH5DoubleDatasetGroup(partGroup, "t_loss");
        weight = Utils::readH5DoubleDatasetGroup(partGroup, "weight");

        wall_id.resize(n_particles);
        vx.resize(n_particles);
        vy.resize(n_particles);
        vz.resize(n_particles);
        energy.resize(n_particles);
        angle.resize(n_particles);

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
    }
};

struct Result {
    int wall_id;
    double surf_temp;
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

    // --- Configuration Output ---
    std::cout << "Configuration:\n"
              << "  Config: " << args.configPath << "\n"
              << "  Wall:   " << args.wallPath << "\n"
              << "  Part:   " << args.partPath << "\n"
              << "  Interp: " << args.interpPath << "\n"
              << "  Output: " << args.outPath << "\n";
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
    hid_t wallFile = H5Fopen(args.wallPath.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    if (wallFile < 0) throw std::runtime_error("Failed to open wall file: " + args.wallPath);

    std::vector<double> wall = Utils::readH5DoubleDataset(wallFile, "nodes");
    H5Fclose(wallFile);

    // Particles
    Particles particles(args.partPath);


    // We physically reorder vectors so data for Wall X is contiguous in memory.
    std::cout << "Sorting and reordering particles..." << std::endl;
    
    size_t n_particles = particles.wall_id.size();
    std::vector<size_t> p_indices(n_particles);
    std::iota(p_indices.begin(), p_indices.end(), 0);

    std::sort(p_indices.begin(), p_indices.end(), [&](size_t i, size_t j) {
        return particles.wall_id[i] < particles.wall_id[j] || (particles.wall_id[i] == particles.wall_id[j] && particles.t_loss[i] < particles.t_loss[j]);
    });

    // 2. Apply Permutation to all data vectors
    apply_permutation(particles.wall_id, p_indices);
    apply_permutation(particles.t_loss, p_indices);
    apply_permutation(particles.weight, p_indices);
    apply_permutation(particles.vx, p_indices);
    apply_permutation(particles.vy, p_indices);
    apply_permutation(particles.vz, p_indices);
    apply_permutation(particles.energy, p_indices);
    // Note: 'angle' is not calculated yet, so we don't need to sort it.

    // --- Filter Zero IDs ---
    auto it_first_nonzero = std::find_if(particles.wall_id.begin(), particles.wall_id.end(), [](int id) {
        return id > 0;
    });

    if (it_first_nonzero == particles.wall_id.end()) {
        std::cerr << "No non-zero particles found." << std::endl;
        return 1;
    }

    // Determine the valid range in the sorted arrays
    size_t start_offset = std::distance(particles.wall_id.begin(), it_first_nonzero);
    // We also drop the last element (end-1) matching previous logic
    size_t end_offset = n_particles; 

    if (start_offset >= end_offset) {
        std::cerr << "Not enough particles after filtering." << std::endl;
        return 1;
    }

    // --- Identify Unique IDs ---
    std::vector<int> unique_wall_ids;
    unique_wall_ids.push_back(particles.wall_id[start_offset]);
    for (size_t i = start_offset + 1; i < end_offset; ++i) {
        if (particles.wall_id[i] != particles.wall_id[i - 1]) {
            unique_wall_ids.push_back(particles.wall_id[i]);
        }
    }

    SimulationParams params(args.configPath);

    double L_1 = params.L / params.L_sub;
    double L_2 = params.L - L_1;

    int N_x1 = static_cast<int>(L_1 / params.delta_x1);
    int N_x2 = static_cast<int>(L_2 / params.delta_x2);
    // --- Prepare Interpolator and Material ---
    
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
    std::vector<NormVec> norm_vecs(N_select);
    
    size_t current_idx = start_offset;
    for (int i = 0; i < N_select; ++i) {
        int uid = selected_wall_ids[i];
        
        // Fast forward to the specific ID (handling gaps if user filtered IDs)
        while(current_idx < end_offset && particles.wall_id[current_idx] < uid) {
            current_idx++;
        }
        
        size_t r_start = current_idx;
        while (current_idx < end_offset && particles.wall_id[current_idx] == uid) {
            current_idx++;
        }
        ranges[i] = {r_start, current_idx};
        
        // Compute Normal
        norm_vecs[i] = NormVec(&wall[0], uid * 9);
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
    
    
    int N_t = 0;
    int N_x = target_depths.size();
    
    if (params.t_interm <= params.t_start || params.t_interm >= params.t_end || params.t_interm == 0.0) {
        N_t = static_cast<int>(std::ceil((params.t_end - params.t_start) / params.dt_small) + 1);
        params.t_interm = params.t_end + 1.0;
    } else {
        int Nt1 = static_cast<int>(std::ceil((params.t_interm - params.t_start) / params.dt_small));
        int Nt2 = static_cast<int>(std::ceil((params.t_end - params.t_interm) / params.dt_large));
        N_t = Nt1 + Nt2 + 1;
    }
    
    vvv all_out_T(N_select, vv(N_t, v(N_x, 0.0)));
    v all_out_times(N_t, 0.0);
    
    // Save T[:, 0]
    for (int wid = 0; wid < N_select; wid++) {
        for (int xi = 0; xi < N_x; xi++) {
            all_out_T[wid][0][xi] = params.T_ini;
        }
    }
    
    // Save t=0
    all_out_times[0] = params.t_start;
    
    const Interpolator interpolator(args.interpPath);
    const Solver solver({}, target_depths);  // Empty material for now.
    
    // --- Parallel Loop ---
    #pragma omp parallel for schedule(dynamic)
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
        std::vector<double> dE_dx(target_depths.size() * count);

        for (size_t j = start; j < end; ++j) {
            // Interpolate for this particle
            std::vector<double> prof = interpolator.getProfile(particles.energy[j], particles.angle[j], target_depths);

            // Store in dE_dx (Depth-Major)
            // dE_dx[depth_idx * count + particle_idx]
            for (size_t d = 0; d < prof.size(); ++d) {
                dE_dx[d * count + (j - start)] = prof[d] * PhysConst::MeV_to_J;  // Apply conv_factor2 here to match
                                                                // python passing `conv_factor2 * dE_dx`
            }
        }

        auto weight_view = std::span(particles.weight).subspan(start, count);
        auto t_loss_view = std::span(particles.t_loss).subspan(start, count);

        solver.solve(dE_dx, weight_view, t_loss_view, target_depths, params, coeff, all_out_T[i], all_out_times);

        double surf_temp = -1e20;

        for (size_t tstep = 0; tstep < all_out_times.size(); ++tstep) {
            for (size_t xi = 0; xi < all_out_T[i].size(); ++xi) {
                if (all_out_T[i][tstep][xi] > surf_temp) {
                    surf_temp = all_out_T[i][tstep][xi];
                }
            }
        }

        results[i].wall_id = wid;
        results[i].surf_temp = surf_temp;
    }

    // --- Write Results ---
    std::cout << "Writing results to " << args.outPath << "..." << std::endl;
    hid_t resFile = H5Fcreate(args.outPath.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    if (resFile < 0) throw std::runtime_error("Failed to create result file");

    std::vector<double> wall_ids_out(N_select);
    std::vector<double> surf_temps_out(N_select);

    for (int i = 0; i < N_select; ++i) {
        wall_ids_out[i] = static_cast<double>(results[i].wall_id);
        surf_temps_out[i] = results[i].surf_temp;
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

    H5Fclose(resFile);

    std::cout << "Done." << std::endl;


    return 0;
}
