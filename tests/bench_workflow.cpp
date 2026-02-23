#include <hdf5.h>
#include <omp.h>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <iterator>  // for back_inserter
#include <numeric>
#include <span>  // Requires C++20. If using C++17, use a polyfill or raw pointers.
#include <string>
#include <vector>

#include "ArgParser.h"
#include "ConfigParser.h"
#include "Interpolator.h"
#include "NormVec.h"
#include "Particle.h"
#include "PhysConst.h"
#include "Solver.h"
#include "Utils.h"

using fpType = double;
using v = std::vector<fpType>;
using vv = std::vector<std::vector<fpType>>;
using vvv = std::vector<std::vector<std::vector<fpType>>>;

std::vector<std::pair<size_t, size_t>> get_particle_ranges(const Particles& particles, const std::vector<int>& selected_wall_ids) {
    int N_select = selected_wall_ids.size();
    std::vector<std::pair<size_t, size_t>> ranges(N_select);
    size_t current_idx = 0;
    for (int i = 0; i < N_select; ++i) {
        int uid = selected_wall_ids[i];

        // Fast forward to the specific ID (handling gaps if user filtered IDs)
        while (current_idx < particles.n_particles && particles.wall_id[current_idx] < uid) {
            current_idx++;
        }

        size_t r_start = current_idx;
        while (current_idx < particles.n_particles && particles.wall_id[current_idx] == uid) {
            current_idx++;
        }
        ranges[i] = {r_start, current_idx};
    }
    return ranges;
}

std::vector<double> construct_target_depths(const SimulationParams& params) {
    double L_1 = params.L / params.L_sub;
    double L_2 = params.L - L_1;
    int N_x1 = static_cast<int>(L_1 / params.delta_x1);
    int N_x2 = static_cast<int>(L_2 / params.delta_x2);

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
    return target_depths;
}

struct Result {
    int wall_id;
    std::vector<double> cum_E_dep;
    bool store_full = false;
    double first_impact_time = -1.0;
    double melting_time = -1.0;
    std::vector<double> profile_at_melting;
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
        std::cout << "  Walls:       All elements\n";
    } else {
        std::cout << "  Walls:       Processing " << args.wallIds.size() << " wall elements" << "\n";
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

    std::vector<int> unique_wall_ids = particles.get_unique_wall_ids();

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

    std::vector<int> selected_full_profile_wall_ids = selected_wall_ids;  // By default, store full profiles for all selected walls

    std::cout << "Processing " << selected_wall_ids.size() << " elements using OpenMP..." << std::endl;
    std::cout << "Max threads: " << omp_get_max_threads() << std::endl;

    // Identify ranges for each unique ID in valid_filtered_indices
    std::vector<std::pair<size_t, size_t>> ranges = get_particle_ranges(particles, selected_wall_ids);

    // Target Depth Grid
    std::vector<double> target_depths = construct_target_depths(params);

    std::vector<double> times;
    times.push_back(params.t_start);
    while (times.back() < params.t_end) times.push_back(times.back() + (times.back() >= params.t_interm && params.t_interm > params.t_start ? params.dt_large : params.dt_small));

    const Interpolator interpolator(args.interpPath);
    const Solver solver({}, target_depths);
    std::vector<Result> results(selected_wall_ids.size());

    // --- Parallel Loop ---
    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < (int)selected_wall_ids.size(); ++i) {
        vv out_T(times.size(), v(target_depths.size(), 0.0));

        // Save T[:, 0]
        for (int xi = 0; xi < target_depths.size(); xi++)
            out_T[0][xi] = params.T_ini;

        const int wid = selected_wall_ids[i];
        const size_t start = ranges[i].first;
        const size_t end = ranges[i].second;
        const size_t count = end - start;

        bool keep_full = std::find(selected_full_profile_wall_ids.begin(),
                                   selected_full_profile_wall_ids.end(),
                                   wid) != selected_full_profile_wall_ids.end();

        // Initialize result structure for this wall element
        results[i].wall_id = wid;
        results[i].cum_E_dep.assign(target_depths.size(), 0.0);

        if (count == 0) continue;

        const NormVec norm_vec(wall.data(), wid * 9);
        const double area = 0.5 * norm_vec.len;
        const double coeff = 1.0 / (area * params.t_dep);

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

        // --- Cumulative Energy Deposition Calculation ---
        for (size_t k = start; k < end; ++k) {
            // Get the energy profile for this specific particle's energy and angle
            std::vector<double> prof = interpolator.getProfile(particles.energy[k], particles.angle[k], target_depths);

            // Accumulate: E_cum = Sum(Profile * weight)
            for (size_t d = 0; d < target_depths.size(); ++d) {
                results[i].cum_E_dep[d] += 0.1 * prof[d] * PhysConst::MeV_mm_to_J_m * particles.weight[k];
            }
        }

        std::vector<double> weights(1, 1.0);
        std::vector<double> coll_times(1, 0.0);
        solver.solve(results[i].cum_E_dep, weights, coll_times, params, coeff, times, out_T);

        const double T_melt = 3695.0;
        bool found_melt = false;

        for (size_t t_idx = 0; t_idx < times.size(); ++t_idx) {
            for (size_t d_idx = 0; d_idx < target_depths.size(); ++d_idx) {
                if (out_T[t_idx][d_idx] >= T_melt) {
                    results[i].melting_time = times[t_idx];
                    results[i].profile_at_melting = out_T[t_idx];  // Copy the whole depth profile at this time
                    found_melt = true;
                    break;
                }
            }
            if (found_melt) break;  // Only need the *first* time it hits melting
        }
    }

    // --- Write Results ---
    std::cout << "Writing results to " << args.outPath << "..." << std::endl;
    hid_t resFile = H5Fcreate(args.outPath.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    if (resFile < 0) throw std::runtime_error("Failed to create result file");

    // Prepare flat data for HDF5 output
    std::vector<double> wall_ids_out(selected_wall_ids.size());
    std::vector<double> first_impacts(selected_wall_ids.size());
    std::vector<double> melting_times(selected_wall_ids.size());
    std::vector<double> all_melt_profiles(selected_wall_ids.size() * target_depths.size(), -1.0);
    // Create a 2D matrix flattened into 1D for energy profiles: [num_walls * num_depths]
    std::vector<double> all_cum_E(selected_wall_ids.size() * target_depths.size());

    for (int i = 0; i < selected_wall_ids.size(); ++i) {
        wall_ids_out[i] = static_cast<double>(results[i].wall_id);
        for (size_t d = 0; d < target_depths.size(); ++d) {
            all_cum_E[i * target_depths.size() + d] = results[i].cum_E_dep[d];
        }
        first_impacts[i] = results[i].first_impact_time;
        melting_times[i] = results[i].melting_time;

        if (results[i].melting_time > 0 && !results[i].profile_at_melting.empty()) {
            for (size_t d = 0; d < target_depths.size(); ++d) {
                all_melt_profiles[i * target_depths.size() + d] = results[i].profile_at_melting[d];
            }
        }
    }

    // Write Wall IDs
    hsize_t dims_id[1] = {(hsize_t)selected_wall_ids.size()};
    hid_t space_id = H5Screate_simple(1, dims_id, NULL);
    hid_t ds_id = H5Dcreate2(resFile, "wall_ids", H5T_NATIVE_DOUBLE, space_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    H5Dwrite(ds_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, wall_ids_out.data());

    // Write Melting Times
    hid_t space_melting = H5Screate_simple(1, dims_id, NULL);
    hid_t ds_melting = H5Dcreate2(resFile, "melting_times", H5T_NATIVE_DOUBLE, space_melting, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    H5Dwrite(ds_melting, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, melting_times.data());

    // Write Profiles at First Melting (2D: Walls x Depths)
    hsize_t dims_melt_prof[2] = {(hsize_t)selected_wall_ids.size(), (hsize_t)target_depths.size()};
    hid_t space_melt_prof = H5Screate_simple(2, dims_melt_prof, NULL);
    hid_t ds_melt_prof = H5Dcreate2(resFile, "profiles_at_melting", H5T_NATIVE_DOUBLE, space_melt_prof, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    H5Dwrite(ds_melt_prof, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, all_melt_profiles.data());

    // Write Cumulative Energy Profiles (2D Dataset: Walls x Depths)
    hsize_t dims_E[2] = {(hsize_t)selected_wall_ids.size(), (hsize_t)target_depths.size()};
    hid_t space_E = H5Screate_simple(2, dims_E, NULL);
    hid_t ds_E = H5Dcreate2(resFile, "cum_energy_dep", H5T_NATIVE_DOUBLE, space_E, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    H5Dwrite(ds_E, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, all_cum_E.data());

    // Write Target Depths (1D Dataset)
    hsize_t dims_depth[1] = {(hsize_t)target_depths.size()};
    hid_t space_depth = H5Screate_simple(1, dims_depth, NULL);
    hid_t ds_depth = H5Dcreate2(resFile, "target_depths", H5T_NATIVE_DOUBLE, space_depth, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    H5Dwrite(ds_depth, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, target_depths.data());

    // Write times (1D Dataset)
    hsize_t dims_times[1] = {(hsize_t)times.size()};
    hid_t space_times = H5Screate_simple(1, dims_times, NULL);
    hid_t ds_times = H5Dcreate2(resFile, "times", H5T_NATIVE_DOUBLE, space_times, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    H5Dwrite(ds_times, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, times.data());

    // Cleanup
    H5Dclose(ds_id);
    H5Dclose(ds_E);
    H5Sclose(space_id);
    H5Sclose(space_E);
    H5Dclose(ds_depth);
    H5Sclose(space_depth);
    H5Dclose(ds_times);
    H5Sclose(space_times);
    H5Dclose(ds_melt_prof);
    H5Sclose(space_melt_prof);
    H5Fclose(resFile);

    std::cout << "Done." << std::endl;
    return 0;
}
