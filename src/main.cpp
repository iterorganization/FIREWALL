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
#include "NormVec.h"
#include "Particle.h"

using fpType = double;
using v = std::vector<fpType>;
using vv = std::vector<std::vector<fpType>>;
using vvv = std::vector<std::vector<std::vector<fpType>>>;

// Helper function to get the ranges of particles for each selected wall ID.
std::vector<std::pair<size_t, size_t>> get_particle_ranges(const Particles& particles, const std::vector<int>& selected_wall_ids) {
    int N_select = selected_wall_ids.size();
    std::vector<std::pair<size_t, size_t>> ranges(N_select);
    size_t current_idx = 0;
        for (int i = 0; i < N_select; ++i) {
            int uid = selected_wall_ids[i];
            
            // Fast forward to the specific ID (handling gaps if user filtered IDs)
            while(current_idx < particles.n_particles && particles.wall_id[current_idx] < uid) {
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

// Helper function to construct the target depth grid based on simulation parameters.
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
    
// Helper struct to hold the results for each wall ID.
struct Result {
    int wall_id;
    std::vector<double> surf_temp;
    std::vector<std::vector<double>> full_profile;
    double energy_fraction = 1.0;
    bool store_full = false;
};

// Splash screen for the FIREWALL simulation.
void make_splash() {
    std::cout << "\n";
    std::cout << "  I tell her, \"Baby, baby, baby, baby, baby, baby,\n";
    std::cout << "  baby, baby, baby, I'm a...\"\n";
    std::cout << "\n";
    std::cout << R"(  ███████╗██╗██████╗ ███████╗██╗    ██╗ █████╗ ██╗     ██╗     )" << "\n";
    std::cout << R"(  ██╔════╝██║██╔══██╗██╔════╝██║    ██║██╔══██╗██║     ██║     )" << "\n";
    std::cout << R"(  █████╗  ██║██████╔╝█████╗  ██║ █╗ ██║███████║██║     ██║     )" << "\n";
    std::cout << R"(  ██╔══╝  ██║██╔══██╗██╔══╝  ██║███╗██║██╔══██║██║     ██║     )" << "\n";
    std::cout << R"(  ██║     ██║██║  ██║███████╗╚███╔███╔╝██║  ██║███████╗███████╗)" << "\n";
    std::cout << R"(  ╚═╝     ╚═╝╚═╝  ╚═╝╚══════╝ ╚══╝╚══╝ ╚═╝  ╚═╝╚══════╝╚══════╝)" << "\n";
    std::cout << "\n";
}

int main(int argc, char* argv[]) {
    ArgParser::Args args = ArgParser::parse(argc, argv);
    if (args.help) {
        ArgParser::printUsage(argv[0]);
        return 0;
    }

    make_splash();

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

    if (args.fullProfileWallIds.empty()) {
        std::cout << "  Full Profile Walls:       None\n";
    } else {
        std::cout << "  Full Profile Walls:       ";
        for (size_t i = 0; i < args.fullProfileWallIds.size(); ++i) {
            std::cout << args.fullProfileWallIds[i] << (i < args.fullProfileWallIds.size() - 1 ? ", " : "");
        }
        std::cout << "\n";
    }
    std::cout << "  Store mode:  " << (args.storeAllTimes ? "All timesteps" : "Last timestep only") << "\n";

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

    // --- Prepare Interpolator and Material ---
    
    std::vector<int> selected_wall_ids;
    
    if (args.wallIds.empty()) {
        selected_wall_ids = unique_wall_ids;
    } else {
        std::copy_if(unique_wall_ids.begin(), unique_wall_ids.end(), std::back_inserter(selected_wall_ids),
        [&](int id) { return std::find(args.wallIds.begin(), args.wallIds.end(), id) != args.wallIds.end(); });
        
        if (selected_wall_ids.empty()) std::cerr << "Warning: None of the requested wall IDs were found.\n";
    }

    std::vector<int> selected_full_profile_wall_ids;

    if (args.fullProfileWallIds.empty()) {
        selected_full_profile_wall_ids = {};
    } else {
        std::copy_if(unique_wall_ids.begin(), unique_wall_ids.end(), std::back_inserter(selected_full_profile_wall_ids),
                     [&](int id) { return std::find(args.fullProfileWallIds.begin(), args.fullProfileWallIds.end(), id) != args.fullProfileWallIds.end(); });
    }


    std::cout << "Processing " << selected_wall_ids.size() << " elements using OpenMP..." << std::endl;
    std::cout << "Max threads: " << omp_get_max_threads() << std::endl;
        
    // Identify ranges for each unique ID in valid_filtered_indices
    std::vector<std::pair<size_t, size_t>> ranges = get_particle_ranges(particles, selected_wall_ids);
    
    // Target Depth Grid
    std::vector<double> target_depths = construct_target_depths(params);

    std::vector<double> times;
    times.push_back(params.t_start);
    while (times.back() < params.t_end) times.push_back(times.back() + (times.back() >= params.t_interm && params.t_interm > params.t_start ? params.dt_large : params.dt_small) );
    
    // --- Initialize Interpolator and Solver ---
    const Interpolator interpolator(args.interpPath);
    const Solver solver({}, target_depths);  // Empty material for now.
    std::vector<Result> results(selected_wall_ids.size());



    // --- Parallel Loop ---
    const int n_selected = static_cast<int>(selected_wall_ids.size());
    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < n_selected; ++i) {
        vv out_T (times.size(), v(target_depths.size(), 0.0));
    
        // Save T[:, 0]
        for (size_t xi = 0; xi < target_depths.size(); xi++)
            out_T[0][xi] = params.T_ini;

        const int wid = selected_wall_ids[i];
        const size_t start = ranges[i].first;
        const size_t end = ranges[i].second;
        const size_t count = end - start;

        bool keep_full = std::find(selected_full_profile_wall_ids.begin(),
                                   selected_full_profile_wall_ids.end(),
                                   wid) != selected_full_profile_wall_ids.end();

        
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

        // Interpolate Profile
        std::vector<double> dE_dx(target_depths.size() * count);
        std::vector<double> cum_E_dep_total(target_depths.size());


        for (size_t j = start; j < end; ++j) {
            // Interpolate for this particle
            std::vector<double> prof = interpolator.getProfile(particles.energy[j], particles.angle[j], target_depths);

            // Store in dE_dx (Depth-Major)
            // dE_dx[depth_idx * count + particle_idx]
            for (size_t d = 0; d < prof.size(); ++d){
                dE_dx[d * count + (j - start)] = prof[d] * PhysConst::MeV_mm_to_J_m;  // Convert MeV/mm to Joules/m for the solver. The mm to m conversion is handled in the interpolation.
                cum_E_dep_total[d] += prof[d] * PhysConst::MeV_mm_to_J_m * particles.weight[j];
            }
        }

        const auto weight_view = std::span(particles.weight).subspan(start, count);
        const auto t_loss_view = std::span(particles.t_loss).subspan(start, count);

        // Solve the heat equation for this wall element
        solver.solve(dE_dx, weight_view, t_loss_view, params, coeff, times, out_T);

        double t_melt = 1e20;  // Default to infinity
        bool melted = false;
        double T_melt = 3695.0;

        for (size_t tstep = 0; tstep < times.size(); ++tstep) {
            if (!melted && out_T[tstep][0] >= T_melt) {
                t_melt = times[tstep];
                melted = true;
            }
        }

        if (melted) {
            std::vector<double> cum_E_dep_melt(target_depths.size(), 0.0);
            for (size_t j = start; j < end; ++j) {
                // Include particle only if it arrived before or at the surface melting time
                if (particles.t_loss[j] <= t_melt) {
                    for (size_t d = 0; d < target_depths.size(); ++d) {
                        cum_E_dep_melt[d] += dE_dx[d * count + (j - start)] * particles.weight[j];
                    }
                }
            }

            // --- Numerical Integration over depth (0 to L) using Trapezoidal Rule ---
            double integral_melt = 0.0;
            double integral_total = 0.0;

            for (size_t d = 0; d < target_depths.size() - 1; ++d) {
                double dx = target_depths[d + 1] - target_depths[d];

                // Average the energy density between point d and d+1
                double avg_melt = (cum_E_dep_melt[d] + cum_E_dep_melt[d + 1]) * 0.5;
                double avg_total = (cum_E_dep_total[d] + cum_E_dep_total[d + 1]) * 0.5;

                integral_melt += avg_melt * dx;
                integral_total += avg_total * dx;
            }

            // Compute fraction: Integral of energy at melting / Integral of total energy
            if (integral_total > 0) {
                results[i].energy_fraction = integral_melt / integral_total;
            } else {
                results[i].energy_fraction = 1.0;
            }
        } else {
            results[i].energy_fraction = 1.0;  // 100% if it never reaches T_melt
        }

        results[i].surf_temp.resize(times.size());

        if (args.storeAllTimes) {
            results[i].surf_temp.resize(times.size());
            for (size_t tstep = 0; tstep < times.size(); ++tstep)
                results[i].surf_temp[tstep] = out_T[tstep][0];
        } else {
            results[i].surf_temp.resize(1);
            results[i].surf_temp[0] = out_T.back()[0];
        }

        if (keep_full) {
            results[i].store_full = true;
            if (args.storeAllTimes) {
                results[i].full_profile = std::move(out_T);  // full history: [time][depth]
            } else {
                results[i].full_profile = {out_T.back()};  // only the last timestep: [1][depth]
            }
        }

        results[i].wall_id = wid;
    }

    // --- Write Results ---
    std::cout << "Writing results to " << args.outPath << "..." << std::endl;
    hid_t resFile = H5Fcreate(args.outPath.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    if (resFile < 0) throw std::runtime_error("Failed to create result file");

    const size_t n_out_times = args.storeAllTimes ? times.size() : 1;

    std::vector<double> wall_ids_out(selected_wall_ids.size());
    std::vector<double> surf_temps_out(selected_wall_ids.size() * n_out_times);
    std::vector<double> energy_fractions_out(selected_wall_ids.size());

    for (size_t i = 0; i < selected_wall_ids.size(); ++i) {
        wall_ids_out[i] = static_cast<double>(results[i].wall_id);
        energy_fractions_out[i] = results[i].energy_fraction;
        for (size_t t = 0; t < n_out_times; ++t) {
            surf_temps_out[i * n_out_times + t] = results[i].surf_temp[t];
        }
    }

    hid_t group = H5Gcreate2(resFile, "full_profiles", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

    for (const auto& res : results) {
        if (res.store_full) {
            std::string ds_name = std::to_string(res.wall_id);

            const size_t n_rows = res.full_profile.size();  // 1, or times.size()
            const size_t n_cols = target_depths.size();

            // Flatten row-major [row][col] -> contiguous buffer for H5Dwrite
            std::vector<double> flat(n_rows * n_cols);
            for (size_t r = 0; r < n_rows; ++r)
                for (size_t c = 0; c < n_cols; ++c)
                    flat[r * n_cols + c] = res.full_profile[r][c];

            hid_t space;
            if (args.storeAllTimes) {
                hsize_t dims_2d[2] = {(hsize_t)n_rows, (hsize_t)n_cols};
                space = H5Screate_simple(2, dims_2d, NULL);
            } else {
                hsize_t dims_1d[1] = {(hsize_t)n_cols};
                space = H5Screate_simple(1, dims_1d, NULL);
            }

            hid_t dataset = H5Dcreate2(group, ds_name.c_str(), H5T_NATIVE_DOUBLE, space,
                                       H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

            H5Dwrite(dataset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, flat.data());

            H5Dclose(dataset);
            H5Sclose(space);
        }
    }

    // Write datasets
    hsize_t dims_wid[1] = {(hsize_t)selected_wall_ids.size()};
    hsize_t dims_time[1] = {(hsize_t)n_out_times};
    hsize_t dims_depth[1] = {(hsize_t)target_depths.size()};
    hid_t space1 = H5Screate_simple(1, dims_wid, NULL);
    hid_t space3 = H5Screate_simple(1, dims_time, NULL);
    hid_t space4 = H5Screate_simple(1, dims_depth, NULL);

    hid_t space2;
    if (args.storeAllTimes) {
        hsize_t dims_temp[2] = {(hsize_t)selected_wall_ids.size(), (hsize_t)n_out_times};
        space2 = H5Screate_simple(2, dims_temp, NULL);
    } else {
        space2 = H5Screate_simple(1, dims_wid, NULL);  // flat (n_walls,)
    }

    hid_t ds1 = H5Dcreate2(resFile, "wall_ids", H5T_NATIVE_DOUBLE, space1, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    H5Dwrite(ds1, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, wall_ids_out.data());
    H5Dclose(ds1);

    hid_t ds_frac = H5Dcreate2(resFile, "energy_fraction", H5T_NATIVE_DOUBLE, space1, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    H5Dwrite(ds_frac, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, energy_fractions_out.data());
    H5Dclose(ds_frac);

    hid_t ds2 = H5Dcreate2(resFile, "surf_temp", H5T_NATIVE_DOUBLE, space2, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    H5Dwrite(ds2, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, surf_temps_out.data());
    H5Dclose(ds2);

    hid_t ds3 = H5Dcreate2(resFile, "times", H5T_NATIVE_DOUBLE, space3, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    if (args.storeAllTimes) {
        H5Dwrite(ds3, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, times.data());
    } else {
        double last_time = times.back();
        H5Dwrite(ds3, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, &last_time);
    }
    H5Dclose(ds3);

    hid_t ds4 = H5Dcreate2(group, "depths", H5T_NATIVE_DOUBLE, space4, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    H5Dwrite(ds4, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, target_depths.data());
    H5Dclose(ds4);

    H5Sclose(space1);
    H5Sclose(space2);
    H5Sclose(space3);
    H5Sclose(space4);
    H5Gclose(group);
    H5Fclose(resFile);

    std::cout << "Done." << std::endl;

    return 0;
}
