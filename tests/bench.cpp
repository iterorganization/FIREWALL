#include <hdf5.h>

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
#include "Particle.h"

int main(int argc, char* argv[]) {
    BenchArgParser::Args args = BenchArgParser::parse(argc, argv);
    if (args.help) {
        BenchArgParser::printUsage(argv[0]);
        return 0;
    }

    std::string configPath = args.configPath;
    std::string partPath = args.partPath;
    std::string interpPath = args.interpPath;
    std::string outPath = args.outPath;

    std::cout << "Configuration:\n"
                << "  Config File: " << configPath << "\n"
                << "  Interp File: " << interpPath << "\n"
                << "  Output File: " << outPath << "\n";

    // --- Load Data ---
    std::cout << "Loading data..." << std::endl;

    // Particles

    double angle_1 = 2.0;
    double energy_1 = 1.0;
    std::vector<double> weight_1 = {10 * 1.0877 * PhysConst::J_to_MeV / energy_1};
    std::vector<double> t_loss_1 = { 0.0 };

    double angle_2 = 10.0;
    double energy_2 = 25.0;
    std::vector<double> weight_2 = {40 * 1.1385 * PhysConst::J_to_MeV / energy_2};
    std::vector<double> t_loss_2 = { 0.0 };

    printf("Benchmark 1, Particle 1: Energy = %f MeV, Angle = %f degrees\n", energy_1, angle_1);
    printf("Benchmark 1, Particle 2: Energy = %f MeV, Angle = %f degrees\n", energy_2, angle_2);

    // --- Sort and Filter ---        
    SimulationParams params(configPath);

    double L_1 = params.L / params.L_sub;
    double L_2 = params.L - L_1;

    int N_x1 = static_cast<int>(L_1 / params.delta_x1);
    int N_x2 = static_cast<int>(L_2 / params.delta_x2);

    // Extract data for this wall element
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
    
    const Interpolator interpolator(interpPath);
    const Solver solver({}, target_depths);

    std::vector<double> dE_dx_1(target_depths.size() * 1);
    std::vector<double> dE_dx_2(target_depths.size() * 1);

    // Interpolate for this particle
    std::vector<double> prof_1 = interpolator.getProfile(energy_1, angle_1, target_depths);
    std::vector<double> prof_2 = interpolator.getProfile(energy_2, angle_2, target_depths);

    for (size_t d = 0; d < prof_1.size(); ++d)
    dE_dx_1[d * 1] = prof_1[d] * PhysConst::MeV_mm_to_J_m;
    for (size_t d = 0; d < prof_2.size(); ++d)        
    dE_dx_2[d * 1] = prof_2[d] * PhysConst::MeV_mm_to_J_m;

    std::vector<double> times;

    times.push_back(params.t_start);
    while (times.back() < params.t_end) times.push_back(times.back() + (times.back() >= params.t_interm && params.t_interm > params.t_start ? params.dt_large : params.dt_small) );

    // Solve Heat Eq
    std::vector<std::vector<double>> out_T_1(times.size(), std::vector<double>(target_depths.size(), 0.0));
    std::vector<std::vector<double>> out_T_2(times.size(), std::vector<double>(target_depths.size(), 0.0));

    for (size_t xi = 0; xi < target_depths.size(); xi++) out_T_1[0][xi] = params.T_ini;
    for (size_t xi = 0; xi < target_depths.size(); xi++) out_T_2[0][xi] = params.T_ini;

    const double coeff = 1.0 / (1e-6 * params.t_dep);
    solver.solve(dE_dx_1, weight_1, t_loss_1, params, coeff, times, out_T_1);
    solver.solve(dE_dx_2, weight_2, t_loss_2, params, coeff, times, out_T_2);

    // --- Write Results ---
    std::cout << "Writing results to " << outPath << "..." << std::endl;
    hid_t resFile = H5Fcreate(outPath.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    if (resFile < 0) throw std::runtime_error("Failed to create result file: " + outPath);

    // Write datasets out_T_1 and out_T_2 (out_Nx, out_Nt) and out_times (out_Nt) and depths_m (out_Nx)
    const hsize_t dims_T_1[2] = {static_cast<hsize_t>(target_depths.size()), static_cast<hsize_t>(times.size())};
    hid_t space_T_1 = H5Screate_simple(2, dims_T_1, NULL);
    hid_t dset_T_1 = H5Dcreate2(resFile, "temperature_1", H5T_IEEE_F64LE, space_T_1, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    std::vector<double> out_T_1_flat;
    std::vector<double> out_T_2_flat;
    for (size_t xi = 0; xi < target_depths.size(); ++xi) {
        for (size_t ti = 0; ti < times.size(); ++ti) {
            out_T_1_flat.push_back(out_T_1[ti][xi]);
            out_T_2_flat.push_back(out_T_2[ti][xi]);
        }
    }
    // Flatten out_T_1 for HDF5 writing
    H5Dwrite(dset_T_1, H5T_IEEE_F64LE, H5S_ALL, H5S_ALL, H5P_DEFAULT, out_T_1_flat.data());
    H5Dclose(dset_T_1);
    H5Sclose(space_T_1);

    const hsize_t dims_T_2[2] = {static_cast<hsize_t>(target_depths.size()), static_cast<hsize_t>(times.size())};
    hid_t space_T_2 = H5Screate_simple(2, dims_T_2, NULL);
    hid_t dset_T_2 = H5Dcreate2(resFile, "temperature_2", H5T_IEEE_F64LE, space_T_2, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    // Flatten out_T_2 for HDF5 writing
    H5Dwrite(dset_T_2, H5T_IEEE_F64LE, H5S_ALL, H5S_ALL, H5P_DEFAULT, out_T_2_flat.data());
    H5Dclose(dset_T_2);
    H5Sclose(space_T_2);

    hsize_t dims_times[1] = {static_cast<hsize_t>(times.size())};
    hid_t space_times = H5Screate_simple(1, dims_times, NULL);
    hid_t dset_times = H5Dcreate2(resFile, "times", H5T_IEEE_F64LE, space_times, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    H5Dwrite(dset_times, H5T_IEEE_F64LE, H5S_ALL, H5S_ALL, H5P_DEFAULT, times.data());
    H5Dclose(dset_times);
    H5Sclose(space_times);

    hsize_t dims_depths[1] = {static_cast<hsize_t>(target_depths.size())};
    hid_t space_depths = H5Screate_simple(1, dims_depths, NULL);
    hid_t dset_depths = H5Dcreate2(resFile, "depths", H5T_IEEE_F64LE, space_depths, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    H5Dwrite(dset_depths, H5T_IEEE_F64LE, H5S_ALL, H5S_ALL, H5P_DEFAULT, target_depths.data());
    H5Dclose(dset_depths);
    H5Sclose(space_depths);

    H5Fclose(resFile);

    std::cout << "Done." << std::endl;

    return 0;
}
