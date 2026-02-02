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
                << "  Part File:   " << partPath << "\n"
                << "  Interp File: " << interpPath << "\n"
                << "  Output File: " << outPath << "\n";

    // --- Load Data ---
    std::cout << "Loading data..." << std::endl;

    // Particles
    hid_t partFile = H5Fopen(partPath.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    if (partFile < 0) throw std::runtime_error("Failed to open particles file: " + partPath);

    hid_t partGroup = H5Gopen2(partFile, "groups/001", H5P_DEFAULT);
    if (partGroup < 0) {
        H5Fclose(partFile);
        throw std::runtime_error("Failed to open group 'groups/001' in " + partPath);
    }

    std::vector<double> t_loss = Utils::readH5DoubleDatasetGroup(partGroup, "t_loss");
    std::vector<double> weight = Utils::readH5DoubleDatasetGroup(partGroup, "weight");
    double energy = Utils::readH5DoubleScalar(partGroup, "energy");
    double angle = Utils::readH5DoubleScalar(partGroup, "angle");

    printf("Particle: Energy = %f MeV, Angle = %f degrees\n", energy, angle);

    H5Gclose(partGroup);
    H5Fclose(partFile);

    size_t n_particles = weight.size();
    BenchParticles particles(n_particles);

    for (size_t i = 0; i < n_particles; ++i) {
        particles.t_loss[i] = t_loss[i];
        particles.weight[i] = weight[i];
        particles.energy[i] = energy;
        particles.angle[i]  = angle;
    }
    // --- Sort and Filter ---        
    SimulationParams params(configPath);

    const double L_1 = params.L / params.L_sub;
    const double L_2 = params.L - L_1;

    const int N_x1 = static_cast<int>(L_1 / params.delta_x1);
    const int N_x2 = static_cast<int>(L_2 / params.delta_x2);

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

    std::vector<double> dE_dx(target_depths.size() * n_particles);
    
    for (size_t j = 0; j < n_particles; ++j) {
        // Interpolate for this particle
        std::vector<double> prof = interpolator.getProfile(particles.energy[j], particles.angle[j], target_depths);
        
        for (size_t d = 0; d < prof.size(); ++d)
        dE_dx[d * n_particles + j] = prof[d] * PhysConst::MeV_to_J;
    }

    std::vector<double> times;

    times.push_back(params.t_start);
    while (times.back() < params.t_end) times.push_back(times.back() + (times.back() >= params.t_interm && params.t_interm > params.t_start ? params.dt_large : params.dt_small) );

    // Solve Heat Eq
    std::vector<std::vector<double>> out_T(times.size(), std::vector<double>(target_depths.size(), 0.0));

    for (size_t xi = 0; xi < target_depths.size(); xi++) out_T[0][xi] = params.T_ini;

    const double coeff = 1.0 / (1e-6 * params.t_dep);
    solver.solve(dE_dx, particles.weight, particles.t_loss, params, coeff, times, out_T);

    // --- Write Results ---
    std::cout << "Writing results to " << outPath << "..." << std::endl;
    hid_t resFile = H5Fcreate(outPath.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    if (resFile < 0) throw std::runtime_error("Failed to create result file: " + outPath);

    // Write datasets out_T (out_Nx, out_Nt) and out_times (out_Nt) and depths_m (out_Nx)
    hid_t space_T = H5Screate_simple(2, (hsize_t[]){static_cast<hsize_t>(target_depths.size()), static_cast<hsize_t>(times.size())}, NULL);
    hid_t dset_T = H5Dcreate2(resFile, "temperature", H5T_IEEE_F64LE, space_T, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    std::vector<double> out_T_flat;
    for (size_t xi = 0; xi < target_depths.size(); ++xi) {
        for (size_t ti = 0; ti < times.size(); ++ti) {
            out_T_flat.push_back(out_T[ti][xi]);
        }
    }
    // Flatten out_T for HDF5 writing
    H5Dwrite(dset_T, H5T_IEEE_F64LE, H5S_ALL, H5S_ALL, H5P_DEFAULT, out_T_flat.data());
    H5Dclose(dset_T);
    H5Sclose(space_T);

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
