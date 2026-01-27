#include <hdf5.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <numeric>
#include <vector>

#include "Interpolator.h"
#include "Solver.h"
#include "Utils.h"

struct Particle {
    int id;
    double t_loss;
    double weight;
};

int main(int argc, char* argv[]) {
    try {
        if (argc < 2) {
            std::cerr << "Usage: " << argv[0] << " <particles_file.h5>" << std::endl;
            return 1;
        }
        // --- Paths ---
        std::string partPath = argv[1];
        std::string interpPath = "../../data/interpolation_data.h5";
        // --- Load Data ---
        std::cout << "Loading data..." << std::endl;

        // Particles
        hid_t partFile = H5Fopen(partPath.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
        if (partFile < 0) {
            throw std::runtime_error("Failed to open particles file");
        }
        hid_t partGroup = H5Gopen2(partFile, "groups/001", H5P_DEFAULT);
        if (partGroup < 0) {
            H5Fclose(partFile);
            throw std::runtime_error("Failed to open group");
        }

        std::vector<int> i_elm = Utils::readH5IntDatasetGroup(partGroup, "i_elm");
        std::vector<double> t_loss = Utils::readH5DoubleDatasetGroup(partGroup, "t_loss");
        std::vector<double> weight = Utils::readH5DoubleDatasetGroup(partGroup, "weight");
        std::vector<double> energy_vec = Utils::readH5DoubleDatasetGroup(partGroup, "energy");
        double energy = energy_vec[0];
        std::vector<double> angle_vec = Utils::readH5DoubleDatasetGroup(partGroup, "angle");
        double angle = angle_vec[0];

        H5Gclose(partGroup);
        H5Fclose(partFile);

        size_t n_particles = i_elm.size();
        std::vector<Particle> particles(n_particles);

        for (size_t i = 0; i < n_particles; ++i) {
            particles[i].id = -i_elm[i];
            particles[i].t_loss = t_loss[i];
            particles[i].weight = weight[i];

            if (i == n_particles - 1) {
                std::cout << particles[i].t_loss << std::endl;
            }
        }

        // --- Simulation Parameters ---
        double c = 3e8;
        double m_0 = 0.000548;
        double conv_factor = 1.66e-27 * 1e-6 * 6.24e18;
        double conv_factor2 = 1.60217663e-19 * 1e6 * 1e3;  // MeV/mm to J/m

        double L = 24e-3;
        double L_1 = L / 30.0;
        double L_2 = L - L_1;

        double delta_x1 = 2e-6;
        double delta_x2 = 2e-6;  // 1.5e-4

        int N_x1 = static_cast<int>(L_1 / delta_x1);
        int N_x2 = static_cast<int>(L_2 / delta_x2);

        SimulationParams params;
        params.T_ini = 300;
        params.dt_small = 1e-7;
        params.dt_large = 1e-3;
        params.t_dep = 10 * params.dt_small;
        params.t_start = 0.0;
        params.t_end = 2.6e-4;  // 1e-3
        params.t_interm = 0.0;

        // --- Prepare Interpolator and Material ---
        Interpolator interpolator(interpPath);
        Material material;
        Solver solver(material);

        int wid = 1;

        // Extract data for this wall element
        std::vector<double> p_coll_times(n_particles);
        std::vector<double> p_weights(n_particles);
        // Momenta and Normals

        double area = 1e-6;
        double coeff_num = 1.0 / (area * params.t_dep);
        params.coeff = coeff_num;

        std::vector<double> p_energies(n_particles);
        std::vector<double> p_angles(n_particles);

        for (size_t k = 0; k < n_particles; ++k) {
            const Particle& p = particles[k];
            p_coll_times[k] = p.t_loss;
            p_weights[k] = p.weight;

            // Angle
            p_angles[k] = angle;

            // Energy
            p_energies[k] = energy;
        }

        // Interpolate Profile

        std::vector<double> target_depths_mm(N_x1 + N_x2);
        double dx1_mm = delta_x1 * 1000.0;
        double dx2_mm = delta_x2 * 1000.0;

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

        std::vector<double> dE_dx(target_depths_mm.size() * n_particles);

        for (size_t j = 0; j < n_particles; ++j) {
            // Interpolate for this particle
            std::vector<double> prof = interpolator.getProfile(p_energies[j], p_angles[j], target_depths_mm);

            // Store in dE_dx (Depth-Major)
            for (size_t d = 0; d < prof.size(); ++d) {
                dE_dx[d * n_particles + j] = prof[d] * conv_factor2;
            }
        }

        // 1. Calculate weighted dE/dx sum across all particles at each depth
        std::vector<double> total_dE_dx(target_depths_mm.size(), 0.0);

        for (size_t d = 0; d < target_depths_mm.size(); ++d) {
            for (size_t j = 0; j < n_particles; ++j) {
                // Multiply by the weight of the macro-particle here!
                total_dE_dx[d] += dE_dx[d * n_particles + j] * p_weights[j];
            }
        }

        // 2. Integrate to find cumulative energy
        std::vector<double> cumulative_energy(target_depths_mm.size(), 0.0);
        double total_sum_joules = 0.0;

        for (size_t d = 1; d < target_depths_mm.size(); ++d) {
            // Distance in meters (target_depths_mm is in mm)
            double delta_x_m = (target_depths_mm[d] - target_depths_mm[d - 1]) * 1e-3;

            // Trapezoidal rule for dE/dx (J/m) over dx (m)
            double avg_dE_dx = (total_dE_dx[d] + total_dE_dx[d - 1]) / 2.0;

            total_sum_joules += avg_dE_dx * delta_x_m;
            cumulative_energy[d] = total_sum_joules;
        }

        // 3. Output the result
        std::cout << "Total Energy Deposited: " << total_sum_joules << " Jul" << std::endl;

        // Solve Heat Eq
        std::vector<double> out_T, out_times;
        int out_Nx, out_Nt;

        std::vector<double> depths_m(target_depths_mm.size());
        for (size_t d = 0; d < depths_m.size(); ++d) depths_m[d] = target_depths_mm[d] * 1e-3;
        solver.solve(dE_dx, p_weights, p_coll_times, depths_m, params, out_T, out_times, out_Nx, out_Nt);

        std::string s = std::to_string(energy);
        s.erase(s.find_last_not_of('0') + 1, std::string::npos);
        if (s.back() == '.') s.pop_back();

        std::string outPath = "simulation_results_" + s + "MeV_" + std::to_string(n_particles) + ".h5";
        hid_t outFile = H5Fcreate(outPath.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
        if (outFile < 0) {
            throw std::runtime_error("Failed to create output file");
        }

        // 1. Save 1D Arrays (Times and Depths)
        hsize_t time_dims[1] = {(hsize_t)out_Nt};
        hid_t time_space = H5Screate_simple(1, time_dims, NULL);
        hid_t time_ds = H5Dcreate2(outFile, "times", H5T_NATIVE_DOUBLE, time_space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        H5Dwrite(time_ds, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, out_times.data());
        H5Dclose(time_ds);
        H5Sclose(time_space);

        hsize_t depth_dims[1] = {(hsize_t)out_Nx};
        hid_t depth_space = H5Screate_simple(1, depth_dims, NULL);
        hid_t depth_ds = H5Dcreate2(outFile, "depths", H5T_NATIVE_DOUBLE, depth_space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        H5Dwrite(depth_ds, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, depths_m.data());
        H5Dclose(depth_ds);
        H5Sclose(depth_space);

        // 2. Save 2D Temperature Grid
        hsize_t temp_dims[2] = {(hsize_t)out_Nx, (hsize_t)out_Nt};
        hid_t temp_space = H5Screate_simple(2, temp_dims, NULL);

        hid_t temp_ds = H5Dcreate2(outFile, "temperature", H5T_NATIVE_DOUBLE, temp_space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        H5Dwrite(temp_ds, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, out_T.data());
        H5Dclose(temp_ds);
        H5Sclose(temp_space);

        H5Fclose(outFile);

        std::cout << "Results successfully saved to " << outPath << " (Layout: Depth x Time)" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
