#ifndef PARTICLE_H
#define PARTICLE_H

#include <vector>
#include <string>
#include <hdf5.h>
#include <cmath>

#include "Utils.h"
#include "PhysConst.h"

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

struct BenchParticles {
    std::vector<double> t_loss;
    std::vector<double> weight;
    std::vector<double> energy, angle;
    BenchParticles (size_t n) : t_loss(n), weight(n), energy(n), angle(n) {}
    BenchParticles () {}
};

struct Particles : public BenchParticles {
    std::vector<int> wall_id;
    std::vector<double> vx, vy, vz;
    
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



#endif // PARTICLE_H