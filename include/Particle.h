#ifndef PARTICLE_H
#define PARTICLE_H

#include <vector>
#include <string>
#include <hdf5.h>
#include <cmath>

#include "Utils.h"
#include "PhysConst.h"

/**
 * @brief Reorders a vector based on a permutation index.
 *
 * @tparam T Type of the vector elements.
 * @param data The vector to be reordered.
 * @param p_indices The permutation indices.
 */
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

/**
 * @brief Base class for particle data used in benchmarks.
 *
 * Stores basic particle properties like loss time, weight, energy, and angle.
 */
class BenchParticles {
    public:
    std::vector<double> t_loss; ///< Time of particle loss.
    std::vector<double> weight; ///< Particle weight.
    std::vector<double> energy; ///< Particle energy.
    std::vector<double> angle;  ///< Particle angle of incidence.
    int n_particles;            ///< Total number of particles.

    /**
     * @brief Constructs BenchParticles with a given size.
     * @param n Number of particles.
     */
    BenchParticles (size_t n) : t_loss(n), weight(n), energy(n), angle(n), n_particles(n) {}

    /**
     * @brief Default constructor.
     */
    BenchParticles () {}
};

/**
 * @brief Class for particle data loaded from HDF5 files.
 *
 * Extends BenchParticles to include wall IDs and velocity components.
 * Loads and processes particle data from a specified HDF5 file.
 */
class Particles : public BenchParticles {
    public:
    std::vector<int> wall_id;    ///< ID of the wall element where the particle hit.
    std::vector<double> vx;      ///< Velocity X component.
    std::vector<double> vy;      ///< Velocity Y component.
    std::vector<double> vz;      ///< Velocity Z component.
    
    /**
     * @brief Constructs Particles object and loads data from HDF5 file.
     *
     * Loads particle data, calculates energy, sorts by wall ID and loss time,
     * and filters out invalid particles.
     *
     * @param partPath Path to the HDF5 particles file.
     * @throws std::runtime_error If file opening or group access fails.
     */
    Particles (std::string partPath){
        hid_t partFile = H5Fopen(partPath.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
        if (partFile < 0) throw std::runtime_error("Failed to open particles file: " + partPath);

        hid_t partGroup = H5Gopen2(partFile, "groups/001", H5P_DEFAULT);
        if (partGroup < 0) {
            H5Fclose(partFile);
            throw std::runtime_error("Failed to open group 'groups/001' in " + partPath);
        }

        std::vector<int> i_elm = Utils::readH5IntDatasetGroup(partGroup, "i_elm");
        n_particles = i_elm.size();

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

        H5Gclose(partGroup);
        H5Fclose(partFile);

        #pragma omp parallel for
        for (size_t i = 0; i < n_particles; ++i) {
            wall_id[i] = -i_elm[i];  // Python code flips sign: wetted_sorted = -wetted_elements[sort_idx]
            vx[i] = v_flat[i * 3 + 0];
            vy[i] = v_flat[i * 3 + 1];
            vz[i] = v_flat[i * 3 + 2];
            const double p_norm = std::sqrt(vx[i] * vx[i] + vy[i] * vy[i] + vz[i] * vz[i]);
            const double pc = p_norm * PhysConst::c;
            const double m0c2 = PhysConst::m_e_u * PhysConst::c * PhysConst::c;
            const double E = (std::sqrt(pc * pc + m0c2 * m0c2) - m0c2) * PhysConst::AMU_m2_s2_to_MeV;
            energy[i] = E;
        }

        // We physically reorder vectors so data for Wall X is contiguous in memory.
        std::cout << "Sorting and reordering particles..." << std::endl;
        
        size_t n_particles = wall_id.size();
        std::vector<size_t> p_indices(n_particles);
        std::iota(p_indices.begin(), p_indices.end(), 0);

        std::sort(p_indices.begin(), p_indices.end(), [&](size_t i, size_t j) {
            return wall_id[i] < wall_id[j] || (wall_id[i] == wall_id[j] && t_loss[i] < t_loss[j]);
        });

        // 2. Apply Permutation to all data vectors
        apply_permutation(wall_id, p_indices);
        apply_permutation(t_loss, p_indices);
        apply_permutation(weight, p_indices);
        apply_permutation(vx, p_indices);
        apply_permutation(vy, p_indices);
        apply_permutation(vz, p_indices);
        apply_permutation(energy, p_indices);
        // Note: 'angle' is not calculated yet, so we don't need to sort it.

        // --- Filter Zero IDs ---
        auto it_first_nonzero = std::find_if(wall_id.begin(), wall_id.end(), [](int id) {
            return id > 0;
        });

        if (it_first_nonzero == wall_id.end()) std::cerr << "No non-zero particles found." << std::endl;

        // Determine the valid range in the sorted arrays
        size_t start_offset = std::distance(wall_id.begin(), it_first_nonzero);
        // We also drop the last element (end-1) matching previous logic
        size_t end_offset = n_particles; 

        if (start_offset >= end_offset) std::cerr << "Not enough particles after filtering." << std::endl;

        // Resize all vectors to keep only valid particles

        wall_id = std::vector<int>(wall_id.begin() + start_offset, wall_id.begin() + end_offset);
        t_loss = std::vector<double>(t_loss.begin() + start_offset, t_loss.begin() + end_offset);
        weight = std::vector<double>(weight.begin() + start_offset, weight.begin() + end_offset);
        vx = std::vector<double>(vx.begin() + start_offset, vx.begin() + end_offset);
        vy = std::vector<double>(vy.begin() + start_offset, vy.begin() + end_offset);
        vz = std::vector<double>(vz.begin() + start_offset, vz.begin() + end_offset);
        energy = std::vector<double>(energy.begin() + start_offset, energy.begin() + end_offset);
    }

    /**
     * @brief Retrieves a list of unique wall IDs present in the particle data.
     *
     * @return std::vector<int> A vector containing unique wall IDs.
     */
    std::vector<int> get_unique_wall_ids () {
        // --- Identify Unique IDs ---
        std::vector<int> unique_wall_ids;
        unique_wall_ids.push_back(wall_id[0]);
        for (size_t i = 1; i < wall_id.size(); ++i) {
            if (wall_id[i] != wall_id[i - 1]) {
                unique_wall_ids.push_back(wall_id[i]);
            }
        }

        return unique_wall_ids;
    }
};



#endif // PARTICLE_H