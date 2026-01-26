#include "Solver.h"
#include "Interpolator.h"
#include "Utils.h"
#include <iostream>
#include <vector>
#include <numeric>
#include <algorithm>
#include <cmath>
#include <H5Cpp.h>
#include <omp.h>

struct Particle {
    int id;
    double t_loss;
    double weight;
    double vx, vy, vz;
};

// Sort ascending by ID (to match Python argsort on positive values)
bool compareParticles(const Particle& a, const Particle& b) {
    return a.id < b.id;
}

struct Result {
    int wall_id;
    double surf_temp;
    std::vector<double> snaps;
};

int main() {
    try {
        // --- Paths ---
        std::string wallPath = "../../data/hannes_data/newiterwall_offset10cm.h5";
        //std::string wallPath = "asdex_wall.h5";
        std::string partPath = "../../data/hannes_data/eta_10x_fo_v2/part_out_eta_10x_fo.h5";
        std::string interpPath = "../../data/interpolation_data.h5";
        std::string outPath = "iter_results.h5";

        // --- Load Data ---
        std::cout << "Loading data..." << std::endl;
        
        // Wall
        H5::H5File wallFile(wallPath, H5F_ACC_RDONLY);
        std::vector<double> wall = Utils::readH5DoubleDataset(wallFile, "nodes");
        
        // Particles
        H5::H5File partFile(partPath, H5F_ACC_RDONLY);
        H5::Group partGroup = partFile.openGroup("groups/001");
        
        std::vector<int> i_elm = Utils::readH5IntDatasetGroup(partGroup, "i_elm");
        std::vector<double> t_loss = Utils::readH5DoubleDatasetGroup(partGroup, "t_loss");
        std::vector<double> weight = Utils::readH5DoubleDatasetGroup(partGroup, "weight");
        std::vector<double> v_flat = Utils::readH5DoubleDatasetGroup(partGroup, "v"); // Nx3 flattened
        
        size_t n_particles = i_elm.size();
        std::vector<Particle> particles(n_particles);
        
        for(size_t i=0; i<n_particles; ++i) {
            particles[i].id = -i_elm[i]; // Python code flips sign: wetted_sorted = -wetted_elements[sort_idx]
            particles[i].t_loss = t_loss[i];
            particles[i].weight = weight[i];
            particles[i].vx = v_flat[i*3 + 0];
            particles[i].vy = v_flat[i*3 + 1];
            particles[i].vz = v_flat[i*3 + 2];
        }
        
        // --- Sort and Filter ---
        std::cout << "Sorting..." << std::endl;
        std::sort(particles.begin(), particles.end(), compareParticles);
        
        
        // Find first non-zero ID
        auto it_first_nonzero = std::find_if(particles.begin(), particles.end(), [](const Particle& p){
            return p.id != 0;
        });
        
        if (it_first_nonzero == particles.end()) {
            std::cerr << "No non-zero particles found." << std::endl;
            return 1;
        }
        
        // Slice from idx to end-1
        // In C++: copy from it_first_nonzero to (end - 1)
        if (std::distance(it_first_nonzero, particles.end()) <= 1) {
             std::cerr << "Not enough particles after filtering." << std::endl;
             return 1;
        }
        
        std::vector<Particle> filtered_particles(it_first_nonzero, particles.end() - 1);
        
        // Unique IDs
        std::vector<int> unique_ids;
        // Since it's sorted descending, we can iterate
        if (!filtered_particles.empty()) {
            unique_ids.push_back(filtered_particles[0].id);
            for(size_t i=1; i<filtered_particles.size(); ++i) {
                if (filtered_particles[i].id != filtered_particles[i-1].id) {
                    unique_ids.push_back(filtered_particles[i].id);
                }
            }
        }
        
        // --- Simulation Parameters ---
        double c = 3e8;
        double m_0 = 0.000548;
        double conv_factor = 1.66e-27 * 1e-6 * 6.24e18;
        double conv_factor2 = 1.6022e-19 * 1e6 * 1e3;
        
        double L = 24e-3;
        double L_1 = L/10.0;
        double L_2 = L - L_1;
        
        double delta_x1 = 1.5e-6;
        double delta_x2 = 1.5e-4;
        
        int N_x1 = (int)(L_1/delta_x1);
        int N_x2 = (int)(L_2/delta_x2);
        
        SimulationParams params;
        params.T_ini = 300;
        params.dt_small = 1e-7;
        params.dt_large = 1e-3;
        params.t_dep = 10 * params.dt_small;
        params.t_start = 0.05125;
        params.t_end = 0.0514;
        params.t_interm = 0;
        
        // --- Prepare Interpolator and Material ---
        Interpolator interpolator(interpPath);
        Material material;
        Solver solver(material);
        
        
        int N_select = unique_ids.size();
        
        std::vector<int> selected_ids(unique_ids.begin(), unique_ids.begin() + N_select);
        
        std::cout << "Processing " << N_select << " elements using OpenMP..." << std::endl;
        std::cout << "Max threads: " << omp_get_max_threads() << std::endl;
        
        std::vector<Result> results(N_select);
        
        // Identify ranges for each unique ID in filtered_particles
        // To avoid searching every time, we can pre-calculate ranges
        std::vector<std::pair<size_t, size_t>> ranges(N_select);
        
        // filtered_particles is sorted descending. unique_ids is sorted descending.
        // We can scan through.
        size_t current_idx = 0;
        for(int i=0; i<N_select; ++i) {
            int uid = selected_ids[i];
            size_t start = current_idx;
            while(current_idx < filtered_particles.size() && filtered_particles[current_idx].id == uid) {
                current_idx++;
            }
            ranges[i] = {start, current_idx};
        }

        // --- Parallel Loop ---
        #pragma omp parallel for schedule(dynamic)
        for(int i=0; i<N_select; ++i) {
            int wid = selected_ids[i];
            size_t start = ranges[i].first;
            size_t end = ranges[i].second;
            size_t count = end - start;
            
            // Extract data for this wall element
            std::vector<double> p_coll_times(count);
            std::vector<double> p_weights(count);
            // Momenta and Normals
            
            // Compute Wall Geometry
            
            
            int offset = wid * 9;
            
            if (offset + 8 >= wall.size()) {
                std::cerr << "Error: Wall ID " << wid << " out of bounds! Wall size: " << wall.size() << std::endl;
                continue;
            }
            
            double p0[3] = { wall[offset+0], wall[offset+1], wall[offset+2] };
            double p1[3] = { wall[offset+3], wall[offset+4], wall[offset+5] };
            double p2[3] = { wall[offset+6], wall[offset+7], wall[offset+8] };
            
            double a[3] = { p1[0]-p0[0], p1[1]-p0[1], p1[2]-p0[2] };
            double b[3] = { p2[0]-p0[0], p2[1]-p0[1], p2[2]-p0[2] };
            
            double normal[3];
            // Cross product a x b
            normal[0] = a[1]*b[2] - a[2]*b[1];
            normal[1] = a[2]*b[0] - a[0]*b[2];
            normal[2] = a[0]*b[1] - a[1]*b[0];
            
            double norm_len = std::sqrt(normal[0]*normal[0] + normal[1]*normal[1] + normal[2]*normal[2]);
            normal[0] /= norm_len;
            normal[1] /= norm_len;
            normal[2] /= norm_len;
            
            
            double area = 0.5 * norm_len;
            double coeff_num = 1.0 / (area * params.t_dep);
            SimulationParams local_params = params;
            local_params.coeff = coeff_num;
            
            std::vector<double> p_energies(count);
            std::vector<double> p_angles(count);
            
            for(size_t k=0; k<count; ++k) {
                const Particle& p = filtered_particles[start + k];
                p_coll_times[k] = p.t_loss;
                p_weights[k] = p.weight;
                
                double p_norm = std::sqrt(p.vx*p.vx + p.vy*p.vy + p.vz*p.vz);
                double dot = p.vx*normal[0] + p.vy*normal[1] + p.vz*normal[2];
                
                // Angle
                // degrees(arccos(dot/norm)) - 90
                double ang_rad = std::acos(dot / p_norm);
                double ang_deg = std::abs(ang_rad * 180.0 / M_PI - 90.0);
                p_angles[k] = ang_deg;
                
                // Energy
                // (sqrt((p*c)^2 + (m0*c^2)^2) - m0*c^2) * conv
                double pc = p_norm * c;
                double m0c2 = m_0 * c * c;
                double E = (std::sqrt(pc*pc + m0c2*m0c2) - m0c2) * conv_factor;
                p_energies[k] = E;
            }
            
            
            // Interpolate Profile
            
            // Target depths
            std::vector<double> target_depths(N_x1 + N_x2);
            target_depths[0] = 0.0;
            for(int d=0; d<(N_x1+N_x2)-1; ++d) {
                double spacing;
                if(d < N_x1 - 1) spacing = delta_x1;
                else if (d == N_x1 - 1) spacing = 0.5 * (delta_x1 + delta_x2);
                else spacing = delta_x2;
                target_depths[d+1] = target_depths[d] + spacing;
            }
            
            std::vector<double> target_depths_mm(N_x1 + N_x2);
            double dx1_mm = delta_x1 * 1000.0;
            double dx2_mm = delta_x2 * 1000.0;
            
            target_depths_mm[0] = 0.0;
            for(int d=0; d<(N_x1+N_x2)-1; ++d) {
                double spacing;
                if(d < N_x1 - 1) spacing = dx1_mm;
                else if (d == N_x1 - 1) spacing = 0.5 * (dx1_mm + dx2_mm);
                else spacing = dx2_mm;
                target_depths_mm[d+1] = target_depths_mm[d] + spacing;
            }
            
            std::vector<double> dE_dx(target_depths_mm.size() * count);
            
            for(size_t j=0; j<count; ++j) {
                // Interpolate for this particle
                std::vector<double> prof = interpolator.getProfile(p_energies[j], p_angles[j], target_depths_mm);
                
                // Store in dE_dx (Depth-Major)
                // dE_dx[depth_idx * count + particle_idx]
                for(size_t d=0; d<prof.size(); ++d) {
                    dE_dx[d * count + j] = prof[d] * conv_factor2; // Apply conv_factor2 here to match python passing `conv_factor2 * dE_dx`
                }
            }
            
            // Solve Heat Eq
            std::vector<double> out_T, out_times;
            int out_Nx, out_Nt;
            
            std::vector<double> depths_m(target_depths_mm.size());
            for(size_t d=0; d<depths_m.size(); ++d) depths_m[d] = target_depths_mm[d] * 1e-3;
            
            solver.solve(dE_dx, p_weights, p_coll_times, depths_m, local_params, out_T, out_times, out_Nx, out_Nt);
            
            
            double max_val = -1e20;
            int max_idx = -1;
            
            for(size_t idx=0; idx<out_T.size(); ++idx) {
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
            for(int s=0; s<50; ++s) {
                // linspace logic
                // 0 to Nt-1
                int t_idx = (int)(s * (out_Nt - 1) / 49.0); // simple linear map
                if (t_idx >= out_Nt) t_idx = out_Nt - 1;
                snaps[s] = out_T[t_idx]; // x=0, time=t_idx -> index=t_idx
            }
            
            results[i].wall_id = wid;
            results[i].surf_temp = surf_temp;
            results[i].snaps = snaps;
        }
        
        // --- Write Results ---
        std::cout << "Writing results to " << outPath << "..." << std::endl;
        H5::H5File resFile(outPath, H5F_ACC_TRUNC);
        
        std::vector<double> wall_ids_out(N_select);
        std::vector<double> surf_temps_out(N_select);
        // temp_snaps is (N_select, 50)
        std::vector<double> temp_snaps_out(N_select * 50);
        
        for(int i=0; i<N_select; ++i) {
            wall_ids_out[i] = (double)results[i].wall_id;
            surf_temps_out[i] = results[i].surf_temp;
            for(int s=0; s<50; ++s) {
                temp_snaps_out[i*50 + s] = results[i].snaps[s];
            }
        }
        
        // Write datasets
        hsize_t dims1[1] = { (hsize_t)N_select };
        H5::DataSpace space1(1, dims1);
        
        H5::DataSet ds1 = resFile.createDataSet("wall_ids", H5::PredType::NATIVE_DOUBLE, space1);
        ds1.write(wall_ids_out.data(), H5::PredType::NATIVE_DOUBLE);
        
        H5::DataSet ds2 = resFile.createDataSet("surf_temp", H5::PredType::NATIVE_DOUBLE, space1);
        ds2.write(surf_temps_out.data(), H5::PredType::NATIVE_DOUBLE);
        
        hsize_t dims2[2] = { (hsize_t)N_select, 50 };
        H5::DataSpace space2(2, dims2);
        H5::DataSet ds3 = resFile.createDataSet("temp_snaps", H5::PredType::NATIVE_DOUBLE, space2);
        ds3.write(temp_snaps_out.data(), H5::PredType::NATIVE_DOUBLE);
        
        std::cout << "Done." << std::endl;
        
    } catch (H5::Exception& e) {
        std::cerr << "HDF5 Error: " << e.getDetailMsg() << std::endl;
        return 1;
    } catch (std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
