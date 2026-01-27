/**
 * Module for interpolating the energy deposition profile.
 */

#include "Interpolator.h"
#include "Utils.h"
#include <iostream>
#include <algorithm>
#include <cmath>

/**
 * Constructor.
 */
Interpolator::Interpolator(const std::string& h5Path) {
    hid_t file_id = H5Fopen(h5Path.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    if (file_id < 0) {
        throw std::runtime_error("Failed to open interpolation file: " + h5Path);
    }
    
    energies_train = Utils::readH5DoubleDataset(file_id, "energies");
    angles_train = Utils::readH5DoubleDataset(file_id, "angles");
    depths_std = Utils::readH5DoubleDataset(file_id, "depths");
    
    profiles_data = Utils::readH5DoubleDataset(file_id, "profiles");
    
    H5Fclose(file_id);
    
    n_energies = energies_train.size();
    n_angles = angles_train.size();
    n_depths = depths_std.size();}

/**
 * Interpolate energy deposition profile from incidence angle and energy.
 * The profile has the same shape as depths_std 
 *
 * energy:   Energy of physical particle in a macroparticle. 
 * angle:    Incidence angle of macroparticle.
 */
std::vector<double> Interpolator::interpolateProfile2D(double energy, double angle) {
    
    // Clamp
    if (energy < energies_train.front()) energy = energies_train.front();
    if (energy > energies_train.back()) energy = energies_train.back();
    
    if (angle < angles_train.front()) angle = angles_train.front();
    if (angle > angles_train.back()) angle = angles_train.back();
    
    auto it_e = std::lower_bound(energies_train.begin(), energies_train.end(), energy);
    int idx_e1 = std::distance(energies_train.begin(), it_e);
    if (idx_e1 > 0 && (it_e == energies_train.end() || *it_e > energy)) idx_e1--;
    int idx_e2 = idx_e1 + 1;
    if (idx_e2 >= n_energies) idx_e2 = idx_e1;

    auto it_a = std::lower_bound(angles_train.begin(), angles_train.end(), angle);
    int idx_a1 = std::distance(angles_train.begin(), it_a);
    if (idx_a1 > 0 && (it_a == angles_train.end() || *it_a > angle)) idx_a1--;
    int idx_a2 = idx_a1 + 1;
    if (idx_a2 >= n_angles) idx_a2 = idx_a1;
    
    // Weights
    double w_e = 0.0;
    if (idx_e2 != idx_e1) {
        w_e = (energy - energies_train[idx_e1]) / (energies_train[idx_e2] - energies_train[idx_e1]);
    }
    
    double w_a = 0.0;
    if (idx_a2 != idx_a1) {
        w_a = (angle - angles_train[idx_a1]) / (angles_train[idx_a2] - angles_train[idx_a1]);
    }
    
    // Retrieve 4 profiles (or fewer if boundaries)
    // Index mapping: (e * n_angles + a) * n_depths
    
    std::vector<double> result(n_depths);
    
    size_t offset_00 = (idx_e1 * n_angles + idx_a1) * n_depths;
    size_t offset_01 = (idx_e1 * n_angles + idx_a2) * n_depths;
    size_t offset_10 = (idx_e2 * n_angles + idx_a1) * n_depths;
    size_t offset_11 = (idx_e2 * n_angles + idx_a2) * n_depths;
    
    // Bilinear:
    // f(x,y) ≈ (1-x)(1-y)f00 + (1-x)y f01 + x(1-y)f10 + xy f11
    
    for (int i = 0; i < n_depths; ++i) {
        double p00 = profiles_data[offset_00 + i];
        double p01 = profiles_data[offset_01 + i];
        double p10 = profiles_data[offset_10 + i];
        double p11 = profiles_data[offset_11 + i];
        
        double val = (1.0 - w_e) * (1.0 - w_a) * p00 +
                     (1.0 - w_e) * w_a * p01 +
                     w_e * (1.0 - w_a) * p10 +
                     w_e * w_a * p11;
        result[i] = val;
    }
    
    return result;
}

/**
 * 1D interpolation of the computed energy deposition profile onto a different grid.
 *
 * x:  Grid onto which the profile is interpolated. 
 * y:  Profile to interpolate.
 * xi: Grid point at which interpolation is done.
 */
double Interpolator::interpolate1D(const std::vector<double>& x, const std::vector<double>& y, double xi) {

    // x is sorted increasing (depths)
    
    if (xi <= x.front()) return y.front();
    if (xi >= x.back()) return y.back();
    
    auto it = std::lower_bound(x.begin(), x.end(), xi);
    int i = std::distance(x.begin(), it);
    if (i == 0) return y[0];
    
    double x0 = x[i-1];
    double x1 = x[i];
    double y0 = y[i-1];
    double y1 = y[i];
    
    double t = (xi - x0) / (x1 - x0);
    return y0 + t * (y1 - y0);
}

/**
 * Method that constructs the energy deposition profile 
 * for the wished energy, angle and for desired grid shape.
 *
 * energy:  Energy of physical particle in a macroparticle. 
 * angle:   Incidence angle of macroparticle.
 * targetDepths: Grid onto which the profile is interpolated.
 */
std::vector<double> Interpolator::getProfile(double energy, double angle, const std::vector<double>& targetDepths) {
    // 1. Interpolate in E, A
    std::vector<double> prof_std = interpolateProfile2D(energy, angle);
    
    // 2. Map to targetDepths
    std::vector<double> result(targetDepths.size());
    for (size_t i = 0; i < targetDepths.size(); ++i) {        
        result[i] = interpolate1D(depths_std, prof_std, targetDepths[i]);
    }
    return result;
}
