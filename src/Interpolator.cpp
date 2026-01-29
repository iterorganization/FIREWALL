/**
 * Module for interpolating the energy deposition profile.
 */

#include "Interpolator.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>

#include "Utils.h"

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
    n_depths = depths_std.size();
}

/**
 * Interpolate energy deposition profile from incidence angle and energy.
 * The profile has the same shape as depths_std
 *
 * energy:   Energy of physical particle in a macroparticle.
 * angle:    Incidence angle of macroparticle.
 */
std::vector<double> Interpolator::interpolateProfile2D(double energy, double angle) const {
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

        double val = (1.0 - w_e) * (1.0 - w_a) * p00 + (1.0 - w_e) * w_a * p01 + w_e * (1.0 - w_a) * p10 + w_e * w_a * p11;
        result[i] = val;
    }

    return result;
}

/**
 * 1D interpolation of the computed energy deposition profile onto a different
 * grid.
 *
 * x:  Grid onto which the profile is interpolated.
 * y:  Profile to interpolate.
 * xi: Grid point at which interpolation is done.
 */
double Interpolator::interpolate1D(const std::vector<double>& x, const std::vector<double>& y, double xi) const {
    // x is sorted increasing (depths)

    if (xi <= x.front()) return y.front();
    if (xi >= x.back()) return y.back();

    auto it = std::lower_bound(x.begin(), x.end(), xi);
    int i = std::distance(x.begin(), it);
    if (i == 0) return y[0];

    double x0 = x[i - 1];
    double x1 = x[i];
    double y0 = y[i - 1];
    double y1 = y[i];

    double t = (xi - x0) / (x1 - x0);
    return y0 + t * (y1 - y0);
}

std::vector<double> Interpolator::getSplineDerivatives(const std::vector<double>& x, const std::vector<double>& y, InterpolationType type) const {
    size_t n = x.size();
    if (n < 5) {
        // Fallback to zeros or simple slope if too small, but here we assume n is large
        return std::vector<double>(n, 0.0);
    }

    std::vector<double> m(n - 1);
    for (size_t i = 0; i < n - 1; ++i) {
        m[i] = (y[i + 1] - y[i]) / (x[i + 1] - x[i]);
    }

    // Extended slopes with boundary extrapolation (Akima style)
    // We need m[-2], m[-1] ... m[n-1], m[n]
    // Store in a vector of size n+3 to handle indices easily: 0->-2, 1->-1, 2->0, ..., n->n-2, n+1->n-1, n+2->n
    // Let's use direct variables for boundaries to avoid allocation if possible,
    // but constructing a padded vector is cleaner.
    
    std::vector<double> m_ext;
    m_ext.reserve(n + 3);

    double m0 = m[0];
    double m1 = m[1];
    double m_minus_1 = 2.0 * m0 - m1;
    double m_minus_2 = 2.0 * m_minus_1 - m0;

    m_ext.push_back(m_minus_2);
    m_ext.push_back(m_minus_1);
    m_ext.insert(m_ext.end(), m.begin(), m.end());

    double m_end_1 = m[n - 2];
    double m_end_2 = m[n - 3];
    double m_plus_1 = 2.0 * m_end_1 - m_end_2;
    double m_plus_2 = 2.0 * m_plus_1 - m_end_1;

    m_ext.push_back(m_plus_1);
    m_ext.push_back(m_plus_2);

    // Now m_ext[i+2] corresponds to m[i]
    // We want d[i] for i=0..n-1
    // d[i] uses m[i-2], m[i-1], m[i], m[i+1]
    // In m_ext, m[i] is at index i+2.
    // m[i-2] -> m_ext[i]
    // m[i-1] -> m_ext[i+1]
    // m[i]   -> m_ext[i+2]
    // m[i+1] -> m_ext[i+3]

    std::vector<double> derivs(n);

    for (size_t i = 0; i < n; ++i) {
        double m_im2 = m_ext[i];
        double m_im1 = m_ext[i + 1];
        double m_i = m_ext[i + 2];
        double m_ip1 = m_ext[i + 3];

        double w1 = 0.0, w2 = 0.0;

        if (type == InterpolationType::AKIMA) {
            w1 = std::abs(m_ip1 - m_i);
            w2 = std::abs(m_im1 - m_im2);
        } else if (type == InterpolationType::MAKIMA) {
            w1 = std::abs(m_ip1 - m_i) + std::abs(m_ip1 + m_i) * 0.5;
            w2 = std::abs(m_im1 - m_im2) + std::abs(m_im1 + m_im2) * 0.5;
        }

        if (std::abs(w1 + w2) < 1e-12) {
            derivs[i] = (m_im1 + m_i) * 0.5;
        } else {
            derivs[i] = (w1 * m_im1 + w2 * m_i) / (w1 + w2);
        }
    }

    return derivs;
}

double Interpolator::interpolateHermite(double xi, double x0, double x1, double y0, double y1, double d0, double d1) const {
    double h = x1 - x0;
    if (std::abs(h) < 1e-12) return y0;

    double t = (xi - x0) / h;
    double t2 = t * t;
    double t3 = t2 * t;

    double h00 = 2.0 * t3 - 3.0 * t2 + 1.0;
    double h10 = t3 - 2.0 * t2 + t;
    double h01 = -2.0 * t3 + 3.0 * t2;
    double h11 = t3 - t2;

    return h00 * y0 + h10 * h * d0 + h01 * y1 + h11 * h * d1;
}

/**
 * Method that constructs the energy deposition profile
 * for the wished energy, angle and for desired grid shape.
 *
 * energy:  Energy of physical particle in a macroparticle.
 * angle:   Incidence angle of macroparticle.
 * targetDepths: Grid onto which the profile is interpolated.
 */
std::vector<double> Interpolator::getProfile(double energy, double angle, const std::vector<double>& targetDepths) const {
    // 1. Interpolate in E, A
    std::vector<double> prof_std = interpolateProfile2D(energy, angle);

    std::vector<double> result(targetDepths.size());

    if (interpolationType == InterpolationType::LINEAR) {
        // 2. Map to targetDepths using Linear
        for (size_t i = 0; i < targetDepths.size(); ++i) {
            result[i] = interpolate1D(depths_std, prof_std, targetDepths[i]);
        }
    } else {
        // 2. Map to targetDepths using Spline (Akima/Makima)
        std::vector<double> derivs = getSplineDerivatives(depths_std, prof_std, interpolationType);

        // Assume sorted targetDepths for efficiency, but using lower_bound is safe and reasonably fast
        for (size_t i = 0; i < targetDepths.size(); ++i) {
            double xi = targetDepths[i];

            // Handle out of bounds
            if (xi <= depths_std.front()) {
                result[i] = prof_std.front();
                continue;
            }
            if (xi >= depths_std.back()) {
                result[i] = prof_std.back();
                continue;
            }

            auto it = std::lower_bound(depths_std.begin(), depths_std.end(), xi);
            int idx = std::distance(depths_std.begin(), it);
            if (idx == 0) idx = 1; // Should be handled by <= front() check but safety first

            int idx0 = idx - 1;
            int idx1 = idx;

            double x0 = depths_std[idx0];
            double x1 = depths_std[idx1];
            double y0 = prof_std[idx0];
            double y1 = prof_std[idx1];
            double d0 = derivs[idx0];
            double d1 = derivs[idx1];

            result[i] = interpolateHermite(xi, x0, x1, y0, y1, d0, d1);
        }
    }
    return result;
}
