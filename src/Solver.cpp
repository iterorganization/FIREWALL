/**
 * Module for solving the 1D nonlinear heat equation for a given triangle.
 */

#include "Solver.h"

#include <omp.h>

#include <algorithm>
#include <cmath>
#include <iostream>

/**
 * Constructor.
 */
Solver::Solver(const Material& material, const std::vector<double>& x) : mat(material) { build_two_region_grid(x); }

/**
 * Build the vectors of inter-node distances and cell volumes.
 *
 * x:       In depth grid.
 * h_face:  Vector containing inter-node distances.
 * dx_cell: Vector containing the volumes around each node.
 */
void Solver::build_two_region_grid(const std::vector<double>& x) {
    size_t N = x.size();
    if (N > 1) {
        h_face.resize(N - 1);
        for (size_t f = 0; f < N - 1; ++f) {
            h_face[f] = x[f + 1] - x[f];
        }
    } else {
        h_face.clear();
    }

    dx_cell.resize(N);
    for (size_t i = 0; i < N; ++i) {
        if (N == 1) {
            dx_cell[i] = 0.0;
        } else {
            if (i == 0) {
                dx_cell[i] = 0.5 * h_face[0];
            } else if (i == N - 1) {
                dx_cell[i] = 0.5 * h_face[N - 2];
            } else {
                dx_cell[i] = 0.5 * (h_face[i - 1] + h_face[i]);
            }
        }
    }
}

/**
 * Assembles the source term.
 *
 * dE_dx:       Matrix containing the energy deposition profiles for each
 * incident macroparticle. weights:     Vector containing the weight (number of
 * physical particles) for each incident macroparticle. active_mask: Vector
 * containing the time intervals over which a macroparticle deposits energy.
 * N_x:         Number of nodes in the in depth grid. N_p:         Number of
 * macroparticles. src:         Source term.
 */
void Solver::compute_source(const std::vector<double>& dE_dx, const std::span<double>& weights, const std::vector<bool>& active_mask, double coeff, std::vector<double>& src) const {

// Parallelize over spatial grid
#pragma omp parallel for
    for (int i = 0; i < src.size(); ++i) {
        double s = 0.0;

        for (int j = 0; j < active_mask.size(); ++j) {
            if (active_mask[j]) {
                s += dE_dx[i * active_mask.size() + j] * weights[j];
            }
        }
        src[i] = coeff * s;
    }
}

/**
 * Assembles the tridiagonal system corre.
 *
 * T_guess:      Guess of the solution in the fixed point iteration.
 * Tn:           Actual temperature profile.
 * rho_cp_nodes: Vector containing the values of rho*c_p.
 * src:          Source term.
 * dt:           Time step.
 * h_face:       Vector containing inter-node distances.
 * dx_cell:      Vector containing the volumes around each node.
 * a:            Vector containing the a_i values.
 * b:            Vector containing the b_i values.
 * c:            Vector containing the c_i values.
 * d:            Vector containing the d_i values.
 */
void Solver::assemble_tridiag(const std::vector<double>& T_guess, const std::vector<double>& Tn, const std::vector<double>& rho_cp_nodes,
                              const std::vector<double>& src, double dt, const std::vector<double>& h_face, const std::vector<double>& dx_cell,
                              std::vector<double>& a, std::vector<double>& b, std::vector<double>& c, std::vector<double>& d) const {
    size_t N = T_guess.size();
    a.assign(N, 0.0);
    b.assign(N, 0.0);
    c.assign(N, 0.0);
    d.assign(N, 0.0);

    // Get k at nodes
    std::vector<double> k_nodes(N);

    for (size_t i = 0; i < N; ++i) k_nodes[i] = mat.getK(T_guess[i]);

    std::vector<double> k_face;
    if (N > 1) {
        k_face.resize(N - 1);
        for (size_t f = 0; f < N - 1; ++f) {
            double kL = k_nodes[f];
            double kR = k_nodes[f + 1];
            if (kL > 0.0 && kR > 0.0) {
                k_face[f] = 2.0 * kL * kR / (kL + kR);
            } else {
                k_face[f] = 0.5 * (kL + kR);
            }
        }
    }

    for (size_t i = 0; i < N; ++i) {
        double b_i = rho_cp_nodes[i];

        if (i > 0) {
            double hL = h_face[i - 1];
            double dx_i = dx_cell[i];
            double val = dt * (k_face[i - 1] / (dx_i * hL));
            a[i] = -val;
            b_i += val;
        } else {
            a[i] = 0.0;
        }

        if (i < N - 1) {
            double hR = h_face[i];
            double dx_i = dx_cell[i];
            double val = dt * (k_face[i] / (dx_i * hR));
            c[i] = -val;
            b_i += val;
        } else {
            c[i] = 0.0;
        }

        b[i] = b_i;
        d[i] = rho_cp_nodes[i] * Tn[i] + dt * src[i];
    }

    // BCs
    if (N >= 2) {
        b[0] = 1.0;
        c[0] = -1.0;
        a[0] = 0.0;
        d[0] = 0.0;
        b[N - 1] = 1.0;
        a[N - 1] = -1.0;
        c[N - 1] = 0.0;
        d[N - 1] = 0.0;
    } else {
        b[0] = 1.0;
        a[0] = 0.0;
        c[0] = 0.0;
        d[0] = Tn[0];
    }
}

/**
 * Solves the tridiagonal system using the Thomas algorithm.
 *
 * a: Vector containing the a_i values.
 * b: Vector containing the b_i values.
 * c: Vector containing the c_i values.
 * d: Vector containing the d_i values.
 */
std::vector<double> Solver::thomas_solve(const std::vector<double>& a, const std::vector<double>& b, const std::vector<double>& c,
                                         const std::vector<double>& d) const {
    size_t N = b.size();
    std::vector<double> cp(N);
    std::vector<double> dp(N);
    std::vector<double> x(N);

    double denom = b[0];
    if (std::abs(denom) < 1e-30) denom = 1e-30;

    cp[0] = c[0] / denom;
    dp[0] = d[0] / denom;

    for (size_t i = 1; i < N; ++i) {
        denom = b[i] - a[i] * cp[i - 1];
        if (std::abs(denom) < 1e-30) denom = 1e-30;
        if (i < N - 1) {
            cp[i] = c[i] / denom;
        }
        dp[i] = (d[i] - a[i] * dp[i - 1]) / denom;
    }

    x[N - 1] = dp[N - 1];
    for (int i = N - 2; i >= 0; --i) {
        x[i] = dp[i] - cp[i] * x[i + 1];
    }
    return x;
}

/**
 * Perform an implicit step of the solver.
 *
 * Tn:      Actual temperature profile.
 * src:     Time dependent source term.
 * dt:      Time step.
 * h_face:  Vector containing inter-node distances.
 * dx_cell: Vector containing the volumes around each node.
 */
std::vector<double> Solver::implicit_step(const std::vector<double>& Tn, const std::vector<double>& src, double dt, const std::vector<double>& h_face,
                           const std::vector<double>& dx_cell) const {
    size_t N = Tn.size();
    std::vector<double> T_guess = Tn;  // Copy
    std::vector<double> T_new;

    std::vector<double> rho_nodes(N);
    std::vector<double> cp_nodes(N);
    std::vector<double> rho_cp_nodes(N);

    std::vector<double> a, b, c, d;

    int max_iter = 10;
    double tol = 1e-11;

    int m; double maxdiff;
    for (m = 0; m < max_iter; ++m) {
        for (size_t i = 0; i < N; ++i) {
            rho_nodes[i] = mat.getRho(T_guess[i]);
            cp_nodes[i] = mat.getCp(T_guess[i]);
            rho_cp_nodes[i] = rho_nodes[i] * cp_nodes[i];
        }

        assemble_tridiag(T_guess, Tn, rho_cp_nodes, src, dt, h_face, dx_cell, a, b, c, d);
        T_new = thomas_solve(a, b, c, d);

        maxdiff = 0.0;
        for (size_t i = 0; i < N; ++i) {
            double diff = std::abs(T_new[i] - T_guess[i]);
            if (diff > maxdiff) maxdiff = diff;
        }

        T_guess = T_new;
        if (maxdiff < tol) break;
    }

    if (m == max_iter) {
        std::cerr << "Warning: implicit_step did not converge in " << max_iter << " iterations. maxdiff=" << maxdiff << std::endl;
    }

    return T_guess;
}

/**
 * Solve 1D nonlinear heat equation.
 *
 * dE_dx:      Matrix containing the energy deposition profiles for each
 * incident macroparticle. weights:    Vector containing the weight (number of
 * physical particles) for each incident macroparticle. coll_times: Vector
 * containing the time of impact of each macroparticle. depths: In depth grid on
 * which the Temperature profile is evaluated. params:     Parameters used for
 * the simulation. out_T:      Matrix containing the temperature profile for
 * every time step. out_times: Vector containing the times at which the
 * temperature profile is evaluated. out_Nx:     Number of nodes in the in depth
 * grid. out_Nt:     Number of time points at which the temperature profile is
 * evaluated.
 */
void Solver::solve(const std::vector<double>& dE_dx, const std::span<double>& weights, const std::span<double>& coll_times,
                   const std::vector<double>& depths, const SimulationParams& params, double coeff, std::vector<std::vector<double>>& out_T, std::vector<double>& out_times) const {
    double t_now = params.t_start;
    double dt;
    int n = 1;

    int N_p = coll_times.size();
    int N_x = depths.size();

    std::vector<bool> active_mask(N_p);
    std::vector<double> src(N_x);

    // Initial condition
    // std::vector<double> T(N_x, params.T_ini);

    while (n < out_times.size()) {
        
        
        // Active mask
        for (int j = 0; j < N_p; ++j) {
            active_mask[j] = (coll_times[j] <= t_now) && (t_now <= (coll_times[j] + params.t_dep));
        }
        
        compute_source(dE_dx, weights, active_mask, coeff, src);
        
        out_T[n] = implicit_step(out_T[n-1], src, dt, h_face, dx_cell);
        
        if (t_now >= params.t_interm)
            dt = params.dt_large;
        else
            dt = params.dt_small;
            
        t_now += dt;

        out_times[n] = t_now;
        
        n++;
    }
}
