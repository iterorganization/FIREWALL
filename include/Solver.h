#ifndef SOLVER_H
#define SOLVER_H

#include <vector>
#include "Material.h"

struct SimulationParams {
    double dt_small;
    double dt_large;
    double t_start;
    double t_interm;
    double t_end;
    double t_dep;    // Deposition duration
    double T_ini;    // Initial temperature
    double coeff;    // Source term coefficient
};

class Solver {
public:
    explicit Solver(const Material& material);

    void solve(
        const std::vector<double>& dE_dx, 
        const std::vector<double>& weights, 
        const std::vector<double>& coll_times, 
        const std::vector<double>& depths, 
        const SimulationParams& params, 
        std::vector<double>& out_T, 
        std::vector<double>& out_times, 
        int& out_Nx, 
        int& out_Nt
    );

private:
    Material mat;

    void build_two_region_grid(
        const std::vector<double>& x, 
        std::vector<double>& h_face, 
        std::vector<double>& dx_cell
    );

    void compute_source(
        const std::vector<double>& dE_dx, 
        const std::vector<double>& weights, 
        const std::vector<bool>& active_mask, 
        double coeff, 
        int N_x, 
        int N_p, 
        std::vector<double>& src
    );

    void assemble_tridiag(
        const std::vector<double>& T_guess,
        const std::vector<double>& Tn,
        const std::vector<double>& rho_cp_nodes,
        const std::vector<double>& src,
        double dt,
        const std::vector<double>& h_face,
        const std::vector<double>& dx_cell,
        std::vector<double>& a,
        std::vector<double>& b,
        std::vector<double>& c,
        std::vector<double>& d
    );

    std::vector<double> thomas_solve(
        const std::vector<double>& a,
        const std::vector<double>& b,
        const std::vector<double>& c,
        const std::vector<double>& d
    );

    void implicit_step(
        std::vector<double>& Tn, 
        const std::vector<double>& src, 
        double dt, 
        const std::vector<double>& h_face, 
        const std::vector<double>& dx_cell
    );
};

#endif
