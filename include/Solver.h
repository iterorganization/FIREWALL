#ifndef INCLUDE_SOLVER_H_
#define INCLUDE_SOLVER_H_

#include <vector>

#include "ConfigParser.h"
#include "Material.h"

struct SimulationParams {
    double dt_small;
    double dt_large;
    double t_start;
    double t_interm;
    double t_end;
    double t_dep;  // Deposition duration
    double T_ini;  // Initial temperature
    double coeff;  // Source term coefficient

    void load(const ConfigParser& config) {
        dt_small = config.getDouble("dt_small", dt_small);
        dt_large = config.getDouble("dt_large", dt_large);
        t_start = config.getDouble("t_start", t_start);
        t_interm = config.getDouble("t_interm", t_interm);
        t_end = config.getDouble("t_end", t_end);
        t_dep = config.getDouble("t_dep", t_dep);
        T_ini = config.getDouble("T_ini", T_ini);
        // coeff is usually calculated at runtime based on area, so maybe don't load it or allow override
        // coeff = config.getDouble("coeff", coeff);
    }
};

class Solver {
   public:
    explicit Solver(const Material& material);

    void solve(const std::vector<double>& dE_dx, const std::vector<double>& weights,
               const std::vector<double>& coll_times, const std::vector<double>& depths, const SimulationParams& params,
               std::vector<double>& out_T,      // NOLINT(runtime/references)
               std::vector<double>& out_times,  // NOLINT(runtime/references)
               int& out_Nx,                     // NOLINT(runtime/references)
               int& out_Nt);                    // NOLINT(runtime/references)

   private:
    Material mat;

    void build_two_region_grid(const std::vector<double>& x,
                               std::vector<double>& h_face,    // NOLINT(runtime/references)
                               std::vector<double>& dx_cell);  // NOLINT(runtime/references)

    void compute_source(const std::vector<double>& dE_dx, const std::vector<double>& weights,
                        const std::vector<bool>& active_mask, double coeff, int N_x, int N_p,
                        std::vector<double>& src);  // NOLINT(runtime/references)

    void assemble_tridiag(const std::vector<double>& T_guess, const std::vector<double>& Tn,
                          const std::vector<double>& rho_cp_nodes, const std::vector<double>& src, double dt,
                          const std::vector<double>& h_face, const std::vector<double>& dx_cell,
                          std::vector<double>& a,   // NOLINT(runtime/references)
                          std::vector<double>& b,   // NOLINT(runtime/references)
                          std::vector<double>& c,   // NOLINT(runtime/references)
                          std::vector<double>& d);  // NOLINT(runtime/references)

    std::vector<double> thomas_solve(const std::vector<double>& a, const std::vector<double>& b,
                                     const std::vector<double>& c, const std::vector<double>& d);

    void implicit_step(std::vector<double>& Tn,  // NOLINT(runtime/references)
                       const std::vector<double>& src, double dt, const std::vector<double>& h_face,
                       const std::vector<double>& dx_cell);
};

#endif  // INCLUDE_SOLVER_H_
