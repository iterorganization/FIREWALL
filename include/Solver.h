#ifndef INCLUDE_SOLVER_H_
#define INCLUDE_SOLVER_H_

#include <vector>
#include <assert.h>
#include <fstream>
#include <iostream>
#include <span>

#include "ConfigParser.h"
#include "Material.h"

/**
 * @brief Parameters for the heat equation simulation.
 *
 * Stores time steps, grid parameters, and physical constants for the simulation.
 */
struct SimulationParams {
    double dt_small;    ///< Small time step for initial high-flux period.
    double dt_large;    ///< Large time step for later cooling period.
    double t_start;     ///< Simulation start time.
    double t_interm;    ///< Time to switch from small to large time steps.
    double t_end;       ///< Simulation end time.
    double t_dep;       ///< Duration of energy deposition.
    double T_ini;       ///< Initial temperature.
    double L;           ///< Total depth of the domain.
    double L_sub;       ///< Depth of the refined grid region.
    double delta_x1;    ///< Grid spacing in the refined region.
    double delta_x2;    ///< Grid spacing in the coarse region.

    /**
     * @brief Loads parameters from a ConfigParser object.
     *
     * @param config The configuration parser containing parameter values.
     */
    void load(const ConfigParser& config) {
        dt_small = config.getDouble("dt_small", dt_small);
        dt_large = config.getDouble("dt_large", dt_large);
        t_start = config.getDouble("t_start", t_start);
        t_interm = config.getDouble("t_interm", t_interm);
        t_end = config.getDouble("t_end", t_end);
        t_dep = config.getDouble("t_dep", t_dep);
        T_ini = config.getDouble("T_ini", T_ini);
        L = config.getDouble("L", L);
        L_sub = config.getDouble("L_sub", L_sub);
        delta_x1 = config.getDouble("delta_x1", delta_x1);
        delta_x2 = config.getDouble("delta_x2", delta_x2);
        assert(t_dep > dt_small && "t_dep must be greater than dt_small");
    }

    /**
     * @brief Constructs parameters by loading from a configuration file.
     *
     * @param configPath Path to the configuration file.
     */
    SimulationParams(std::string configPath) {
        // --- Load Config ---
        ConfigParser config;
        // Check if config file exists before loading, or let it throw
        std::ifstream f(configPath.c_str());
        if (f.good()) {
            config.load(configPath);
            std::cout << "Loaded configuration from " << configPath << "\n";
        } else {
            std::cout << "Config file " << configPath << " not found, using defaults.\n";
        }
        
        this->load(config);
    }
};

/**
 * @brief Solves the 1D heat equation using the Finite Volume Method.
 *
 * Handles mesh generation, source term computation, and time integration (implicit Euler).
 */
class Solver {
   public:
    /**
     * @brief Constructs a solver with material properties and an initial grid.
     *
     * @param material The material properties.
     * @param x The initial spatial grid.
     */
    explicit Solver(const Material& material, const std::vector<double>& x);

    /**
     * @brief Solves the heat equation for a given energy deposition profile.
     *
     * @param dE_dx Energy deposition profile.
     * @param weights Particle weights.
     * @param coll_times Particle collision times.
     * @param params Simulation parameters.
     * @param coeff Scaling coefficient for source term.
     * @param times Time points for saving snapshots.
     * @param out_T Output vector to store temperature snapshots.
     */
    void solve(const std::vector<double>& dE_dx, const std::span<double>& weights,
               const std::span<double>& coll_times, const SimulationParams& params,
               double coeff,
               const std::vector<double>& times,
               std::vector<std::vector<double>>& out_T) const;  // NOLINT(runtime/references)
               
    /**
     * @brief Builds a non-uniform grid with two regions of different resolution.
     *
     * @param depths Vector to store the generated grid points.
     */
    void build_two_region_grid(const std::vector<double>& depths);

   private:
    Material mat; ///< Material properties.


    /**
     * @brief Computes the source term for the heat equation.
     *
     * @param dE_dx Spatial energy deposition profile.
     * @param weights Particle weights.
     * @param active_mask Mask indicating which particles are active in the current time step.
     * @param coeff Scaling coefficient.
     * @param src Output source term vector.
     */
    void compute_source(const std::vector<double>& dE_dx, const std::span<double>& weights, const std::vector<char>& active_mask, double coeff, std::vector<double>& src) const ;

    /**
     * @brief Assembles the tridiagonal matrix system for the implicit time step.
     *
     * @param T_guess Initial guess for temperatures (from previous iteration/step).
     * @param Tn Temperatures at the previous time step.
     * @param rho_cp_nodes Heat capacity at nodes.
     * @param src Source term.
     * @param dt Time step size.
     * @param a Output lower diagonal.
     * @param b Output main diagonal.
     * @param c Output upper diagonal.
     * @param d Output RHS vector.
     */
    void assemble_tridiag(const std::vector<double>& T_guess, const std::vector<double>& Tn,
                          const std::vector<double>& rho_cp_nodes, const std::vector<double>& src, double dt,
                          std::vector<double>& a,   // NOLINT(runtime/references)
                          std::vector<double>& b,   // NOLINT(runtime/references)
                          std::vector<double>& c,   // NOLINT(runtime/references)
                          std::vector<double>& d) const;  // NOLINT(runtime/references)

    /**
     * @brief Solves a tridiagonal linear system using the Thomas algorithm.
     *
     * @param a Lower diagonal.
     * @param b Main diagonal.
     * @param c Upper diagonal.
     * @param d RHS vector.
     * @return std::vector<double> Solution vector.
     */
    std::vector<double> thomas_solve(const std::vector<double>& a, const std::vector<double>& b,
                                     const std::vector<double>& c, const std::vector<double>& d) const;

    /**
     * @brief Performs a single implicit time step.
     *
     * Uses Newton-Raphson iteration to handle non-linear material properties.
     *
     * @param Tn Temperatures at previous time step.
     * @param src Source term.
     * @param dt Time step size.
     * @return std::vector<double> New temperatures.
     */
    std::vector<double> implicit_step(const std::vector<double>& Tn, const std::vector<double>& src, double dt) const;

    std::vector<double> h_face; ///< Distance between face centers.
    std::vector<double> dx_cell; ///< Cell widths.
};

#endif  // INCLUDE_SOLVER_H_
