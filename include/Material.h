#ifndef INCLUDE_MATERIAL_H_
#define INCLUDE_MATERIAL_H_

#include <string>
#include <vector>

/**
 * @brief Represents material properties for thermal calculations.
 *
 * Currently implements properties for Tungsten (W) including thermal conductivity,
 * density, and specific heat capacity as functions of temperature.
 */
class Material {
   public:
    /**
     * @brief Constructs a new Material object.
     *
     * Initializes material properties (currently hardcoded for Tungsten).
     */
    Material();

    /**
     * @brief Calculates thermal conductivity (k) at a given temperature.
     *
     * @param T Temperature in Kelvin.
     * @return double Thermal conductivity in W/(m*K).
     */
    double getK(double T) const;

    /**
     * @brief Calculates density (rho) at a given temperature.
     *
     * @param T Temperature in Kelvin.
     * @return double Density in kg/m^3.
     */
    double getRho(double T) const;

    /**
     * @brief Calculates specific heat capacity (cp) at a given temperature.
     *
     * @param T Temperature in Kelvin.
     * @return double Specific heat capacity in J/(kg*K).
     */
    double getCp(double T) const;

    /**
     * @brief Helper to get all properties for a vector of temperatures.
     *
     * Computes k, rho, and cp for each temperature in the input vector.
     *
     * @param T Vector of input temperatures.
     * @param k Output vector for thermal conductivity.
     * @param rho Output vector for density.
     * @param cp Output vector for specific heat capacity.
     */
    void getProperties(const std::vector<double>& T,
                       std::vector<double>& k,    // NOLINT(runtime/references)
                       std::vector<double>& rho,  // NOLINT(runtime/references)
                       std::vector<double>& cp)   // NOLINT(runtime/references)
        const;
};

#endif  // INCLUDE_MATERIAL_H_
