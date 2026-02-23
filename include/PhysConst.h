#ifndef PHYS_CONST_H
#define PHYS_CONST_H

/**
 * @brief Namespace for physical constants and conversion factors.
 */
namespace PhysConst {
    // Physical constants are best defined as constexpr
    constexpr double c      = 299'792'458.0;      ///< Speed of light (m/s) exact.
    constexpr double m_e_u  = 0.000548579909;     ///< Electron mass in atomic units (u).
    constexpr double e      = 1.602176634e-19;    ///< Elementary charge (C).
    constexpr double m_u_kg = 1.66053906660e-27;  ///< 1 atomic mass unit in kg.
    
    // Conversion factors can be calculated by the compiler
    constexpr double AMU_m2_s2_to_MeV = 6.24150907e12 * m_u_kg;           ///< AMU*m^2/s^2 to MeV conversion factor.
    constexpr double MeV_mm_to_J_m = 1.602176634e-10;         ///< MeV to Joules conversion factor.
}

#endif  // PHYS_CONST_H