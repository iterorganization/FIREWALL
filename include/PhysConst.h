#ifndef PHYS_CONST_H
#define PHYS_CONST_H

namespace PhysConst {
    // Physical constants are best defined as constexpr
    constexpr double c      = 299'792'458.0;      // Speed of light (m/s) exact
    constexpr double m_e_u  = 0.000548579909;     // Electron mass in atomic units (u)
    constexpr double e      = 1.602176634e-19;    // Elementary charge (C)
    constexpr double m_u_kg = 1.66053906660e-27;  // 1 atomic mass unit in kg
    
    // Conversion factors can be calculated by the compiler
    constexpr double J_to_MeV = m_u_kg * 6.24e12;           // Joules to eV conversion
    constexpr double MeV_to_J = e * 1e9;                 // eV to Joules
}

#endif  // PHYS_CONST_H