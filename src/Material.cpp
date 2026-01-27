/**
 * Module for setting the material properties of Tungsten (W).
 */

#include "Material.h"

#include <algorithm>
#include <cmath>

/**
 * Constructor.
 */
Material::Material() {}

/**
 * Retreive the thermal conductivity of W at given temperature.
 *
 * T : Temperature at which thermal conductivity is evaluated.
 */
double Material::getK(double T) const {
    T = std::max(T, 300.0);
    double Tm = 3695.0;
    if (T <= Tm) {
        return 149.441 + (3.866e6) / (T * T) - (45.466e-3) * T + (13.193e-6) * (T * T) - (1.484e-9) * std::pow(T, 3);
    } else {
        double deltaT = T - Tm;
        return 66.6212 + 0.02086 * deltaT - (3.7585e-6) * (deltaT * deltaT);
    }
}

/**
 * Retreive the mass density of W at given temperature.
 *
 * T : Temperature at which mass density is evaluated.
 */
double Material::getRho(double T) const {
    T = std::max(T, 293.15);
    const double conv = 1000.0;
    const double T0 = 293.15;
    const double Tm = 3695.0;
    double dT = T - T0;

    if (T0 <= T && T <= Tm) {
        return conv * (19.25 - (2.66207e-4) * dT - (3.0595e-9) * (dT * dT) - (9.5185e-12) * std::pow(dT, 3));
    } else {
        double dTm = T - Tm;
        return conv * (16.267 - (7.679e-4) * dTm - (8.091e-8) * (dTm * dTm));
    }
}

/**
 * Retreive the heat capacity of W at given temperature.
 *
 * T : Temperature at which heat capacity is evaluated.
 */
double Material::getCp(double T) const {
    T = std::max(T, 300.0);
    const double conv = 1000.0 / 183.84;

    if (T <= 3080.0) {
        return conv * (21.868372 + (8.068661e-3) * T - (3.756196e-6) * (T * T) + (1.075862e-9) * std::pow(T, 3) + (1.406637e4) / (T * T));
    } else if (T <= 3695.0) {
        return conv * (2.022 + (1.315e-2) * T);
    } else {
        return conv * 51.3;
    }
}
