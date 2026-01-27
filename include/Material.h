#ifndef MATERIAL_H
#define MATERIAL_H

#include <string>
#include <vector>

class Material {
   public:
    // No longer requires a CSV path for initialization
    Material();

    // Analytical property functions for Tungsten
    double getK(double T) const;
    double getRho(double T) const;
    double getCp(double T) const;

    // Helper to get all properties for a vector of temperatures
    void getProperties(const std::vector<double>& T, std::vector<double>& k,
                       std::vector<double>& rho, std::vector<double>& cp) const;
};

#endif
