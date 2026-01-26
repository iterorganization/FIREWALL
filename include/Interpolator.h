#ifndef INTERPOLATOR_H
#define INTERPOLATOR_H

#include <vector>
#include <string>

class Interpolator {
public:
    Interpolator(const std::string& h5Path);

    // Main function: given energy and angle, return the deposition profile 
    // mapped to the provided target spatial grid (dx1, dx2 logic handles grid generation externally or internally)
    // The python code generates the grid inside 'interpolate_with_dx'.
    // Here we will return the profile on the target grid provided as argument.
    std::vector<double> getProfile(double energy, double angle, const std::vector<double>& targetDepths);

    const std::vector<double>& getDepthsStd() const { 
        return depths_std; 
    }

private:
    std::vector<double> energies_train; // size 10
    std::vector<double> angles_train;   // size 10
    std::vector<double> depths_std;     // size ~12000
    
    // Profiles data: [Energy_idx][Angle_idx][Depth_idx]
    // Flattened or structured. Let's use flattened 1D array for contiguous memory.
    // Index = (e_idx * 10 + a_idx) * n_depths + d_idx
    std::vector<double> profiles_data; 
    
    int n_energies;
    int n_angles;
    int n_depths;

    // Helper to interpolate the profile vector at specific (E, A) on the standard depth grid
    std::vector<double> interpolateProfile2D(double energy, double angle);
    
    // Helper to interpolate 1D (standard grid -> target grid)
    double interpolate1D(const std::vector<double>& x, const std::vector<double>& y, double xi);
};

#endif
