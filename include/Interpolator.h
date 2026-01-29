#ifndef INCLUDE_INTERPOLATOR_H_
#define INCLUDE_INTERPOLATOR_H_

#include <string>
#include <vector>

class Interpolator {
   public:
    enum class InterpolationType { LINEAR, AKIMA, MAKIMA };

    explicit Interpolator(const std::string& h5Path);

    // Main function: given energy and angle, return the deposition profile
    // mapped to the provided target spatial grid (dx1, dx2 logic handles grid
    // generation externally or internally) The python code generates the grid
    // inside 'interpolate_with_dx'. Here we will return the profile on the
    // target grid provided as argument.
    std::vector<double> getProfile(double energy, double angle, const std::vector<double>& targetDepths);

    const std::vector<double>& getDepthsStd() const { return depths_std; }

    void setInterpolationType(InterpolationType type) { interpolationType = type; }

   private:
    std::vector<double> energies_train;  // size 10
    std::vector<double> angles_train;    // size 10
    std::vector<double> depths_std;      // size ~12000

    // Profiles data: [Energy_idx][Angle_idx][Depth_idx]
    // Flattened or structured. Let's use flattened 1D array for contiguous
    // memory. Index = (e_idx * 10 + a_idx) * n_depths + d_idx
    std::vector<double> profiles_data;

    int n_energies;
    int n_angles;
    int n_depths;

    InterpolationType interpolationType = InterpolationType::LINEAR;

    // Helper to interpolate the profile vector at specific (E, A) on the
    // standard depth grid
    std::vector<double> interpolateProfile2D(double energy, double angle);

    // Helper to interpolate 1D (standard grid -> target grid)
    double interpolate1D(const std::vector<double>& x, const std::vector<double>& y, double xi);

    // Spline helpers
    std::vector<double> getSplineDerivatives(const std::vector<double>& x, const std::vector<double>& y, InterpolationType type);
    double interpolateHermite(double xi, double x0, double x1, double y0, double y1, double d0, double d1);
};

#endif  // INCLUDE_INTERPOLATOR_H_