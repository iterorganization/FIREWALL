#ifndef INCLUDE_INTERPOLATOR_H_
#define INCLUDE_INTERPOLATOR_H_

#include <string>
#include <vector>

/**
 * @brief Handles interpolation of deposition profiles.
 *
 * Reads training data from an HDF5 file and provides interpolated profiles based on energy and angle.
 */
class Interpolator {
   public:
    /**
     * @brief Enumeration of supported interpolation methods.
     */
    enum class InterpolationType { LINEAR, AKIMA, MAKIMA };

    /**
     * @brief Constructs the Interpolator and loads data from the specified HDF5 file.
     *
     * @param h5Path Path to the HDF5 file containing training data.
     */
    explicit Interpolator(const std::string& h5Path);

    /**
     * @brief Gets the deposition profile for a specific energy and angle, mapped to a target depth grid.
     *
     * @param energy The particle energy.
     * @param angle The particle angle of incidence.
     * @param targetDepths The target depth grid for the output profile.
     * @return std::vector<double> The interpolated deposition profile on the target grid.
     */
    std::vector<double> getProfile(double energy, double angle, const std::vector<double>& targetDepths) const;

    /**
     * @brief Returns the standard depth grid used in the training data.
     *
     * @return const std::vector<double>& The standard depths.
     */
    const std::vector<double>& getDepthsStd() const { return depths_std; }

    /**
     * @brief Sets the interpolation type.
     *
     * @param type The desired interpolation method.
     */
    void setInterpolationType(InterpolationType type) { interpolationType = type; }

   private:
    std::vector<double> energies_train;  ///< Training energies (size 10).
    std::vector<double> angles_train;    ///< Training angles (size 10).
    std::vector<double> depths_std;      ///< Standard depths grid (size ~12000).

    /**
     * @brief Flattened profiles data: [Energy_idx][Angle_idx][Depth_idx].
     * Index = (e_idx * 10 + a_idx) * n_depths + d_idx.
     */
    std::vector<double> profiles_data;

    int n_energies; ///< Number of energy points.
    int n_angles;   ///< Number of angle points.
    int n_depths;   ///< Number of depth points.

    InterpolationType interpolationType = InterpolationType::MAKIMA;

    /**
     * @brief Helper to interpolate the profile vector at specific (E, A) on the standard depth grid.
     *
     * @param energy Energy value.
     * @param angle Angle value.
     * @return std::vector<double> Interpolated profile on standard grid.
     */
    std::vector<double> interpolateProfile2D(double energy, double angle) const;

    /**
     * @brief Helper to interpolate 1D (standard grid -> target grid).
     *
     * @param x Source grid points.
     * @param y Source values.
     * @param xi Target grid point.
     * @return double Interpolated value.
     */
    double interpolate1D(const std::vector<double>& x, const std::vector<double>& y, double xi) const;

    /**
     * @brief Computes spline derivatives for the given data.
     *
     * @param x Grid points.
     * @param y Values.
     * @param type Interpolation type.
     * @return std::vector<double> Derivatives at grid points.
     */
    std::vector<double> getSplineDerivatives(const std::vector<double>& x, const std::vector<double>& y, InterpolationType type) const;

    /**
     * @brief Performs Hermite interpolation.
     *
     * @param xi Target point.
     * @param x0 Interval start x.
     * @param x1 Interval end x.
     * @param y0 Interval start y.
     * @param y1 Interval end y.
     * @param d0 Interval start derivative.
     * @param d1 Interval end derivative.
     * @return double Interpolated value.
     */
    double interpolateHermite(double xi, double x0, double x1, double y0, double y1, double d0, double d1) const;
};

#endif  // INCLUDE_INTERPOLATOR_H_