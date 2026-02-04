#ifndef INCLUDE_RESULTS_H
#define INCLUDE_RESULTS_H

#include <vector>

/**
 * @brief Stores the results of a simulation for a single wall element.
 */
struct Result {
    int wall_id;                ///< ID of the wall element.
    double surf_temp;           ///< Maximum surface temperature reached.
    std::vector<double> snaps;  ///< Temperature snapshots at specific times.
};

#endif  // INCLUDE_RESULTS_H