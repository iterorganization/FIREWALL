#ifndef INCLUDE_RESULTS_H
#define INCLUDE_RESULTS_H

#include <vector>

struct Result {
    int wall_id;
    double surf_temp;
    std::vector<double> snaps;
};

#endif  // INCLUDE_RESULTS_H