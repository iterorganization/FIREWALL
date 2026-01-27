#ifndef UTILS_H
#define UTILS_H

#include <string>
#include <vector>
#include <hdf5.h>

namespace Utils {
    std::vector<double> readH5DoubleDataset(hid_t location_id, const std::string& datasetName);
    std::vector<int> readH5IntDataset(hid_t location_id, const std::string& datasetName);
    // Groups are also identified by hid_t in C API, so we can merge or keep aliases.
    // We'll keep the function names for compatibility but use hid_t.
    std::vector<double> readH5DoubleDatasetGroup(hid_t group_id, const std::string& datasetName);
    std::vector<int> readH5IntDatasetGroup(hid_t group_id, const std::string& datasetName);
    
    // Reads a CSV file with headers into a map of column name -> vector of doubles
    // Assumes all columns are numeric
    struct CSVData {
        std::vector<std::string> headers;
        std::vector<std::vector<double>> columns;
        int numRows;
    };
    
    CSVData readCSV(const std::string& filename);
}

#endif
