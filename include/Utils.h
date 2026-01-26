#ifndef UTILS_H
#define UTILS_H

#include <string>
#include <vector>
#include <H5Cpp.h>

namespace Utils {
    std::vector<double> readH5DoubleDataset(const H5::H5File& file, const std::string& datasetName);
    std::vector<int> readH5IntDataset(const H5::H5File& file, const std::string& datasetName);
    std::vector<double> readH5DoubleDatasetGroup(const H5::Group& group, const std::string& datasetName);
    std::vector<int> readH5IntDatasetGroup(const H5::Group& group, const std::string& datasetName);
    
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
