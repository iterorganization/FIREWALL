#ifndef INCLUDE_UTILS_H_
#define INCLUDE_UTILS_H_

#include <hdf5.h>

#include <string>
#include <vector>

/**
 * @brief Utility functions for HDF5 I/O and data processing.
 */
namespace Utils {

/**
 * @brief Reads a double dataset from an HDF5 location.
 *
 * @param location_id HDF5 location identifier (file or group).
 * @param datasetName Name of the dataset to read.
 * @return std::vector<double> The dataset contents.
 */
std::vector<double> readH5DoubleDataset(hid_t location_id, const std::string& datasetName);

/**
 * @brief Reads an integer dataset from an HDF5 location.
 *
 * @param location_id HDF5 location identifier (file or group).
 * @param datasetName Name of the dataset to read.
 * @return std::vector<int> The dataset contents.
 */
std::vector<int> readH5IntDataset(hid_t location_id, const std::string& datasetName);

// Groups are also identified by hid_t in C API, so we can merge or keep
// aliases. We'll keep the function names for compatibility but use hid_t.

/**
 * @brief Reads a double dataset from a specific HDF5 group.
 *
 * Alias for readH5DoubleDataset.
 *
 * @param group_id HDF5 group identifier.
 * @param datasetName Name of the dataset.
 * @return std::vector<double> The dataset contents.
 */
std::vector<double> readH5DoubleDatasetGroup(hid_t group_id, const std::string& datasetName);

/**
 * @brief Reads an integer dataset from a specific HDF5 group.
 *
 * Alias for readH5IntDataset.
 *
 * @param group_id HDF5 group identifier.
 * @param datasetName Name of the dataset.
 * @return std::vector<int> The dataset contents.
 */
std::vector<int> readH5IntDatasetGroup(hid_t group_id, const std::string& datasetName);

/**
 * @brief Reads a scalar double value from an HDF5 dataset.
 *
 * @param location_id HDF5 location identifier.
 * @param datasetName Name of the dataset.
 * @return double The scalar value.
 */
double readH5DoubleScalar(hid_t location_id, const std::string& datasetName);

/**
 * @brief Determines the number of particles in an HDF5 file.
 *
 * Reads the 'i_elm' dataset size from the 'groups/001' group.
 *
 * @param partPath Path to the HDF5 file.
 * @return size_t Number of particles.
 */
size_t getNumParticles(const std::string& partPath);

// Reads a CSV file with headers into a map of column name -> vector of doubles
// Assumes all columns are numeric

/**
 * @brief Structure to hold data read from a CSV file.
 */
struct CSVData {
    std::vector<std::string> headers;          ///< Column headers.
    std::vector<std::vector<double>> columns;  ///< Data organized by column.
    int numRows;                               ///< Number of rows in the CSV.
};

/**
 * @brief Reads a CSV file containing numeric data.
 *
 * @param filename Path to the CSV file.
 * @return CSVData Structure containing the CSV data.
 */
CSVData readCSV(const std::string& filename);
}  // namespace Utils

#endif  // INCLUDE_UTILS_H_
