/**
 * Module containing methods for reading HDF5 files.
 */

#include "Utils.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace Utils {

    std::vector<double> readH5DoubleDataset(hid_t location_id, const std::string& datasetName) {
        hid_t dataset_id = H5Dopen2(location_id, datasetName.c_str(), H5P_DEFAULT);
        if (dataset_id < 0) {
            throw std::runtime_error("Failed to open dataset: " + datasetName);
        }

        hid_t dataspace_id = H5Dget_space(dataset_id);
        int ndims = H5Sget_simple_extent_ndims(dataspace_id);
        std::vector<hsize_t> dims(ndims);
        H5Sget_simple_extent_dims(dataspace_id, dims.data(), NULL);

        size_t total_size = 1;
        for (int i = 0; i < ndims; ++i) {
            total_size *= dims[i];
        }

        std::vector<double> data(total_size);
        herr_t status = H5Dread(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, data.data());

        H5Sclose(dataspace_id);
        H5Dclose(dataset_id);

        if (status < 0) {
             throw std::runtime_error("Failed to read dataset: " + datasetName);
        }

        return data;
    }

    std::vector<int> readH5IntDataset(hid_t location_id, const std::string& datasetName) {
        hid_t dataset_id = H5Dopen2(location_id, datasetName.c_str(), H5P_DEFAULT);
        if (dataset_id < 0) {
            throw std::runtime_error("Failed to open dataset: " + datasetName);
        }

        hid_t dataspace_id = H5Dget_space(dataset_id);
        int ndims = H5Sget_simple_extent_ndims(dataspace_id);
        std::vector<hsize_t> dims(ndims);
        H5Sget_simple_extent_dims(dataspace_id, dims.data(), NULL);

        size_t total_size = 1;
        for (int i = 0; i < ndims; ++i) {
            total_size *= dims[i];
        }

        std::vector<int> data(total_size);
        herr_t status = H5Dread(dataset_id, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, data.data());

        H5Sclose(dataspace_id);
        H5Dclose(dataset_id);

        if (status < 0) {
             throw std::runtime_error("Failed to read dataset: " + datasetName);
        }

        return data;
    }

    std::vector<double> readH5DoubleDatasetGroup(hid_t group_id, const std::string& datasetName) {
        return readH5DoubleDataset(group_id, datasetName);
    }

    std::vector<int> readH5IntDatasetGroup(hid_t group_id, const std::string& datasetName) {
         return readH5IntDataset(group_id, datasetName);
    }
}