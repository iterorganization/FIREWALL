/**
 * Module containing methods for reading HDF5 files.
 */

#include "Utils.h"

#include <fstream>
#include <iostream>
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

double readH5DoubleScalar(hid_t location_id, const std::string& datasetName) {
    hid_t dataset_id = H5Dopen2(location_id, datasetName.c_str(), H5P_DEFAULT);
    if (dataset_id < 0) {
        throw std::runtime_error("Failed to open dataset: " + datasetName);
    }

    double value;
    herr_t status = H5Dread(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, &value);

    H5Dclose(dataset_id);

    if (status < 0) {
        throw std::runtime_error("Failed to read dataset: " + datasetName);
    }

    return value;
}

std::vector<double> readH5DoubleDatasetGroup(hid_t group_id, const std::string& datasetName) { return readH5DoubleDataset(group_id, datasetName); }

std::vector<int> readH5IntDatasetGroup(hid_t group_id, const std::string& datasetName) { return readH5IntDataset(group_id, datasetName); }

namespace {
struct IterData {
    size_t max_size;
};

herr_t find_max_dataset_cb(hid_t group_id, const char* name, const H5L_info_t* /*info*/, void* op_data) {
    IterData* data = static_cast<IterData*>(op_data);
    
    // Try to open as dataset
    // We suppress error reporting for this check since we expect some opens to fail (if not a dataset)
    H5E_auto2_t old_func;
    void *old_client_data;
    H5Eget_auto2(H5E_DEFAULT, &old_func, &old_client_data);
    H5Eset_auto2(H5E_DEFAULT, NULL, NULL);

    hid_t dataset_id = H5Dopen2(group_id, name, H5P_DEFAULT);
    
    H5Eset_auto2(H5E_DEFAULT, old_func, old_client_data);

    if (dataset_id < 0) return 0; // Not a dataset or failure

    hid_t dataspace_id = H5Dget_space(dataset_id);
    if (dataspace_id >= 0) {
        hsize_t dims[1];
        if (H5Sget_simple_extent_dims(dataspace_id, dims, NULL) > 0) {
            size_t s = (size_t)dims[0];
            if (s > data->max_size) {
                data->max_size = s;
            }
        }
        H5Sclose(dataspace_id);
    }
    H5Dclose(dataset_id);

    return 0; // Continue iteration
}
}

size_t getNumParticles(const std::string& partPath) {
    hid_t file_id = H5Fopen(partPath.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    if (file_id < 0) {
        throw std::runtime_error("Failed to open particles file: " + partPath);
    }

    hid_t group_id = H5Gopen2(file_id, "groups/001", H5P_DEFAULT);
    if (group_id < 0) {
        H5Fclose(file_id);
        throw std::runtime_error("Failed to open group 'groups/001' in " + partPath);
    }

    IterData data = {0};
    hsize_t idx = 0; // Start iteration from index 0
    herr_t status = H5Literate(group_id, H5_INDEX_NAME, H5_ITER_INC, &idx, find_max_dataset_cb, &data);

    H5Gclose(group_id);
    H5Fclose(file_id);

    if (status < 0) {
        throw std::runtime_error("Failed to iterate over group 'groups/001' in " + partPath);
    }

    if (data.max_size == 0) {
         throw std::runtime_error("No valid dataset found in 'groups/001' to determine particle count.");
    }

    return data.max_size;
}

}  // namespace Utils