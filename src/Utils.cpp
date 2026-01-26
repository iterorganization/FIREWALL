/**
 * Module containing methods for reading HDF5 files.
 */

#include "Utils.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace Utils {

    std::vector<double> readH5DoubleDataset(const H5::H5File& file, const std::string& datasetName) {
        H5::DataSet dataset = file.openDataSet(datasetName);
        H5::DataSpace dataspace = dataset.getSpace();
        hsize_t dims_out[4];
        int ndims = dataspace.getSimpleExtentDims(dims_out, NULL);
        
        size_t total_size = 1;
        for(int i=0; i<ndims; ++i) total_size *= dims_out[i];
        
        std::vector<double> data(total_size);
        dataset.read(data.data(), H5::PredType::NATIVE_DOUBLE);
        return data;
    }

    std::vector<int> readH5IntDataset(const H5::H5File& file, const std::string& datasetName) {
        H5::DataSet dataset = file.openDataSet(datasetName);
        H5::DataSpace dataspace = dataset.getSpace();
        hsize_t dims_out[4];
        int ndims = dataspace.getSimpleExtentDims(dims_out, NULL);
        
        size_t total_size = 1;
        for(int i=0; i<ndims; ++i) total_size *= dims_out[i];
        
        std::vector<int> data(total_size);
        dataset.read(data.data(), H5::PredType::NATIVE_INT);
        return data;
    }

    std::vector<double> readH5DoubleDatasetGroup(const H5::Group& group, const std::string& datasetName) {
        H5::DataSet dataset = group.openDataSet(datasetName);
        H5::DataSpace dataspace = dataset.getSpace();
        hsize_t dims_out[4];
        int ndims = dataspace.getSimpleExtentDims(dims_out, NULL);
        
        size_t total_size = 1;
        for(int i=0; i<ndims; ++i) total_size *= dims_out[i];
        
        std::vector<double> data(total_size);
        dataset.read(data.data(), H5::PredType::NATIVE_DOUBLE);
        return data;
    }

    std::vector<int> readH5IntDatasetGroup(const H5::Group& group, const std::string& datasetName) {
        H5::DataSet dataset = group.openDataSet(datasetName);
        H5::DataSpace dataspace = dataset.getSpace();
        hsize_t dims_out[4];
        int ndims = dataspace.getSimpleExtentDims(dims_out, NULL);
        
        size_t total_size = 1;
        for(int i=0; i<ndims; ++i) total_size *= dims_out[i];
        
        std::vector<int> data(total_size);
        dataset.read(data.data(), H5::PredType::NATIVE_INT);
        return data;
    }
}
