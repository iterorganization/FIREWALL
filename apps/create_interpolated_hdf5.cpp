#include <iostream>
#include <string>
#include <vector>

#include "H5Cpp.h"
#include "Interpolator.h"
#include "Utils.h"

// Helper function to create/write datasets since Utils mainly handles reading
void writeDoubleDataset(H5::H5File& file, const std::string& name, const std::vector<double>& data,
                        int rank, const hsize_t* dims) {
    H5::DataSpace dataspace(rank, dims);
    H5::DataSet dataset = file.createDataSet(name, H5::PredType::NATIVE_DOUBLE, dataspace);
    dataset.write(data.data(), H5::PredType::NATIVE_DOUBLE);
}

int main() {
    try {
        // 1. Initialize Interpolator
        // Assumes "training_data.h5" exists and contains 'energies', 'angles', 'depths', 'profiles'
        std::string inputPath = "../../data/interpolation_data.h5";
        Interpolator interp(inputPath);

        // 2. Define Output Grid
        std::vector<double> out_energies;
        for (double e = 0.5; e <= 50.0; e += 0.5) out_energies.push_back(e);

        std::vector<double> out_angles;
        for (double a = 1.0; a <= 90.0; a += 1.0) out_angles.push_back(a);

        // We use the training depths as our reference target depths
        std::vector<double> out_depths = interp.getDepthsStd();  // Assumes you added this getter

        size_t nE = out_energies.size();
        size_t nA = out_angles.size();
        size_t nD = out_depths.size();

        // 3. Perform Interpolation
        std::cout << "Interpolating " << nE << " energies and " << nA << " angles..." << std::endl;

        // Flattened 1D vector to hold 3D data: [nE][nA][nD]
        std::vector<double> flat_results;
        flat_results.reserve(nE * nA * nD);

        for (double e : out_energies) {
            for (double a : out_angles) {
                // This calls your bilinear interpolation + 1D depth mapping
                std::vector<double> profile = interp.getProfile(e, a, out_depths);
                flat_results.insert(flat_results.end(), profile.begin(), profile.end());
            }
        }

        // 4. Write to HDF5
        H5::H5File outFile("interpolated_output.h5", H5F_ACC_TRUNC);

        // Write Axes
        hsize_t dimE[1] = {nE};
        writeDoubleDataset(outFile, "energies", out_energies, 1, dimE);

        hsize_t dimA[1] = {nA};
        writeDoubleDataset(outFile, "angles", out_angles, 1, dimA);

        hsize_t dimD[1] = {nD};
        writeDoubleDataset(outFile, "depths", out_depths, 1, dimD);

        // Write Main 3D Profile Dataset
        hsize_t dim3D[3] = {nE, nA, nD};
        writeDoubleDataset(outFile, "profiles", flat_results, 3, dim3D);

        std::cout << "Success! Created 'interpolated_output.h5'" << std::endl;

    } catch (const H5::Exception& error) {
        error.printErrorStack();
        return -1;
    } catch (const std::exception& e) {
        std::cerr << "Standard Error: " << e.what() << std::endl;
        return -1;
    }

    return 0;
}
