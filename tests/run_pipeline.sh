#!/bin/bash
set -e

# Get the directory where this script is actually saved
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

# Change to that directory
cd "$SCRIPT_DIR/.."

echo "Running from: $(pwd)"

# Define paths
TEST_DIR="tests"
DATA_DIR="${TEST_DIR}/data"
ZIP_FILE="${TEST_DIR}/J2_data.zip"
MAT_FILE="${DATA_DIR}/3deg7T.mat"
INTERP_FILE="${DATA_DIR}/interpolation_data.h5"
BUILD_DIR="build"
OUTPUT_FILE="${TEST_DIR}/results/results.h5"
REF_FILE="${TEST_DIR}/results/pipeline_results.h5"

echo "Starting pipeline..."

# 1. Download and unzip
echo "Step 1: Downloading data..."
if [ ! -d "$DATA_DIR" ]; then
    mkdir -p "$TEST_DIR"
    curl -L -o "$ZIP_FILE" https://zenodo.org/records/18391920/files/J2_data.zip
    
    echo "Unzipping data..."
    # Unzip into tests directory. Assuming the zip contains a 'data' folder or we need to organize it.
    # Based on user prompt: "unzip it, it will create a data folder"
    unzip -o "$ZIP_FILE" -d "$TEST_DIR"
else
    echo "Data directory exists. Skipping download."
fi

# 2. Prepare data
echo "Step 2: Preparing interpolation data..."
if [ -f "$MAT_FILE" ]; then
    python3 scripts/prepare_data.py "$MAT_FILE" "$INTERP_FILE"
else
    echo "Error: MAT file not found at $MAT_FILE"
    exit 1
fi

# 3. Compile
echo "Step 3: Compiling..."
cmake -S . -B "$BUILD_DIR"
cmake --build "$BUILD_DIR" --config Release -- -j 8

# 4. Execute
echo "Step 4: Running simulation..."
/usr/bin/time -f "Real: %E \nUser: %U \nSys: %S" \
    ./"$BUILD_DIR"/heat_sim \
    --config examples/config.txt \
    --wall "${DATA_DIR}/newiterwall_offset10cm.h5" \
    --part "${DATA_DIR}/part_out_eta_10x_fo.h5" \
    --interp "$INTERP_FILE" \
    --out "$OUTPUT_FILE" \

# 5. Compare
echo "Step 5: Comparing results..."
if command -v h5diff &> /dev/null; then
    h5diff -v --relative=0.01 "$REF_FILE" "$OUTPUT_FILE" /surf_temp
    echo "Comparison complete."
else
    echo "Warning: h5diff not found. Skipping comparison."
    echo "You can install hdf5-tools to get h5diff."
fi

echo "Pipeline finished successfully."
