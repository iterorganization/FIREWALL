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
MAT_FILE="${DATA_DIR}/3deg7T.mat"
WALL_FILE="${DATA_DIR}/newiterwall_offset10cm.h5"
PART_FILE="${DATA_DIR}/part_out_eta_10x_fo.h5"
INTERP_FILE="${DATA_DIR}/interpolation_data.h5"
BUILD_DIR="build"
OUTPUT_FILE="${TEST_DIR}/results/results.h5"
REF_FILE="${TEST_DIR}/results/pipeline_results.h5"

echo "Starting pipeline..."

# 1. Fetch the input datasets
# Each dataset is described in tests/data_sources.json with its own Zenodo
# record, provenance and checksums; see docs/DATA_SOURCES.md. The fetcher
# verifies checksums and skips anything already present.
echo "Step 1: Fetching data..."
python3 scripts/fetch_data.py --dest "$DATA_DIR"

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
    ./"$BUILD_DIR"/firewall \
    --config examples/config.txt \
    --wall "$WALL_FILE" \
    --part "$PART_FILE" \
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
