#!/bin/bash

PYTHON=${PYTHON:-"python"}

set -euo pipefail

ACTION=${1:-build}

echo "Building package resample2d"
cd ./mmaction/ops/resample2d_package
if [ -d "build" ]; then
    rm -r build
fi

$PYTHON setup.py build_ext --inplace

echo "Building package trajectory_conv..."
cd ../trajectory_conv_package
if [ -d "build" ]; then
    rm -r build
fi

$PYTHON setup.py build_ext --inplace

echo "Building roi align op..."
cd ../roi_align
if [ -d "build" ]; then
    rm -r build
fi
$PYTHON setup.py build_ext --inplace

echo "Building roi pool op..."
cd ../roi_pool
if [ -d "build" ]; then
    rm -r build
fi
$PYTHON setup.py build_ext --inplace

echo "Building nms op..."
cd ../nms
if [ -d "build" ]; then
    rm -r build
fi
$PYTHON setup.py build_ext --inplace

if [[ "$ACTION" == "test" || "$ACTION" == "tests" || "$ACTION" == "--test" ]]; then
  echo "Running fp16/bf16 smoke + parity tests..."
  $PYTHON tools/test_fp16_bf16_ops.py
fi
