#!/bin/bash

if [[ -n "${SIMULATOR_ROOT}" ]]; then
    echo "SIMULATOR_ROOT is: ${SIMULATOR_ROOT}"
else
    echo "The environment variable SIMULATOR_ROOT is not defined."
    exit
fi

# Pathc for Sniper
cd ${SIMULATOR_ROOT}/snipersim
git apply ../interchiplet/patch/snipersim.diff

# Patch for GPGPUSim
cd ${SIMULATOR_ROOT}/gpgpu-sim
git apply ../interchiplet/patch/gpgpu-sim.diff

# Patch for GEM5
cd ${SIMULATOR_ROOT}/gem5
git apply ../interchiplet/patch/gem5.diff
