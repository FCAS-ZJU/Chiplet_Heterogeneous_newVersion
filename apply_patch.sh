#!/bin/bash

if [[ -n "${SIMULATOR_ROOT}" ]]; then
    echo "SIMULATOR_ROOT is: ${SIMULATOR_ROOT}"
else
    echo "The environment variable SIMULATOR_ROOT is not defined."
fi

cd ${SIMULATOR_ROOT}/snipersim
git apply ../interchiplet/patch/snipersim.diff

