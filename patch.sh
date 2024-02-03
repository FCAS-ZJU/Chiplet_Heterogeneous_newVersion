#!/bin/bash

if [[ -n "${SIMULATOR_ROOT}" ]]; then
    echo "SIMULATOR_ROOT is: ${SIMULATOR_ROOT}"
else
    echo "The environment variable SIMULATOR_ROOT is not defined."
    exit
fi

cd ${SIMULATOR_ROOT}
rm -rf .changed_files
mkdir .changed_files

cd ${SIMULATOR_ROOT}/snipersim
git diff > ../interchiplet/patch/snipersim.diff
snipersim_changed_file_list=$(git diff --name-only)

cd ${SIMULATOR_ROOT}
file_list=($snipersim_changed_file_list)
echo ${#file_list[@]} "Files has changed."
for item in "${file_list[@]}"; do
    echo $item
    cp --parent snipersim/$item .changed_files/
done
