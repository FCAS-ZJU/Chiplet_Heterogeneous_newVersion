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

# Pathc for Sniper
cd ${SIMULATOR_ROOT}/snipersim
git diff > ../interchiplet/patch/snipersim.diff
snipersim_changed_file_list="$(git diff --name-only)"

cd ${SIMULATOR_ROOT}
file_list=($snipersim_changed_file_list)
echo ${#file_list[@]} "Files has changed."
for item in "${file_list[@]}"; do
    echo $item
    cp --parent snipersim/$item .changed_files/
done

# Patch for GPGPUSim
cd ${SIMULATOR_ROOT}/gpgpu-sim
git diff > ../interchiplet/patch/gpgpu-sim.diff
gpgpu_sim_changed_file_list="$(git diff --name-only)"

cd ${SIMULATOR_ROOT}
file_list=($gpgpu_sim_changed_file_list)
echo ${#file_list[@]} "Files has changed."
for item in "${file_list[@]}"; do
    echo $item
    cp --parent gpgpu-sim/$item .changed_files/
done

# Patch for GEM5
cd ${SIMULATOR_ROOT}/gem5
git diff > ../interchiplet/patch/gem5.diff
gem5_changed_file_list="$(git diff --name-only)"

cd ${SIMULATOR_ROOT}
file_list=($gem5_changed_file_list)
echo ${#file_list[@]} "Files has changed."
for item in "${file_list[@]}"; do
    echo $item
    cp --parent gem5/$item .changed_files/
done
