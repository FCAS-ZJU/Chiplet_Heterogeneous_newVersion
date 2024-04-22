#pragma once

#include "cuda_runtime.h"

/**
 * @brief Send data from GPU to CPU.
 * @param __dst_x Destination address (CPU address) in X-axis.
 * @param __dst_y Destination address (CPU address) in Y-axis.
 * @param __src_x Source address (GPU address) in X-axis.
 * @param __src_y Source address (GPU address) in Y-axis.
 * @param __addr Data address.
 * @param __size Number of elements.
 */
__global__ void passMessage(
    int __dst_x, int __dst_y, int __src_x, int __srx_y, int64_t* __addr, int __size, int* __res);

/**
 * @brief Read data from CPU to GPU.
 * @param __dst_x Destination address (GPU address) in X-axis.
 * @param __dst_y Destination address (GPU address) in Y-axis.
 * @param __src_x Source address (CPU address) in X-axis.
 * @param __src_y Source address (CPU address) in Y-axis.
 * @param __addr Data address.
 * @param __size Number of elements.
 */
__global__ void readMessage(
    int __dst_x, int __dst_y, int __src_x, int __srx_y, int64_t* __addr, int __size, int* __res);
