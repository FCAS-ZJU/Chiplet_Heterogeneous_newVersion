#pragma once

#include <tuple>

#include "cuda_runtime_api.h"

/**
 * @brief Enter Barrier.
 * @param __uid Barrier ID.
 * @param __src_x Source address in X-axis.
 * @param __src_y Source address in Y-axis.
 * @param __count Number of item in barrier.
 */
extern __host__ cudaError_t CUDARTAPI barrier(int __uid, int __src_x, int __src_y, int __count = 0);

/**
 * @brief Lock remote chiplet.
 * @param __dst_x Destination address in X-axis.
 * @param __dst_y Destination address in Y-axis.
 * @param __src_x Source address in X-axis.
 * @param __src_y Source address in Y-axis.
 */
extern __host__ cudaError_t CUDARTAPI launchResource(int __dst_x, int __dst_y, int __src_x,
                                                   int __src_y);

/**
 * @brief Unlock remote chiplet.
 * @param __dst_x Destination address in X-axis.
 * @param __dst_y Destination address in Y-axis.
 * @param __src_x Source address in X-axis.
 * @param __src_y Source address in Y-axis.
 */
extern __host__ cudaError_t CUDARTAPI unlockResource(int __dst_x, int __dst_y, int __src_x,
                                                     int __src_y);

/**
 * @brief Lock remote chiplet.
 * @param __dst_x Destination address in X-axis.
 * @param __dst_y Destination address in Y-axis.
 * @param __src_x Source address in X-axis.
 * @param __src_y Source address in Y-axis.
 */
extern __host__ cudaError_t CUDARTAPI waitLauncher(int __dst_x, int __dst_y, int* __src_x,
                                                 int* __src_y);

/**
 * @brief Send data to remote chiplet.
 * @param __dst_x Destination address in X-axis.
 * @param __dst_y Destination address in Y-axis.
 * @param __src_x Source address in X-axis.
 * @param __src_y Source address in Y-axis.
 * @param __addr Data address.
 * @param __nbyte Number of bytes.
 */
extern __host__ cudaError_t CUDARTAPI sendMessage(int __dst_x, int __dst_y, int __src_x,
                                                  int __srx_y, void* __addr, int __nbyte);

/**
 * @brief Read data from remote chiplet.
 * @param __dst_x Destination address in X-axis.
 * @param __dst_y Destination address in Y-axis.
 * @param __src_x Source address in X-axis.
 * @param __src_y Source address in Y-axis.
 * @param __addr Data address.
 * @param __nbyte Number of bytes.
 */
extern __host__ cudaError_t CUDARTAPI receiveMessage(int __dst_x, int __dst_y, int __src_x,
                                                     int __srx_y, void* __addr, int __nbyte);
