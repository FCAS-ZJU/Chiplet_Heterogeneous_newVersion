#pragma once

#include <tuple>

#include "cuda_runtime_api.h"

extern __host__ cudaError_t CUDARTAPI barrier(int __uid, int __src_x, int __src_y, int __count = 0);

extern __host__ cudaError_t CUDARTAPI waitLocker(int __dst_x, int __dst_y, int& __src_x,
                                                 int& __src_y);

extern __host__ cudaError_t CUDARTAPI sendMessage(int __dst_x, int __dst_y, int __src_x,
                                                  int __srx_y, void* __addr, int __nbyte);

extern __host__ cudaError_t CUDARTAPI receiveMessage(int __dst_x, int __dst_y, int __src_x,
                                                     int __srx_y, void* __addr, int __nbyte);
