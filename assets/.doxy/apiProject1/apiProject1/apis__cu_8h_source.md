
# File apis\_cu.h

[**File List**](files.md) **>** [**includes**](dir_943fa6db2bfb09b7dcf1f02346dde40e.md) **>** [**apis\_cu.h**](apis__cu_8h.md)

[Go to the documentation of this file.](apis__cu_8h.md) 

```C++

#pragma once

#include "cuda_runtime_api.h"

extern __host__ cudaError_t CUDARTAPI launch(int __dst_x, int __dst_y, int __src_x, int __src_y);

extern __host__ cudaError_t CUDARTAPI waitLaunch(int __dst_x, int __dst_y, int* __src_x,
                                                 int* __src_y);

extern __host__ cudaError_t CUDARTAPI barrier(int __uid, int __src_x, int __src_y, int __count = 0);

extern __host__ cudaError_t CUDARTAPI lock(int __uid, int __src_x, int __src_y);

extern __host__ cudaError_t CUDARTAPI unlock(int __uid, int __src_x, int __src_y);

extern __host__ cudaError_t CUDARTAPI sendMessage(int __dst_x, int __dst_y, int __src_x,
                                                  int __srx_y, void* __addr, int __nbyte);

extern __host__ cudaError_t CUDARTAPI receiveMessage(int __dst_x, int __dst_y, int __src_x,
                                                     int __srx_y, void* __addr, int __nbyte);

```