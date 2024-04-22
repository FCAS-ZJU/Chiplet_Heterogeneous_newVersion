#pragma once

#include <cstdint>
#include <unistd.h>

namespace InterChiplet
{
    typedef decltype(syscall(0)) syscall_return_t;

    // TODO: provide API for other data type.
    /**
     * @brief Send data from CPU to GPU.
     * @param __dst_x Destination address (GPU address) in X-axis.
     * @param __dst_y Destination address (GPU address) in Y-axis.
     * @param __src_x Source address (CPU address) in X-axis.
     * @param __src_y Source address (CPU address) in Y-axis.
     * @param __addr Data address.
     * @param __size Number of elements.
    */
    syscall_return_t sendGpuMessage(int64_t __dst_x,
                                    int64_t __dst_y,
                                    int64_t __src_x,
                                    int64_t __src_y,
                                    int64_t* __addr,
                                    int64_t __size);

    /**
     * @brief Read data from GPU to CPU.
     * @param __dst_x Destination address (CPU address) in X-axis.
     * @param __dst_y Destination address (CPU address) in Y-axis.
     * @param __src_x Source address (GPU address) in X-axis.
     * @param __src_y Source address (GPU address) in Y-axis.
     * @param __addr Data address.
     * @param __size Number of elements.
    */
    syscall_return_t readGpuMessage(int64_t __dst_x,
                                    int64_t __dst_y,
                                    int64_t __src_x,
                                    int64_t __src_y,
                                    int64_t* __addr,
                                    int64_t __size);
}