#pragma once

#include <unistd.h>

#include <cstdint>

namespace InterChiplet {
typedef decltype(syscall(0)) syscall_return_t;

/**
 * @defgroup apis_for_cpu APIs for CPU
 * @brief APIs for CPU.
 * @{
 */
/**
 * @brief Launch application to remote chiplet.
 * @param __dst_x Destination address in X-axis.
 * @param __dst_y Destination address in Y-axis.
 * @param __src_x Source address in X-axis.
 * @param __src_y Source address in Y-axis.
 */
syscall_return_t launch(int64_t __dst_x, int64_t __dst_y, int64_t __src_x, int64_t __src_y);

/**
 * @brief Wait launch from remote chiplet.
 * @param __dst_x Destination address in X-axis.
 * @param __dst_y Destination address in Y-axis.
 * @param __src_x Source address in X-axis. Return value.
 * @param __src_y Source address in Y-axis. Return value.
 */
syscall_return_t waitLaunch(int64_t __dst_x, int64_t __dst_y, int64_t* __src_x, int64_t* __src_y);

/**
 * @brief Barrier.
 * @param __uid Barrier ID.
 * @param __src_x Source address in X-axis.
 * @param __src_y Source address in Y-axis.
 * @param __count Number of threads to barrier. 
 */
syscall_return_t barrier(int64_t __uid, int64_t __src_x, int64_t __src_y, int64_t __count = 0);

/**
 * @brief Lock mutex.
 * @param __uid Mutex ID.
 * @param __src_x Source address in X-axis.
 * @param __src_y Source address in Y-axis.
 */
syscall_return_t lock(int64_t __uid, int64_t __src_x, int64_t __src_y);

/**
 * @brief Unlock mutex.
 * @param __uid Mutex ID.
 * @param __src_x Source address in X-axis.
 * @param __src_y Source address in Y-axis.
 */
syscall_return_t unlock(int64_t __uid, int64_t __src_x, int64_t __src_y);

/**
 * @brief Send data to remote chiplet.
 * @param __dst_x Destination address in X-axis.
 * @param __dst_y Destination address in Y-axis.
 * @param __src_x Source address in X-axis.
 * @param __src_y Source address in Y-axis.
 * @param __addr Data address.
 * @param __nbyte Number of bytes.
 */
syscall_return_t sendMessage(int64_t __dst_x, int64_t __dst_y, int64_t __src_x, int64_t __src_y,
                             void* __addr, int64_t __nbyte);

/**
 * @brief Read data from remote chiplet.
 * @param __dst_x Destination address in X-axis.
 * @param __dst_y Destination address in Y-axis.
 * @param __src_x Source address in X-axis.
 * @param __src_y Source address in Y-axis.
 * @param __addr Data address.
 * @param __nbyte Number of bytes.
 */
syscall_return_t receiveMessage(int64_t __dst_x, int64_t __dst_y, int64_t __src_x, int64_t __src_y,
                                void* __addr, int64_t __nbyte);

/**
 * @}
 */
}  // namespace InterChiplet
