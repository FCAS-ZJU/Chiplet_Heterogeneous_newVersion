#pragma once

#include <cstdint>

namespace InterChiplet
{
    /**
     * @brief Syscall ID used in CPU/GPU.
     */
    enum SysCallID
    {
        SYSCALL_TEST_CHANGE = 500,          // Test
        SYSCALL_REMOTE_READ = 501,          // Read cross chiplet
        SYSCALL_REMOTE_WRITE = 502,         // Write cross chiplet
        SYSCALL_REG_FUNC = 503,             // Send register function to pin (depreciate)
        SYSCALL_CONNECT = 504,              // Setup connection.
        SYSCALL_DISCONNECT = 505,           // Stop connection.
        SYSCALL_GET_LOCAL_ADDR = 506,       // Get address of current processor.
        SYSCALL_CHECK_REMOTE_READ = 507,    // Check remote read
        SYSCALL_SEND_TO_GPU = 508,          // Send data to GPU
        SYSCALL_READ_FROM_GPU = 509,        // Read data from CPU
    };

    /**
     * @brief Command type of CUDA API syscall.
     */
    enum CudaSysCallType
    {
        CUDA_SYSCALL_ARG = 0,   // Add argument of Syscall.
        CUDA_SYSCALL_CMD = 1,   // Send command of Syscall and trigger syscall execution.
    };

    /**
     * Time type used between simulators.
     */
    typedef unsigned long long TimeType;

    /**
     * @brief Type of synchronization command between simulators.
     */
    enum SyncCommType
    {
        SC_CYCLE,
        SC_PIPE,
        SC_READ,
        SC_WRITE,
        SC_BARRIER,
        SC_LOCK,
        SC_UNLOCK,
        SC_SYNC,
        SC_RESULT,
    };

    /**
     * @brief Structure of synchronization command.
     */
    class SyncCommand
    {
    public:
        /**
         * @brief Type of synchronization command.
         */
        SyncCommType m_type;
        /**
         * @brief Cycle to send/receive command.
         */
        TimeType m_cycle;
        /**
         * @brief Source address in X-axis.
         */
        int m_src_x;
        /**
         * @brief Source address in Y-axis.
         */
        int m_src_y;
        /**
         * @brief Destiantion address in X-axis.
         */
        int m_dst_x;
        /**
         * @brief Destination address in Y-axis.
         */
        int m_dst_y;
        /**
         * @brief Number of bytes to write.
         */
        int m_nbytes;

        /**
         * @brief File descriptor to write response of this command.
         * 
         * For example, if one entity presents READ command, the SYNC command to response this READ
         * command should to send to this file descriptor.
         */
        int m_stdin_fd;
    };
}
