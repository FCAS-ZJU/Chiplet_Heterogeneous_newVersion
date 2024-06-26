#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace InterChiplet {
/**
 * @brief Syscall ID used in CPU/GPU.
 */
enum SysCallID {
    // SYSCALL_TEST_CHANGE = 500,        // Test
    // SYSCALL_REG_FUNC = 503,           // Send register function to pin (depreciate)
    // SYSCALL_CONNECT = 504,            // Setup connection.
    // SYSCALL_DISCONNECT = 505,         // Stop connection.
    // SYSCALL_GET_LOCAL_ADDR = 506,     // Get address of current processor.
    // SYSCALL_CHECK_REMOTE_READ = 507,  // Check remote read

    SYSCALL_LAUNCH = 501,        // Launch request.
    SYSCALL_WAITLAUNCH = 502,    // Waiit launch request.
    SYSCALL_BARRIER = 503,       // Enter barrier.
    SYSCALL_LOCK = 504,          // Lock mutex.
    SYSCALL_UNLOCK = 505,        // Unlock mutex.
    SYSCALL_REMOTE_READ = 506,   // Read cross chiplet
    SYSCALL_REMOTE_WRITE = 507,  // Write cross chiplet
};

/**
 * @brief Time type used between simulators.
 */
typedef unsigned long long TimeType;

/**
 * @brief Time type used by interchiplet module.
 */
typedef double InnerTimeType;

/**
 * @brief Address type;
 */
typedef std::vector<long> AddrType;

#define DIM_X(addr) (addr[0])
#define DIM_Y(addr) (addr[1])
#define UNSPECIFIED_ADDR(addr) ((addr[0]) < 0 && (addr[1]) < 0)

/**
 * @brief Type of synchronization command between simulators.
 */
enum SyncCommType {
    SC_CYCLE,
    SC_SEND,
    SC_RECEIVE,
    SC_BARRIER,
    SC_LOCK,
    SC_UNLOCK,
    SC_LAUNCH,
    SC_WAITLAUNCH,
    SC_READ,
    SC_WRITE,
    SC_SYNC,
    SC_RESULT,
};

/**
 * @brief Behavior descriptor of synchronization protocol.
 */
enum SyncProtocolDesc {
    /**
     * @brief Acknowledge. bit 0.
     */
    SPD_ACK = 0x01,
    /**
     * @brief Synchronization before data transmission. bit 1.
     */
    SPD_PRE_SYNC = 0x02,
    /**
     * @brief Synchronization after data transmission. bit 2.
     */
    SPD_POST_SYNC = 0x04,
    /**
     * @brief Launch behavior. bit 16.
     */
    SPD_LAUNCH = 0x10000,
    /**
     * @brief Barrier behavior. bit 17.
     */
    SPD_BARRIER = 0x20000,
    /**
     * @brief Lock behavior. bit 18.
     */
    SPD_LOCK = 0x40000,
    /**
     * @brief Lock behavior. bit 19.
     */
    SPD_UNLOCK = 0x80000,
};

/**
 * @brief Structure of synchronization command.
 */
class SyncCommand {
   public:
    /**
     * @brief Type of synchronization command.
     */
    SyncCommType m_type;
    /**
     * @brief Cycle to send/receive command.
     */
    InnerTimeType m_cycle;
    /**
     * @brief Cycle convert rate.
     */
    double m_clock_rate;
    /**
     * @brief Source address.
     */
    AddrType m_src;
    /**
     * @brief Destiantion address in X-axis.
     */
    AddrType m_dst;
    /**
     * @brief Number of bytes to write.
     */
    int m_nbytes;
    /**
     * @brief Descriptor of synchronization behavior.
     */
    long m_desc;

    /**
     * @brief List of result strings.
     */
    std::vector<std::string> m_res_list;

    /**
     * @brief File descriptor to write response of this command.
     *
     * For example, if one entity presents READ command, the SYNC command to response this READ
     * command should to send to this file descriptor.
     */
    int m_stdin_fd;
};
}  // namespace InterChiplet
