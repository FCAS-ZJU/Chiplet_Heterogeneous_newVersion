
# File global\_define.h

[**File List**](files.md) **>** [**includes**](dir_943fa6db2bfb09b7dcf1f02346dde40e.md) **>** [**global\_define.h**](global__define_8h.md)

[Go to the documentation of this file.](global__define_8h.md) 

```C++

#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace InterChiplet {
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

typedef unsigned long long TimeType;

typedef double InnerTimeType;

typedef std::vector<long> AddrType;

#define DIM_X(addr) (addr[0])
#define DIM_Y(addr) (addr[1])
#define UNSPECIFIED_ADDR(addr) ((addr[0]) < 0 && (addr[1]) < 0)

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

enum SyncProtocolDesc {
    SPD_ACK = 0x01,
    SPD_PRE_SYNC = 0x02,
    SPD_POST_SYNC = 0x04,
    SPD_LAUNCH = 0x10000,
    SPD_BARRIER = 0x20000,
    SPD_LOCK = 0x40000,
    SPD_UNLOCK = 0x80000,
};

class SyncCommand {
   public:
    SyncCommType m_type;
    InnerTimeType m_cycle;
    double m_clock_rate;
    AddrType m_src;
    AddrType m_dst;
    int m_nbytes;
    long m_desc;

    std::vector<std::string> m_res_list;

    int m_stdin_fd;
};

typedef std::vector<SyncCommand> SyncCmdList;
}  // namespace InterChiplet

```