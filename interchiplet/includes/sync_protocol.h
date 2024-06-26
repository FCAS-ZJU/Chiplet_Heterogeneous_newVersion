
#pragma once

#include <fcntl.h>
#include <unistd.h>

#include <cstring>
#include <iostream>
#include <sstream>

#include "global_define.h"

#define NSINTERCHIPLET_CMD_HEAD "[INTERCMD]"

namespace InterChiplet {
/**
 * @defgroup sync_proto
 * @brief Synchronization protocol interface.
 * @{
 */
/**
 * @brief Return name of file name in a std::string.
 * Return directory related to the directory of main process.
 * @param __src Source address.
 * @param __dst Destiantion address.
 */
inline std::string pipeName(const AddrType& __src, const AddrType& __dst) {
    std::stringstream ss;
    ss << "./buffer" << DIM_X(__src) << "_" << DIM_Y(__src) << "_" << DIM_X(__dst) << "_"
       << DIM_Y(__dst);
    return ss.str();
}

/**
 * @brief Parse command from string.
 * @param __message String of message.
 * @return Structure of synchronization command.
 */
inline SyncCommand parseCmd(const std::string& __message) {
    // Remove command head.
    std::string message;
    if (__message.substr(0, 10) == NSINTERCHIPLET_CMD_HEAD) {
        message = __message.substr(11);
    } else {
        message = __message;
    }

    // Read command from message
    std::stringstream ss(message);
    std::string command;
    ss >> command;

    // Decode command to enumerate.
    SyncCommand cmd;
    cmd.m_type = command == "CYCLE"        ? SC_CYCLE
                 : command == "SEND"       ? SC_SEND
                 : command == "RECEIVE"    ? SC_RECEIVE
                 : command == "BARRIER"    ? SC_BARRIER
                 : command == "LOCK"       ? SC_LOCK
                 : command == "UNLOCK"     ? SC_UNLOCK
                 : command == "LAUNCH"     ? SC_LAUNCH
                 : command == "WAITLAUNCH" ? SC_WAITLAUNCH
                 : command == "READ"       ? SC_READ
                 : command == "WRITE"      ? SC_WRITE
                 : command == "SYNC"       ? SC_SYNC
                 : command == "RESULT"     ? SC_RESULT
                                           : SC_CYCLE;

    // Read cycle.
    if (cmd.m_type == SC_CYCLE || cmd.m_type == SC_READ || cmd.m_type == SC_WRITE ||
        cmd.m_type == SC_SYNC) {
        ss >> cmd.m_cycle;
    } else {
        cmd.m_cycle = 0;
    }
    // Read source address.
    if (cmd.m_type == SC_SEND || cmd.m_type == SC_RECEIVE || cmd.m_type == SC_BARRIER ||
        cmd.m_type == SC_LOCK || cmd.m_type == SC_UNLOCK || cmd.m_type == SC_LAUNCH ||
        cmd.m_type == SC_WAITLAUNCH || cmd.m_type == SC_READ || cmd.m_type == SC_WRITE) {
        long src_x, src_y;
        ss >> src_x >> src_y;
        cmd.m_src.push_back(src_x);
        cmd.m_src.push_back(src_y);
    } else {
        cmd.m_src.push_back(-1);
        cmd.m_src.push_back(-1);
    }
    // Read destination address.
    if (cmd.m_type == SC_SEND || cmd.m_type == SC_RECEIVE || cmd.m_type == SC_LAUNCH ||
        cmd.m_type == SC_WAITLAUNCH || cmd.m_type == SC_READ || cmd.m_type == SC_WRITE) {
        long dst_x, dst_y;
        ss >> dst_x >> dst_y;
        cmd.m_dst.push_back(dst_x);
        cmd.m_dst.push_back(dst_y);
    }
    // Read target address.
    else if (cmd.m_type == SC_BARRIER || cmd.m_type == SC_LOCK || cmd.m_type == SC_UNLOCK) {
        long dst_x;
        ss >> dst_x;
        cmd.m_dst.push_back(dst_x);
        cmd.m_dst.push_back(0);
    } else {
        cmd.m_dst.push_back(-1);
        cmd.m_dst.push_back(-1);
    }
    // Read number of bytes and descriptor.
    if (cmd.m_type == SC_READ || cmd.m_type == SC_WRITE) {
        ss >> cmd.m_nbytes >> cmd.m_desc;
    }
    // Read barrier count.
    else if (cmd.m_type == SC_BARRIER) {
        ss >> cmd.m_nbytes;
        cmd.m_desc = 0;
    } else {
        cmd.m_nbytes = 0;
        cmd.m_desc = 0;
    }

    // Read result
    if (cmd.m_type == SC_RESULT) {
        int res_cnt;
        ss >> res_cnt;
        for (int i = 0; i < res_cnt; i++) {
            std::string item;
            ss >> item;
            cmd.m_res_list.push_back(item);
        }
    } else {
        cmd.m_res_list.clear();
    }

    return cmd;
}

/**
 * @brief Receive command from stdin and parse the message. Used by simulator only.
 * @param __fd_in Input file descriptor.
 * @return Structure of synchronization command.
 */
inline SyncCommand parseCmd(int __fd_in = STDIN_FILENO) {
    // Read message from input file descriptor.
    char* message = new char[1024];
    while (read(__fd_in, message, 1024) == 0);

    // Split message from '\n', ignore characters after '\n'.
    for (std::size_t i = 0; i < strlen(message); i++) {
        if (message[i] == '\n') message[i + 1] = 0;
    }
    std::cout << "[RESPONSE]" << message;

    // Parse command.
    SyncCommand cmd = parseCmd(std::string(message));
    delete message;

    // Return message.
    return cmd;
}

/**
 * @brief Dump command to a string for debugging.
 * @param __cmd Structure of synchronization command.
 * @return String of message.
 */
inline std::string dumpCmd(const SyncCommand& __cmd) {
    std::stringstream ss;
    std::string type_str = __cmd.m_type == SC_CYCLE        ? "CYCLE"
                           : __cmd.m_type == SC_SEND       ? "SEND"
                           : __cmd.m_type == SC_RECEIVE    ? "RECEIVE"
                           : __cmd.m_type == SC_BARRIER    ? "BARRIER"
                           : __cmd.m_type == SC_LOCK       ? "LOCK"
                           : __cmd.m_type == SC_UNLOCK     ? "UNLOCK"
                           : __cmd.m_type == SC_LAUNCH     ? "LAUNCH"
                           : __cmd.m_type == SC_WAITLAUNCH ? "WAITLAUNCH"
                           : __cmd.m_type == SC_READ       ? "READ"
                           : __cmd.m_type == SC_WRITE      ? "WRITE"
                           : __cmd.m_type == SC_SYNC       ? "SYNC"
                           : __cmd.m_type == SC_RESULT     ? "RESULT"
                                                           : "CYCLE";
    ss << type_str << " command";

    // Write cycle
    if (__cmd.m_type == SC_CYCLE || __cmd.m_type == SC_READ || __cmd.m_type == SC_WRITE ||
        __cmd.m_type == SC_SYNC) {
        ss << " at " << static_cast<TimeType>(__cmd.m_cycle) << " cycle";
    }
    // Write source address.
    if (__cmd.m_type == SC_SEND || __cmd.m_type == SC_RECEIVE || __cmd.m_type == SC_BARRIER ||
        __cmd.m_type == SC_LOCK || __cmd.m_type == SC_UNLOCK || __cmd.m_type == SC_LAUNCH ||
        __cmd.m_type == SC_WAITLAUNCH || __cmd.m_type == SC_READ || __cmd.m_type == SC_WRITE) {
        ss << " from " << DIM_X(__cmd.m_src) << "," << DIM_Y(__cmd.m_src);
    }
    // Write destination address.
    if (__cmd.m_type == SC_SEND || __cmd.m_type == SC_RECEIVE || __cmd.m_type == SC_LAUNCH ||
        __cmd.m_type == SC_WAITLAUNCH || __cmd.m_type == SC_READ || __cmd.m_type == SC_WRITE) {
        ss << " to " << DIM_X(__cmd.m_dst) << "," << DIM_Y(__cmd.m_dst);
    }
    // Write target address.
    if (__cmd.m_type == SC_BARRIER || __cmd.m_type == SC_LOCK || __cmd.m_type == SC_UNLOCK) {
        ss << " to " << DIM_X(__cmd.m_dst);
    }
    // Write Result.
    if (__cmd.m_type == SC_RESULT) {
        ss << ":";
        for (auto& item : __cmd.m_res_list) {
            ss << " " << item;
        }
    }
    ss << ".";
    return ss.str();
}

/**
 * @brief Send CYCLE command.
 * @param __cycle Cycle to send CYCLE command.
 */
inline void sendCycleCmd(TimeType __cycle) {
    std::cout << NSINTERCHIPLET_CMD_HEAD << " CYCLE " << __cycle << std::endl;
}

/**
 * @brief Send SEND command.
 * @param __src_x Source address in X-axis.
 * @param __src_y Source address in Y-axis.
 * @param __dst_x Destiantion address in X-axis.
 * @param __dst_y Destination address in Y-axis.
 */
inline void sendSendCmd(int __src_x, int __src_y, int __dst_x, int __dst_y) {
    std::cout << NSINTERCHIPLET_CMD_HEAD << " SEND " << __src_x << " " << __src_y << " " << __dst_x
              << " " << __dst_y << std::endl;
}

/**
 * @brief Send RECEIVE command.
 * @param __src_x Source address in X-axis.
 * @param __src_y Source address in Y-axis.
 * @param __dst_x Destiantion address in X-axis.
 * @param __dst_y Destination address in Y-axis.
 */
inline void sendReceiveCmd(int __src_x, int __src_y, int __dst_x, int __dst_y) {
    std::cout << NSINTERCHIPLET_CMD_HEAD << " RECEIVE " << __src_x << " " << __src_y << " "
              << __dst_x << " " << __dst_y << std::endl;
}

/**
 * @brief Send BARRIER command.
 * @param __src_x Source address in X-axis.
 * @param __src_y Source address in Y-axis.
 * @param __uid Barrier ID.
 * @param __count Number of items in barrier.
 */
inline void sendBarrierCmd(int __src_x, int __src_y, int __uid, int __count) {
    std::cout << NSINTERCHIPLET_CMD_HEAD << " BARRIER " << __src_x << " " << __src_y << " " << __uid
              << " " << __count << std::endl;
}

/**
 * @brief Send LOCK command.
 * @param __src_x Source address in X-axis.
 * @param __src_y Source address in Y-axis.
 * @param __uid Barrier ID.
 */
inline void sendLockCmd(int __src_x, int __src_y, int __uid) {
    std::cout << NSINTERCHIPLET_CMD_HEAD << " LOCK " << __src_x << " " << __src_y << " " << __uid
              << std::endl;
}

/**
 * @brief Send UNLOCK command.
 * @param __src_x Source address in X-axis.
 * @param __src_y Source address in Y-axis.
 * @param __uid Barrier ID.
 */
inline void sendUnlockCmd(int __src_x, int __src_y, int __uid) {
    std::cout << NSINTERCHIPLET_CMD_HEAD << " UNLOCK " << __src_x << " " << __src_y << " " << __uid
              << std::endl;
}

/**
 * @brief Send LAUNCH command.
 * @param __src_x Source address in X-axis.
 * @param __src_y Source address in Y-axis.
 * @param __dst_x Destiantion address in X-axis.
 * @param __dst_y Destination address in Y-axis.
 */
inline void sendLaunchCmd(int __src_x, int __src_y, int __dst_x, int __dst_y) {
    std::cout << NSINTERCHIPLET_CMD_HEAD << " LAUNCH " << __src_x << " " << __src_y << " "
              << __dst_x << " " << __dst_y << std::endl;
}

/**
 * @brief Send WAITLAUNCH command.
 * @param __src_x Source address in X-axis.
 * @param __src_y Source address in Y-axis.
 * @param __dst_x Destiantion address in X-axis.
 * @param __dst_y Destination address in Y-axis.
 */
inline void sendWaitlaunchCmd(int __src_x, int __src_y, int __dst_x, int __dst_y) {
    std::cout << NSINTERCHIPLET_CMD_HEAD << " WAITLAUNCH " << __src_x << " " << __src_y << " "
              << __dst_x << " " << __dst_y << std::endl;
}

/**
 * @brief Send READ command.
 * @param __cycle Cycle to send READ command.
 * @param __src_x Source address in X-axis.
 * @param __src_y Source address in Y-axis.
 * @param __dst_x Destiantion address in X-axis.
 * @param __dst_y Destination address in Y-axis.
 * @param __nbyte Number of bytes to read.
 * @param __desc Synchronization protocol descriptor.
 */
inline void sendReadCmd(TimeType __cycle, int __src_x, int __src_y, int __dst_x, int __dst_y,
                        int __nbyte, long __desc) {
    std::cout << NSINTERCHIPLET_CMD_HEAD << " READ " << __cycle << " " << __src_x << " " << __src_y
              << " " << __dst_x << " " << __dst_y << " " << __nbyte << " " << __desc << std::endl;
}

/**
 * @brief Send WRITE command.
 * @param __cycle Cycle to send WRITE command.
 * @param __src_x Source address in X-axis.
 * @param __src_y Source address in Y-axis.
 * @param __dst_x Destiantion address in X-axis.
 * @param __dst_y Destination address in Y-axis.
 * @param __nbyte Number of bytes to write.
 * @param __desc Synchronization protocol descriptor.
 */
inline void sendWriteCmd(TimeType __cycle, int __src_x, int __src_y, int __dst_x, int __dst_y,
                         int __nbyte, long __desc) {
    std::cout << NSINTERCHIPLET_CMD_HEAD << " WRITE " << __cycle << " " << __src_x << " " << __src_y
              << " " << __dst_x << " " << __dst_y << " " << __nbyte << " " << __desc << std::endl;
}

/**
 * @brief Send SYNC command.
 * @param __cycle Cycle to receive SYNC command.
 */
inline void sendSyncCmd(TimeType __cycle) {
    std::cout << NSINTERCHIPLET_CMD_HEAD << " SYNC " << __cycle << std::endl;
}

/**
 * @brief Send SYNC command to specified file descriptor.
 * @param __fd File descriptor.
 * @param __cycle Cycle to receive SYNC command.
 */
inline void sendSyncCmd(int __fd, TimeType __cycle) {
    std::stringstream ss;
    ss << NSINTERCHIPLET_CMD_HEAD << " SYNC " << __cycle << std::endl;
    if (write(__fd, ss.str().c_str(), ss.str().size()) < 0) {
        perror("write");
        exit(EXIT_FAILURE);
    };
}

/**
 * @brief Send RESULT command.
 */
inline void sendResultCmd() {
    std::cout << NSINTERCHIPLET_CMD_HEAD << " RESULT " << 0 << std::endl;
}

/**
 * @brief Send RESULT command.
 * @param __res_list Result list.
 */
inline void sendResultCmd(const std::vector<std::string>& __res_list) {
    std::cout << NSINTERCHIPLET_CMD_HEAD << " RESULT " << __res_list.size();
    for (auto& item : __res_list) {
        std::cout << " " << item;
    }
    std::cout << std::endl;
}

/**
 * @brief Send RESULT command.
 * @param __res_list Result list.
 */
inline void sendResultCmd(const std::vector<long>& __res_list) {
    std::cout << NSINTERCHIPLET_CMD_HEAD << " RESULT " << __res_list.size();
    for (auto& item : __res_list) {
        std::cout << " " << item;
    }
    std::cout << std::endl;
}

/**
 * @brief Send RESULT command.
 * @param __fd File descriptor.
 */
inline void sendResultCmd(int __fd) {
    std::stringstream ss;
    ss << NSINTERCHIPLET_CMD_HEAD << " RESULT " << 0 << std::endl;
    if (write(__fd, ss.str().c_str(), ss.str().size()) < 0) {
        perror("write");
        exit(EXIT_FAILURE);
    };
}

/**
 * @brief Send RESULT command.
 * @param __fd File descriptor.
 * @param __res_list Result list.
 */
inline void sendResultCmd(int __fd, const std::vector<std::string>& __res_list) {
    std::stringstream ss;
    ss << NSINTERCHIPLET_CMD_HEAD << " RESULT " << __res_list.size();
    for (auto& item : __res_list) {
        ss << " " << item;
    }
    ss << std::endl;
    if (write(__fd, ss.str().c_str(), ss.str().size()) < 0) {
        perror("write");
        exit(EXIT_FAILURE);
    };
}

/**
 * @brief Send RESULT command.
 * @param __fd File descriptor.
 * @param __res_list Result list.
 */
inline void sendResultCmd(int __fd, const std::vector<long>& __res_list) {
    std::stringstream ss;
    ss << NSINTERCHIPLET_CMD_HEAD << " RESULT " << __res_list.size();
    for (auto& item : __res_list) {
        ss << " " << item;
    }
    ss << std::endl;
    if (write(__fd, ss.str().c_str(), ss.str().size()) < 0) {
        perror("write");
        exit(EXIT_FAILURE);
    };
}

/**
 * @brief Send CYCLE command and wait for SYNC command.
 * @param __cycle Cycle to send CYCLE command.
 * @return Cycle to receive SYNC command.
 */
inline TimeType cycleSync(TimeType __cycle) {
    // Send CYCLE command.
    sendCycleCmd(__cycle);
    // Read message from stdin.
    SyncCommand resp_cmd = parseCmd();
    // Only handle SYNC message, return cycle to receive SYNC command.
    return resp_cmd.m_type == SC_SYNC ? resp_cmd.m_cycle : -1;
}

/**
 * @brief Send SEND command and wait for SYNC command.
 * @param __src_x Source address in X-axis.
 * @param __src_y Source address in Y-axis.
 * @param __dst_x Destiantion address in X-axis.
 * @param __dst_y Destination address in Y-axis.
 * @return Pipe name.
 */
inline std::string sendSync(int __src_x, int __src_y, int __dst_x, int __dst_y) {
    // Send SEND command.
    sendSendCmd(__src_x, __src_y, __dst_x, __dst_y);
    // Read message from stdin.
    SyncCommand resp_cmd = parseCmd();
    // Return Pipe name.
    return resp_cmd.m_res_list[0];
}

/**
 * @brief Send RECEIVE command and wait for SYNC command.
 * @param __src_x Source address in X-axis.
 * @param __src_y Source address in Y-axis.
 * @param __dst_x Destiantion address in X-axis.
 * @param __dst_y Destination address in Y-axis.
 * @return Pipe name.
 */
inline std::string receiveSync(int __src_x, int __src_y, int __dst_x, int __dst_y) {
    // Send RECEIVE command.
    sendReceiveCmd(__src_x, __src_y, __dst_x, __dst_y);
    // Read message from stdin.
    SyncCommand resp_cmd = parseCmd();
    // Return Pipe name.
    return resp_cmd.m_res_list[0];
}

/**
 * @brief Send LAUNCH command and wait for SYNC command.
 * @param __src_x Source address in X-axis.
 * @param __src_y Source address in Y-axis.
 * @param __dst_x Destiantion address in X-axis.
 * @param __dst_y Destination address in Y-axis.
 */
inline void launchSync(int __src_x, int __src_y, int __dst_x, int __dst_y) {
    // Send LAUNCH command.
    sendLaunchCmd(__src_x, __src_y, __dst_x, __dst_y);
    // Read message from stdin.
    SyncCommand resp_cmd = parseCmd();

    return;
}

/**
 * @brief Send WAITLAUNCH command and wait for LAUNCH command.
 * @param __src_x Source address in X-axis.
 * @param __src_y Source address in Y-axis.
 * @param __dst_x Destiantion address in X-axis.
 * @param __dst_y Destination address in Y-axis.
 */
inline void waitlaunchSync(int* __src_x, int* __src_y, int __dst_x, int __dst_y) {
    // Send LAUNCH command.
    sendWaitlaunchCmd(*__src_x, *__src_y, __dst_x, __dst_y);
    // Read message from stdin.
    SyncCommand resp_cmd = parseCmd();
    *__src_x = atoi(resp_cmd.m_res_list[0].c_str());
    *__src_y = atoi(resp_cmd.m_res_list[1].c_str());

    return;
}

/**
 * @brief Send BARRIER command and wait for SYNC command.
 * @param __src_x Source address in X-axis.
 * @param __src_y Source address in Y-axis.
 * @param __uid Barrier ID.
 * @param __count Number of items in barrier.
 */
inline void barrierSync(int __src_x, int __src_y, int __uid, int __count) {
    // Send BARRIER command.
    sendBarrierCmd(__src_x, __src_y, __uid, __count);
    // Read message from stdin.
    SyncCommand resp_cmd = parseCmd();

    return;
}

/**
 * @brief Send UNLOCK command and wait for SYNC command.
 * @param __src_x Source address in X-axis.
 * @param __src_y Source address in Y-axis.
 * @param __uid Mutex ID.
 */
inline void lockSync(int __src_x, int __src_y, int __uid) {
    // Send UNLOCK command.
    sendLockCmd(__src_x, __src_y, __uid);
    // Read message from stdin.
    SyncCommand resp_cmd = parseCmd();

    return;
}

/**
 * @brief Send UNLOCK command and wait for SYNC command.
 * @param __src_x Source address in X-axis.
 * @param __src_y Source address in Y-axis.
 * @param __uid Mutex ID.
 */
inline void unlockSync(int __src_x, int __src_y, int __uid) {
    // Send UNLOCK command.
    sendUnlockCmd(__src_x, __src_y, __uid);
    // Read message from stdin.
    SyncCommand resp_cmd = parseCmd();

    return;
}

/**
 * @brief Send READ command and wait for SYNC command.
 * @param __cycle Cycle to send READ command.
 * @param __src_x Source address in X-axis.
 * @param __src_y Source address in Y-axis.
 * @param __dst_x Destiantion address in X-axis.
 * @param __dst_y Destination address in Y-axis.
 * @param __nbyte Number of bytes to read.
 * @param __desc Synchronization protocol descriptor.
 * @return Cycle to receive SYNC command.
 */
inline TimeType readSync(TimeType __cycle, int __src_x, int __src_y, int __dst_x, int __dst_y,
                         int __nbyte, long __desc) {
    // Send READ command.
    sendReadCmd(__cycle, __src_x, __src_y, __dst_x, __dst_y, __nbyte, __desc);
    // Read message from stdin.
    SyncCommand resp_cmd = parseCmd();
    // Only handle SYNC message, return cycle to receive SYNC command.
    return resp_cmd.m_type == SC_SYNC ? resp_cmd.m_cycle : -1;
}

/**
 * @brief Send WRITE command and wait for SYNC command.
 * @param __cycle Cycle to send WRITE command.
 * @param __src_x Source address in X-axis.
 * @param __src_y Source address in Y-axis.
 * @param __dst_x Destiantion address in X-axis.
 * @param __dst_y Destination address in Y-axis.
 * @param __nbyte Number of bytes to write.
 * @param __desc Synchronization protocol descriptor.
 * @return Cycle to receive SYNC command.
 */
inline TimeType writeSync(TimeType __cycle, int __src_x, int __src_y, int __dst_x, int __dst_y,
                          int __nbyte, long __desc) {
    // Send WRITE command.
    sendWriteCmd(__cycle, __src_x, __src_y, __dst_x, __dst_y, __nbyte, __desc);
    // Read message from stdin.
    SyncCommand resp_cmd = parseCmd();
    // Only handle SYNC message, return cycle to receive SYNC command.
    return resp_cmd.m_type == SC_SYNC ? resp_cmd.m_cycle : -1;
}
/**
 * @}
 */
}  // namespace InterChiplet
