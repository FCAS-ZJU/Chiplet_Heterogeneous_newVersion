
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
 * @brief Interface for synchronize protocol.
 *
 * To address flexibile integration, the functions are included into one class so that it is not
 * necessary to add more source files into simulators.
 */
class SyncProtocol {
   public:
    /**
     * @brief Return name of file name in a array of character.
     * Return directory related to the directory of subprocess.
     * @param __src_x Source address in X-axis.
     * @param __src_y Source address in Y-axis.
     * @param __dst_x Destiantion address in X-axis.
     * @param __dst_y Destination address in Y-axis.
     */
    static char* pipeName(int __src_x, int __src_y, int __dst_x, int __dst_y) {
        char* fileName = new char[100];
        sprintf(fileName, "../buffer%d_%d_%d_%d", __src_x, __src_y, __dst_x, __dst_y);
        return fileName;
    }

    /**
     * @brief Return name of file name in a std::string.
     * Return directory related to the directory of main process.
     * @param __src_x Source address in X-axis.
     * @param __src_y Source address in Y-axis.
     * @param __dst_x Destiantion address in X-axis.
     * @param __dst_y Destination address in Y-axis.
     */
    static std::string pipeNameMaster(int __src_x, int __src_y, int __dst_x, int __dst_y) {
        std::stringstream ss;
        ss << "./buffer" << __src_x << "_" << __src_y << "_" << __dst_x << "_" << __dst_y;
        return ss.str();
    }

    /**
     * @brief Parse command from string.
     * @param __message String of message.
     * @return Structure of synchronization command.
     */
    static SyncCommand parseCmd(const std::string& __message) {
        // Remove command head.
        std::string message;
        if (__message.substr(0, 10) == NSINTERCHIPLET_CMD_HEAD) {
            message = __message.substr(11);
        } else {
            message = __message;
        }

        // Read cycle and command from message
        std::stringstream ss(message);
        std::string command;
        InnerTimeType cycle;
        ss >> command >> cycle;

        // Decode command to enumerate.
        SyncCommand cmd;
        cmd.m_cycle = cycle;
        cmd.m_type = command == "CYCLE"        ? SC_CYCLE
                     : command == "PIPE"       ? SC_PIPE
                     : command == "READ"       ? SC_READ
                     : command == "WRITE"      ? SC_WRITE
                     : command == "BARRIER"    ? SC_BARRIER
                     : command == "LAUNCH"     ? SC_LAUNCH
                     : command == "UNLOCK"     ? SC_UNLOCK
                     : command == "WAITLAUNCH" ? SC_WAITLAUNCH
                     : command == "SYNC"       ? SC_SYNC
                                               : SC_CYCLE;

        // Read source/destination address.
        if (cmd.m_type == SC_PIPE || cmd.m_type == SC_LAUNCH || cmd.m_type == SC_UNLOCK ||
            cmd.m_type == SC_WAITLAUNCH || cmd.m_type == SC_READ || cmd.m_type == SC_WRITE) {
            ss >> cmd.m_src_x >> cmd.m_src_y >> cmd.m_dst_x >> cmd.m_dst_y;
        }
        // Read number of bytes.
        if (cmd.m_type == SC_READ || cmd.m_type == SC_WRITE) {
            ss >> cmd.m_nbytes >> cmd.m_desc;
        }
        // Read barrier.
        if (cmd.m_type == SC_BARRIER) {
            ss >> cmd.m_src_x >> cmd.m_src_y >> cmd.m_dst_x >> cmd.m_nbytes;
        }

        return cmd;
    }

    /**
     * @brief Receive command from stdin and parse the message. Used by simulator only.
     * @param __fd_in Input file descriptor.
     * @return Structure of synchronization command.
     */
    static SyncCommand parseCmd(int __fd_in = STDIN_FILENO) {
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

   public:
    /**
     * @brief Send CYCLE command.
     * @param __cycle Cycle to send CYCLE command.
     */
    static void sendCycleCmd(TimeType __cycle) {
        std::cout << NSINTERCHIPLET_CMD_HEAD << " CYCLE " << __cycle << std::endl;
    }

    /**
     * @brief Send BARRIER command.
     * @param __src_x Source address in X-axis.
     * @param __src_y Source address in Y-axis.
     * @param __uid Barrier ID.
     * @param __count Number of items in barrier.
     */
    static void sendBarrierCmd(int __src_x, int __src_y, int __uid, int __count) {
        std::cout << NSINTERCHIPLET_CMD_HEAD << " BARRIER " << 0 << " " << __src_x << " " << __src_y
                  << " " << __uid << " " << __count << std::endl;
    }

    /**
     * @brief Send PIPE command. Cycle is ignored and treated as 0.
     * @param __src_x Source address in X-axis.
     * @param __src_y Source address in Y-axis.
     * @param __dst_x Destiantion address in X-axis.
     * @param __dst_y Destination address in Y-axis.
     */
    static void sendPipeCmd(int __src_x, int __src_y, int __dst_x, int __dst_y) {
        std::cout << NSINTERCHIPLET_CMD_HEAD << " PIPE " << 0 << " " << __src_x << " " << __src_y
                  << " " << __dst_x << " " << __dst_y << std::endl;
    }

    /**
     * @brief Send LAUNCH command. Cycle is ignored and treated as 0.
     * @param __src_x Source address in X-axis.
     * @param __src_y Source address in Y-axis.
     * @param __dst_x Destiantion address in X-axis.
     * @param __dst_y Destination address in Y-axis.
     */
    static void sendLaunchCmd(int __src_x, int __src_y, int __dst_x, int __dst_y) {
        std::cout << NSINTERCHIPLET_CMD_HEAD << " LAUNCH " << 0 << " " << __src_x << " " << __src_y
                  << " " << __dst_x << " " << __dst_y << std::endl;
    }

    /**
     * @brief Send LAUNCH command. Cycle is ignored and treated as 0.
     * @param __fd File descriptor.
     * @param __src_x Source address in X-axis.
     * @param __src_y Source address in Y-axis.
     * @param __dst_x Destiantion address in X-axis.
     * @param __dst_y Destination address in Y-axis.
     */
    static void sendLaunchCmd(int __fd, int __src_x, int __src_y, int __dst_x, int __dst_y) {
        std::stringstream ss;
        ss << NSINTERCHIPLET_CMD_HEAD << " LAUNCH " << 0 << " " << __src_x << " " << __src_y << " "
           << __dst_x << " " << __dst_y << std::endl;
        if (write(__fd, ss.str().c_str(), ss.str().size()) < 0) {
            perror("write");
            exit(EXIT_FAILURE);
        };
    }

    /**
     * @brief Send UNLOCK command. Cycle is ignored and treated as 0.
     * @param __src_x Source address in X-axis.
     * @param __src_y Source address in Y-axis.
     * @param __dst_x Destiantion address in X-axis.
     * @param __dst_y Destination address in Y-axis.
     */
    static void sendUnlockCmd(int __src_x, int __src_y, int __dst_x, int __dst_y) {
        std::cout << NSINTERCHIPLET_CMD_HEAD << " UNLOCK " << 0 << " " << __src_x << " " << __src_y
                  << " " << __dst_x << " " << __dst_y << std::endl;
    }

    /**
     * @brief Send WAITLAUNCH command.
     * @param __src_x Source address in X-axis.
     * @param __src_y Source address in Y-axis.
     * @param __dst_x Destiantion address in X-axis.
     * @param __dst_y Destination address in Y-axis.
     */
    static void sendWaitlaunchCmd(int __src_x, int __src_y, int __dst_x, int __dst_y) {
        std::cout << NSINTERCHIPLET_CMD_HEAD << " WAITLAUNCH " << 0 << " " << __src_x << " "
                  << __src_y << " " << __dst_x << " " << __dst_y << std::endl;
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
    static void sendReadCmd(TimeType __cycle, int __src_x, int __src_y, int __dst_x, int __dst_y,
                            int __nbyte, long __desc) {
        std::cout << NSINTERCHIPLET_CMD_HEAD << " READ " << __cycle << " " << __src_x << " "
                  << __src_y << " " << __dst_x << " " << __dst_y << " " << __nbyte << " " << __desc
                  << std::endl;
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
    static void sendWriteCmd(TimeType __cycle, int __src_x, int __src_y, int __dst_x, int __dst_y,
                             int __nbyte, long __desc) {
        std::cout << NSINTERCHIPLET_CMD_HEAD << " WRITE " << __cycle << " " << __src_x << " "
                  << __src_y << " " << __dst_x << " " << __dst_y << " " << __nbyte << " " << __desc
                  << std::endl;
    }

    /**
     * @brief Send SYNC command.
     * @param __cycle Cycle to receive SYNC command.
     */
    static void sendSyncCmd(TimeType __cycle) {
        std::cout << NSINTERCHIPLET_CMD_HEAD << " SYNC " << __cycle << std::endl;
    }

    /**
     * @brief Send SYNC command to specified file descriptor.
     * @param __fd File descriptor.
     * @param __cycle Cycle to receive SYNC command.
     */
    static void sendSyncCmd(int __fd, TimeType __cycle) {
        std::stringstream ss;
        ss << NSINTERCHIPLET_CMD_HEAD << " SYNC " << __cycle << std::endl;
        if (write(__fd, ss.str().c_str(), ss.str().size()) < 0) {
            perror("write");
            exit(EXIT_FAILURE);
        };
    }

   public:
    /**
     * @brief Send CYCLE command and wait for SYNC command.
     * @param __cycle Cycle to send CYCLE command.
     * @return Cycle to receive SYNC command.
     */
    static TimeType cycleSync(TimeType __cycle) {
        // Send CYCLE command.
        sendCycleCmd(__cycle);
        // Read message from stdin.
        SyncCommand resp_cmd = parseCmd();
        // Only handle SYNC message, return cycle to receive SYNC command.
        return resp_cmd.m_type == SC_SYNC ? resp_cmd.m_cycle : -1;
    }

    /**
     * @brief Send PIPE command and wait for SYNC command.
     * @param __src_x Source address in X-axis.
     * @param __src_y Source address in Y-axis.
     * @param __dst_x Destiantion address in X-axis.
     * @param __dst_y Destination address in Y-axis.
     * @return Cycle to receive SYNC command.
     */
    static TimeType pipeSync(int __src_x, int __src_y, int __dst_x, int __dst_y) {
        // Send PIPE command.
        sendPipeCmd(__src_x, __src_y, __dst_x, __dst_y);
        // Read message from stdin.
        SyncCommand resp_cmd = parseCmd();
        // Only handle SYNC message, return cycle to receive SYNC command.
        return resp_cmd.m_type == SC_SYNC ? resp_cmd.m_cycle : -1;
    }

    /**
     * @brief Send LAUNCH command and wait for SYNC command.
     * @param __src_x Source address in X-axis.
     * @param __src_y Source address in Y-axis.
     * @param __dst_x Destiantion address in X-axis.
     * @param __dst_y Destination address in Y-axis.
     * @return Cycle to receive SYNC command.
     */
    static TimeType launchSync(int __src_x, int __src_y, int __dst_x, int __dst_y) {
        // Send LAUNCH command.
        sendLaunchCmd(__src_x, __src_y, __dst_x, __dst_y);
        // Read message from stdin.
        SyncCommand resp_cmd = parseCmd();
        // Only handle SYNC message, return cycle to receive SYNC command.
        return resp_cmd.m_type == SC_SYNC ? resp_cmd.m_cycle : -1;
    }

    /**
     * @brief Send UNLOCK command and wait for SYNC command.
     * @param __src_x Source address in X-axis.
     * @param __src_y Source address in Y-axis.
     * @param __dst_x Destiantion address in X-axis.
     * @param __dst_y Destination address in Y-axis.
     * @return Cycle to receive SYNC command.
     */
    static TimeType unlockSync(int __src_x, int __src_y, int __dst_x, int __dst_y) {
        // Send UNLOCK command.
        sendUnlockCmd(__src_x, __src_y, __dst_x, __dst_y);
        // Read message from stdin.
        SyncCommand resp_cmd = parseCmd();
        // Only handle SYNC message, return cycle to receive SYNC command.
        return resp_cmd.m_type == SC_SYNC ? resp_cmd.m_cycle : -1;
    }

    /**
     * @brief Send WAITLAUNCH command and wait for LAUNCH command.
     * @param __src_x Source address in X-axis.
     * @param __src_y Source address in Y-axis.
     * @param __dst_x Destiantion address in X-axis.
     * @param __dst_y Destination address in Y-axis.
     * @return Cycle to receive SYNC command.
     */
    static TimeType waitlaunchSync(int* __src_x, int* __src_y, int __dst_x, int __dst_y) {
        // Send LAUNCH command.
        sendWaitlaunchCmd(*__src_x, *__src_y, __dst_x, __dst_y);
        // Read message from stdin.
        SyncCommand resp_cmd = parseCmd();
        *__src_x = resp_cmd.m_src_x;
        *__src_y = resp_cmd.m_src_y;
        // Only handle SYNC message, return cycle to receive SYNC command.
        return resp_cmd.m_type == SC_LAUNCH ? resp_cmd.m_cycle : -1;
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
    static TimeType readSync(TimeType __cycle, int __src_x, int __src_y, int __dst_x, int __dst_y,
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
    static TimeType writeSync(TimeType __cycle, int __src_x, int __src_y, int __dst_x, int __dst_y,
                              int __nbyte, long __desc) {
        // Send WRITE command.
        sendWriteCmd(__cycle, __src_x, __src_y, __dst_x, __dst_y, __nbyte, __desc);
        // Read message from stdin.
        SyncCommand resp_cmd = parseCmd();
        // Only handle SYNC message, return cycle to receive SYNC command.
        return resp_cmd.m_type == SC_SYNC ? resp_cmd.m_cycle : -1;
    }

    /**
     * @brief Send BARRIER command and wait for SYNC command.
     * @param __src_x Source address in X-axis.
     * @param __src_y Source address in Y-axis.
     * @param __uid Barrier ID.
     * @param __count Number of items in barrier.
     * @return Cycle to receive SYNC command.
     */
    static TimeType barrierSync(int __src_x, int __src_y, int __uid, int __count) {
        // Send BARRIER command.
        sendBarrierCmd(__src_x, __src_y, __uid, __count);
        // Read message from stdin.
        SyncCommand resp_cmd = parseCmd();
        // Only handle SYNC message, return cycle to receive SYNC command.
        return resp_cmd.m_type == SC_SYNC ? resp_cmd.m_cycle : -1;
    }
};
}  // namespace InterChiplet
