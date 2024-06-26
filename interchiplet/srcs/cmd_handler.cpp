
#include "cmd_handler.h"

#include <sstream>
#include <string>

#include "spdlog/spdlog.h"

void handle_cycle_cmd(const InterChiplet::SyncCommand& __cmd, SyncStruct* __sync_struct) {
    // Update global cycle.
    __sync_struct->m_cycle_struct.update(__cmd.m_cycle);
    spdlog::debug("{}", dumpCmd(__cmd));
}

void handle_pipe_cmd(const InterChiplet::SyncCommand& __cmd, SyncStruct* __sync_struct) {
    // Create Pipe file.
    std::string file_name = InterChiplet::pipeName(__cmd.m_src, __cmd.m_dst);
    if (access(file_name.c_str(), F_OK) == -1) {
        // Report error if FIFO file does not exist and mkfifo error.
        if (mkfifo(file_name.c_str(), 0664) == -1) {
            spdlog::error("{} Cannot create pipe file {}.", dumpCmd(__cmd), file_name);
        }
        // Report success.
        else {
            // Register file name in pipe set.
            __sync_struct->m_pipe_struct.insert(file_name);
            spdlog::debug("{} Create pipe file {}.", dumpCmd(__cmd), file_name);
        }
    }
    // Reuse exist FIFO and reports.
    else {
        spdlog::debug("{} Reuse exist pipe file {}.", dumpCmd(__cmd), file_name);
    }

    // Send RESULT command.
    std::string resp_file_name = "../" + file_name;
    InterChiplet::sendResultCmd(__cmd.m_stdin_fd, {resp_file_name});
}

void handle_barrier_cmd(const InterChiplet::SyncCommand& __cmd, SyncStruct* __sync_struct) {
    int uid = DIM_X(__cmd.m_dst);
    int count = __cmd.m_nbytes;

    // Register BARRIER command.
    __sync_struct->m_barrier_struct.insertBarrier(uid, count, __cmd);
    // Barrier override.
    if (__sync_struct->m_barrier_struct.overflow(uid)) {
        // Send RESULT command to each barrier.
        for (auto& item : __sync_struct->m_barrier_struct.barrierCmd(uid)) {
            InterChiplet::sendResultCmd(item.m_stdin_fd);
        }
        __sync_struct->m_barrier_struct.reset(uid);
        spdlog::debug("{} Register BARRIER command. Barrier overflow.", dumpCmd(__cmd));
    }
    // Wait further barrier commands.
    else {
        spdlog::debug("{} Register BARRIER command.", dumpCmd(__cmd));
    }
}

void handle_lock_cmd(const InterChiplet::SyncCommand& __cmd, SyncStruct* __sync_struct) {
    // Get mutex ID.
    int uid = DIM_X(__cmd.m_dst);

    if (__sync_struct->m_lock_struct.isLocked(uid)) {
        // If the mutex is locked, check whether the LOCK command comes from the same source.
        if (__sync_struct->m_lock_struct.hasLastCmd(uid) &&
            (__sync_struct->m_lock_struct.getLastCmd(uid).m_src == __cmd.m_src)) {
            // If the LOCK command comes from the same source, ignore the LOCK command.
            spdlog::debug("{} Mutex {} has been locked by the same source. Do nothing.",
                          dumpCmd(__cmd), uid);
            // Send RESULT command to response UNLOCK command.
            InterChiplet::sendResultCmd(__cmd.m_stdin_fd);
        } else {
            // Otherwise, pending the LOCK command until the mutex is unlocked.
            __sync_struct->m_lock_struct.insertLockCmd(uid, __cmd);
            spdlog::debug("{} Register LOCK command to wait UNLOCK command.", dumpCmd(__cmd));
        }
    } else {
        // If the mutex is not locked, whether this LOCK command can be response dependes on the
        // order determined by the delay information.
        if (__sync_struct->m_delay_list.hasLock(__cmd.m_dst)) {
            // If the order of the mutex is determined by the delay information, check whether
            // the source address of this LOCK command matches the order determined by the delay
            // information.
            if (__sync_struct->m_delay_list.frontLockSrc(__cmd.m_dst) == __cmd.m_src) {
                // If the source address of this LOCK command matches the order determined by the
                // delay information, lock the mutex.
                __sync_struct->m_lock_struct.lock(uid, __cmd);
                spdlog::debug("{} Lock mutex {}.", dumpCmd(__cmd), uid);
                // Send RESULT command to response the LOCK command.
                InterChiplet::sendResultCmd(__cmd.m_stdin_fd);
                // Pop one item from delay information.
                __sync_struct->m_delay_list.popLock(__cmd.m_dst);
            } else {
                // If the source address of this LOCK command does not match the order determined
                // by the delay informtion, pending this LOCK command until the correct order.
                __sync_struct->m_lock_struct.insertLockCmd(uid, __cmd);
                spdlog::debug("{} Register LOCK command to wait correct order.", dumpCmd(__cmd));
            }
        } else {
            // If the order of the mutex is not determined by the delay information, lock the mutex.
            __sync_struct->m_lock_struct.lock(uid, __cmd);
            spdlog::debug("{} Lock mutex {}.", dumpCmd(__cmd), uid);
            // Send RESULT command to response the LOCK command.
            InterChiplet::sendResultCmd(__cmd.m_stdin_fd);
        }
    }
}

void handle_unlock_cmd(const InterChiplet::SyncCommand& __cmd, SyncStruct* __sync_struct) {
    // Get mutex ID.
    int uid = DIM_X(__cmd.m_dst);

    if (__sync_struct->m_lock_struct.isLocked(uid)) {
        // If the mutex is locked, unlock the mutex first.
        __sync_struct->m_lock_struct.unlock(uid, __cmd);
        spdlog::debug("{} Unlock mutex {}.", dumpCmd(__cmd), uid);
        // Send RESULT command to response UNLOCK command.
        InterChiplet::sendResultCmd(__cmd.m_stdin_fd);

        // Lock the mutex by pending LOCK command.
        if (__sync_struct->m_delay_list.hasLock(__cmd.m_dst)) {
            // If the order of the mutex is determined by the delay information, check whether
            // there is any pending LOCK command matching the order determined by the delay
            // information.
            InterChiplet::AddrType order_src =
                __sync_struct->m_delay_list.frontLockSrc(__cmd.m_dst);
            if (__sync_struct->m_lock_struct.hasLockCmd(uid, order_src)) {
                // If there is one pending LOCK command matching the order determined by the delay
                // information. Lock the mutex by this pending LOCK command.
                InterChiplet::SyncCommand lock_cmd =
                    __sync_struct->m_lock_struct.popLockCmd(uid, order_src);
                __sync_struct->m_lock_struct.lock(uid, lock_cmd);
                spdlog::debug("\t{} Lock mutex {}.", dumpCmd(lock_cmd), uid);
                // Send RESULT command to response the pending LOCK command.
                InterChiplet::sendResultCmd(lock_cmd.m_stdin_fd);
                // Pop one item from the delay information.
                __sync_struct->m_delay_list.popLock(__cmd.m_dst);
            }
        } else {
            // If the order of the mutex is not determined by the delay information, check whether
            // there is any pending LOCK command.
            if (__sync_struct->m_lock_struct.hasLockCmd(uid)) {
                // If there is any pending LOCK command, lock the mutex by the first pending LOCK
                // command.
                InterChiplet::SyncCommand lock_cmd = __sync_struct->m_lock_struct.popLockCmd(uid);
                __sync_struct->m_lock_struct.lock(uid, lock_cmd);
                spdlog::debug("\t{} Lock mutex {}.", dumpCmd(lock_cmd), uid);
                // Send RESULT command to response the pending LOCK command.
                InterChiplet::sendResultCmd(lock_cmd.m_stdin_fd);
            }
        }
    } else {
        // If the mutex is unlocked, ignore the UNLOCK command.
        spdlog::debug("{} Mutex {} has not locked. Do nothing.", dumpCmd(__cmd), uid);
        // Send RESULT command to response UNLOCK command.
        InterChiplet::sendResultCmd(__cmd.m_stdin_fd);
    }
}

void handle_launch_cmd(const InterChiplet::SyncCommand& __cmd, SyncStruct* __sync_struct) {
    // Check launch order and remove item.
    if (__sync_struct->m_delay_list.hasLaunch(__cmd.m_dst)) {
        if (__sync_struct->m_delay_list.frontLaunchSrc(__cmd.m_dst) != __cmd.m_src) {
            __sync_struct->m_launch_struct.insertLaunch(__cmd);
            spdlog::debug("{} Register LAUNCH command to pair with WAITLAUNCH command.",
                          dumpCmd(__cmd));
            return;
        }
    }

    // Check for unconfirmed waitlaunch command.
    bool has_waitlaunch_cmd = __sync_struct->m_launch_struct.hasMatchWaitlaunch(__cmd);
    InterChiplet::SyncCommand waitlaunch_cmd =
        __sync_struct->m_launch_struct.popMatchWaitlaunch(__cmd);

    // If there is not waitlaunch command, waitlaunch command.
    if (!has_waitlaunch_cmd) {
        __sync_struct->m_launch_struct.insertLaunch(__cmd);
        spdlog::debug("{} Register LAUNCH command to pair with WAITLAUNCH command.",
                      dumpCmd(__cmd));
    }
    // If there is waitlaunch command, response launch and waitlaunch command.
    else {
        __sync_struct->m_delay_list.popLaunch(__cmd.m_dst);
        spdlog::debug("{} Pair with WAITLAUNCH command.", dumpCmd(__cmd));
        // Send SYNC to response LAUNCH command.
        InterChiplet::sendResultCmd(__cmd.m_stdin_fd);
        // Send LAUNCH to response WAITLAUNCH command.
        InterChiplet::sendResultCmd(waitlaunch_cmd.m_stdin_fd,
                                    {DIM_X(__cmd.m_src), DIM_Y(__cmd.m_src)});
    }
}

void handle_waitlaunch_cmd(const InterChiplet::SyncCommand& __cmd, SyncStruct* __sync_struct) {
    InterChiplet::SyncCommand cmd = __cmd;

    // Check launch order and remove item.
    if (__sync_struct->m_delay_list.hasLaunch(cmd.m_dst)) {
        cmd.m_src = __sync_struct->m_delay_list.frontLaunchSrc(cmd.m_dst);
    }

    // Check for unconfirmed waitlaunch command.
    bool has_launch_cmd = __sync_struct->m_launch_struct.hasMatchLaunch(cmd);
    InterChiplet::SyncCommand launch_cmd = __sync_struct->m_launch_struct.popMatchLaunch(cmd);

    // If there is not waitlaunch command, waitlaunch command.
    if (!has_launch_cmd) {
        __sync_struct->m_launch_struct.insertWaitlaunch(__cmd);
        spdlog::debug("{} Register WAITLAUNCH command to pair with LAUNCH command.", dumpCmd(cmd));
    } else {
        __sync_struct->m_delay_list.popLaunch(cmd.m_dst);
        spdlog::debug("{} Pair with LAUNCH command from {},{} to {},{}.", dumpCmd(cmd),
                      DIM_X(launch_cmd.m_src), DIM_Y(launch_cmd.m_src), DIM_X(launch_cmd.m_dst),
                      DIM_Y(launch_cmd.m_dst));
        // Send RESULT to response LAUNCH command.
        InterChiplet::sendResultCmd(launch_cmd.m_stdin_fd);
        // Send RESULT to response WAITLAUNCH command.
        InterChiplet::sendResultCmd(cmd.m_stdin_fd,
                                    {DIM_X(launch_cmd.m_src), DIM_Y(launch_cmd.m_src)});
    }
}

void handle_read_cmd(const InterChiplet::SyncCommand& __cmd, SyncStruct* __sync_struct) {
    // Check for paired write command.
    bool has_write_cmd = __sync_struct->m_comm_struct.hasMatchWrite(__cmd);
    InterChiplet::SyncCommand write_cmd = __sync_struct->m_comm_struct.popMatchWrite(__cmd);

    if (!has_write_cmd) {
        // If there is no paired write command, add this command to read command queue to wait.
        __sync_struct->m_comm_struct.insertRead(__cmd);
        spdlog::debug("{} Register READ command to pair with WRITE command.", dumpCmd(__cmd));
    } else {
        // Insert event to benchmark list.
        NetworkBenchItem bench_item(write_cmd, __cmd);
        __sync_struct->m_bench_list.insert(bench_item);

        // If there is a paired write command, get the end cycle of transaction.
        CmdDelayPair end_cycle = __sync_struct->m_delay_list.getEndCycle(write_cmd, __cmd);
        spdlog::debug("{} Pair with WRITE command. Transation ends at [{},{}] cycle.",
                      dumpCmd(__cmd), static_cast<InterChiplet::TimeType>(SRC_DELAY(end_cycle)),
                      static_cast<InterChiplet::TimeType>(DST_DELAY(end_cycle)));
        // Send synchronize command to response READ command.
        InterChiplet::sendSyncCmd(__cmd.m_stdin_fd, static_cast<InterChiplet::TimeType>(
                                                        DST_DELAY(end_cycle) * __cmd.m_clock_rate));
        // Send synchronize command to response WRITE command.
        InterChiplet::sendSyncCmd(
            write_cmd.m_stdin_fd,
            static_cast<InterChiplet::TimeType>(SRC_DELAY(end_cycle) * write_cmd.m_clock_rate));
    }
}

/**
 * @brief Handle WRITE command with barrier flag.
 * @param __cmd Command to handle.
 * @param __sync_struct Pointer to global synchronize structure.
 */
void handle_barrier_write_cmd(const InterChiplet::SyncCommand& __cmd, SyncStruct* __sync_struct) {
    int uid = DIM_X(__cmd.m_dst);
    int count = __cmd.m_desc & 0xFFFF;

    // Insert event to benchmark list.
    NetworkBenchItem bench_item(__cmd);
    __sync_struct->m_bench_list.insert(bench_item);

    // Register BARRIER command.
    __sync_struct->m_barrier_timing_struct.insertBarrier(uid, count, __cmd);
    // Barrier override.
    if (__sync_struct->m_barrier_timing_struct.overflow(uid)) {
        // Get barrier overflow time.
        InterChiplet::InnerTimeType barrier_cycle = __sync_struct->m_delay_list.getBarrierCycle(
            __sync_struct->m_barrier_timing_struct.barrierCmd(uid));
        spdlog::debug("{} Barrier overflow at {} cycle.", dumpCmd(__cmd),
                      static_cast<InterChiplet::TimeType>(barrier_cycle));

        // Generate a command as read command.
        InterChiplet::SyncCommand sync_cmd = __cmd;
        sync_cmd.m_cycle = barrier_cycle;

        // Send synchronization command to all barrier items.
        for (auto& item : __sync_struct->m_barrier_timing_struct.barrierCmd(uid)) {
            CmdDelayPair end_cycle = __sync_struct->m_delay_list.getEndCycle(item, sync_cmd);
            spdlog::debug("\t{} Transaction ends at {} cycle.", dumpCmd(item),
                          static_cast<InterChiplet::TimeType>(SRC_DELAY(end_cycle)));
            // Send synchronization comand to response WRITE command.
            InterChiplet::sendSyncCmd(
                item.m_stdin_fd,
                static_cast<InterChiplet::TimeType>(SRC_DELAY(end_cycle) * item.m_clock_rate));
        }
        __sync_struct->m_barrier_timing_struct.reset(uid);
    }
    // Wait further barrier commands.
    else {
        spdlog::debug("{} Register WRITE command with BARRIER flag.", dumpCmd(__cmd));
    }
}

/**
 * @brief Handle WRITE command with LOCK flag.
 * @param __cmd Command to handle.
 * @param __sync_struct Pointer to global synchronize structure.
 */
void handle_lock_write_cmd(const InterChiplet::SyncCommand& __cmd, SyncStruct* __sync_struct) {
    // Get mutex ID.
    int uid = DIM_X(__cmd.m_dst);

    // Insert event to benchmark list.
    NetworkBenchItem bench_item(__cmd);
    __sync_struct->m_bench_list.insert(bench_item);

    if (__sync_struct->m_lock_timing_struct.isLocked(uid)) {
        // If the mutex is locked, check whether the WRITE(LOCK) command comes from the same source.
        if (__sync_struct->m_lock_timing_struct.hasLastCmd(uid) &&
            (__sync_struct->m_lock_timing_struct.getLastCmd(uid).m_src == __cmd.m_src)) {
            // If the WRITE(LOCK) command comes from the same source, ignore the LOCK command.
            // Response this WRITE(LOCK) command immediately when it is received by destination.
            CmdDelayPair end_cycle = __sync_struct->m_delay_list.getEndCycle(__cmd, __cmd);
            spdlog::debug("{} Transaction ends at {} cycle.", dumpCmd(__cmd),
                          static_cast<InterChiplet::TimeType>(SRC_DELAY(end_cycle)));
            // Send SYNC comand with end cycle to response WRITE command.
            InterChiplet::sendSyncCmd(
                __cmd.m_stdin_fd,
                static_cast<InterChiplet::TimeType>(SRC_DELAY(end_cycle) * __cmd.m_clock_rate));
        } else {
            // Otherwise, pending the LOCK command until the mutex is unlocked.
            __sync_struct->m_lock_timing_struct.insertLockCmd(uid, __cmd);
            spdlog::debug("{} Register WRITE command with LOCK flag.", dumpCmd(__cmd));
        }
    } else {
        // The order of mutex is handled by LOCK/UNLOCK command. WRITE(LOCK) command can lock the
        // mutex directly.

        // Get the last command. If there is no last command, use this WRITE(LOCK) command as the
        // last command.
        InterChiplet::SyncCommand last_cmd = __cmd;
        if (__sync_struct->m_lock_timing_struct.hasLastCmd(uid)) {
            last_cmd = __sync_struct->m_lock_timing_struct.getLastCmd(uid);
        }
        // Response this WRITE(LOCK) command after it is received by the destination and the
        // destination finished the last command.
        CmdDelayPair end_cycle = __sync_struct->m_delay_list.getEndCycle(__cmd, last_cmd);

        // Calculate the end cycle of mutex.
        InterChiplet::SyncCommand sync_cmd = __cmd;
        sync_cmd.m_cycle = DST_DELAY(end_cycle);

        // Lock the mutex
        __sync_struct->m_lock_timing_struct.lock(uid, sync_cmd);
        spdlog::debug("{} Transaction ends at {} cycle.", dumpCmd(__cmd),
                      static_cast<InterChiplet::TimeType>(SRC_DELAY(end_cycle)));
        // Send SYNC comand with end cycle to response WRITE command.
        InterChiplet::sendSyncCmd(__cmd.m_stdin_fd, static_cast<InterChiplet::TimeType>(
                                                        SRC_DELAY(end_cycle) * __cmd.m_clock_rate));
    }
}

/**
 * @brief Handle WRITE command with UNLOCK flag.
 * @param __cmd Command to handle.
 * @param __sync_struct Pointer to global synchronize structure.
 */
void handle_unlock_write_cmd(const InterChiplet::SyncCommand& __cmd, SyncStruct* __sync_struct) {
    // Get mutex ID.
    int uid = DIM_X(__cmd.m_dst);

    // Insert event to benchmark list.
    NetworkBenchItem bench_item(__cmd);
    __sync_struct->m_bench_list.insert(bench_item);

    if (__sync_struct->m_lock_timing_struct.isLocked(uid)) {
        // If the mutex is locked, unlock the mutex first.

        // Get the last command. If there is no last command, use this WRITE(UNLOCK) command as the
        // last command.
        InterChiplet::SyncCommand last_cmd = __cmd;
        if (__sync_struct->m_lock_timing_struct.hasLastCmd(uid)) {
            last_cmd = __sync_struct->m_lock_timing_struct.getLastCmd(uid);
        }
        // Response this WRITE(UNLOCK) command after it is received by the destination and the
        // destination finished the last command.
        CmdDelayPair end_cycle = __sync_struct->m_delay_list.getEndCycle(__cmd, last_cmd);

        // Calculate the end cycle of mutex.
        InterChiplet::SyncCommand sync_cmd = __cmd;
        sync_cmd.m_cycle = DST_DELAY(end_cycle);

        // unlock the mutex.
        __sync_struct->m_lock_timing_struct.unlock(uid, sync_cmd);
        spdlog::debug("{} Transaction ends at {} cycle.", dumpCmd(__cmd),
                      static_cast<InterChiplet::TimeType>(SRC_DELAY(end_cycle)));
        // Send SYNC comand with end cycle to response WRITE command.
        InterChiplet::sendSyncCmd(__cmd.m_stdin_fd, static_cast<InterChiplet::TimeType>(
                                                        SRC_DELAY(end_cycle) * __cmd.m_clock_rate));

        // Lock the mutex by pending LOCK command.
        // The order of the mutex is handled by LOCK/UNLOCK commands. WRITE(UNLOCK) only need to
        // handle the reorder of one UNLOCK command with pending LOCK commands.
        // Check whether there is any pending LOCK command.
        if (__sync_struct->m_lock_timing_struct.hasLockCmd(uid)) {
            // If there is any pending LOCK command, lock the mutex by the first pending LOCK
            // command.
            InterChiplet::SyncCommand lock_cmd =
                __sync_struct->m_lock_timing_struct.popLockCmd(uid);
            // Response this WRITE(LOCK) command after it is received by the destination and the
            // destination finished the WRITE(UNLOCK) command.
            CmdDelayPair end_cycle = __sync_struct->m_delay_list.getEndCycle(lock_cmd, sync_cmd);

            // Calculate the end cycle of mutex.
            InterChiplet::SyncCommand sync_cmd = lock_cmd;
            sync_cmd.m_cycle = DST_DELAY(end_cycle);

            // lock the mutex.
            __sync_struct->m_lock_timing_struct.lock(uid, sync_cmd);
            spdlog::debug("\t{} Transaction ends at {} cycle.", dumpCmd(lock_cmd),
                          static_cast<InterChiplet::TimeType>(SRC_DELAY(end_cycle)));
            // Send SYNC comand with end cycle to response pending LOCK command.
            InterChiplet::sendSyncCmd(
                lock_cmd.m_stdin_fd,
                static_cast<InterChiplet::TimeType>(SRC_DELAY(end_cycle) * lock_cmd.m_clock_rate));
        }
    } else {
        // If the mutex is unlocked, ignore the UNLOCK command.
        // Response this WRITE(UNLOCK) command immediately when it is received by destination.
        CmdDelayPair end_cycle = __sync_struct->m_delay_list.getEndCycle(__cmd, __cmd);
        spdlog::debug("{} Transaction ends at {} cycle.", dumpCmd(__cmd),
                      static_cast<InterChiplet::TimeType>(SRC_DELAY(end_cycle)));
        // Send SYNC comand with end cycle to response WRITE command.
        InterChiplet::sendSyncCmd(__cmd.m_stdin_fd, static_cast<InterChiplet::TimeType>(
                                                        SRC_DELAY(end_cycle) * __cmd.m_clock_rate));
    }
}

void handle_write_cmd(const InterChiplet::SyncCommand& __cmd, SyncStruct* __sync_struct) {
    if (__cmd.m_desc & InterChiplet::SPD_BARRIER) {
        // Special handle WRITE cmmand after BARRIER. WRITE(BARRIER)
        return handle_barrier_write_cmd(__cmd, __sync_struct);
    } else if (__cmd.m_desc & InterChiplet::SPD_LOCK) {
        // Special handle WRITE cmmand after LOCK. WRITE(LOCK)
        return handle_lock_write_cmd(__cmd, __sync_struct);
    } else if (__cmd.m_desc & InterChiplet::SPD_UNLOCK) {
        // Special handle WRITE cmmand after UNLOCK. WRITE(UNLOCK)
        return handle_unlock_write_cmd(__cmd, __sync_struct);
    }

    // Check for paired read command.
    bool has_read_cmd = __sync_struct->m_comm_struct.hasMatchRead(__cmd);
    InterChiplet::SyncCommand read_cmd = __sync_struct->m_comm_struct.popMatchRead(__cmd);

    if (!has_read_cmd) {
        // If there is no paired read command, add this command to write command queue to wait.
        __sync_struct->m_comm_struct.insertWrite(__cmd);
        spdlog::debug("{} Register WRITE command to pair with READ command.", dumpCmd(__cmd));
    } else {
        // Insert event to benchmark list.
        NetworkBenchItem bench_item(__cmd, read_cmd);
        __sync_struct->m_bench_list.insert(bench_item);

        // If there is a paired read command, get the end cycle of transaction.
        CmdDelayPair end_cycle = __sync_struct->m_delay_list.getEndCycle(__cmd, read_cmd);
        spdlog::debug("{} Pair with READ command. Transation ends at [{},{}] cycle.",
                      dumpCmd(__cmd), static_cast<InterChiplet::TimeType>(SRC_DELAY(end_cycle)),
                      static_cast<InterChiplet::TimeType>(DST_DELAY(end_cycle)));

        // Send synchronize command to response WRITE command.
        InterChiplet::sendSyncCmd(__cmd.m_stdin_fd, static_cast<InterChiplet::TimeType>(
                                                        SRC_DELAY(end_cycle) * __cmd.m_clock_rate));
        // Send synchronize command to response READ command.
        InterChiplet::sendSyncCmd(
            read_cmd.m_stdin_fd,
            static_cast<InterChiplet::TimeType>(DST_DELAY(end_cycle) * read_cmd.m_clock_rate));
    }
}
