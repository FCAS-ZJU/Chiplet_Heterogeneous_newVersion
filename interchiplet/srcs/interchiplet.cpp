
#include <fcntl.h>
#include <poll.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <sys/types.h>
#include <sys/wait.h>

#include <ctime>
#include <vector>

#include "benchmark_yaml.h"
#include "cmdline_options.h"
#include "net_bench.h"
#include "net_delay.h"
#include "spdlog/spdlog.h"
#include "sync_protocol.h"

/**
 * @brief Data structure of synchronize operation.
 */
class SyncStruct {
   public:
    /**
     * @brief Construct synchronize stucture.
     *
     * Initializete mutex.
     */
    SyncStruct() : m_cycle(0) {
        if (pthread_mutex_init(&m_mutex, NULL) < 0) {
            perror("pthread_mutex_init");
            exit(EXIT_FAILURE);
        }
    }

    /**
     * @brief Destory synchronize structure.
     *
     * Destory mutex.
     */
    ~SyncStruct() { pthread_mutex_destroy(&m_mutex); }

   public:
    /**
     * @brief Mutex to access this structure.
     */
    pthread_mutex_t m_mutex;

    /**
     * @brief Global simulation cycle, which is the largest notified cycle count.
     */
    InterChiplet::InnerTimeType m_cycle;

    /**
     * @brief Benchmark list, recording the communication transactions have sent out.
     */
    InterChiplet::NetworkBenchList m_bench_list;
    /**
     * @brief Delay list, recording the delay of each communication transactions
     */
    InterChiplet::NetworkDelayList m_delay_list;
    /**
     * @brief Lock delay list. recording the delay of lock/waitlock transactions
     */
    InterChiplet::NetworkDelayList m_lock_delay_list;

    /**
     * @brief List of PIPE file names.
     */
    std::vector<std::string> m_fifo_list;
    /**
     * @brief List of Read commands, used to pair with write commands.
     */
    std::vector<InterChiplet::SyncCommand> m_read_cmd_list;
    /**
     * @brief List of Write commands, used to pair with write commands.
     */
    std::vector<InterChiplet::SyncCommand> m_write_cmd_list;

    /**
     * @brief List of Pipe commands.
     */
    std::vector<InterChiplet::SyncCommand> m_lock_cmd_list;
    /**
     * @brief List of Pending Pipe commands.
     */
    std::vector<InterChiplet::SyncCommand> m_pending_lock_cmd_list;
    /**
     * @brief List of Pending Pipe commands.
     */
    std::vector<InterChiplet::SyncCommand> m_waitlock_cmd_list;

    /**
     * @brief Barrier count.
     */
    std::map<int, int> m_barrier_count_map;
    /**
     * @brief Barrier items.
     */
    std::map<int, std::vector<InterChiplet::SyncCommand> > m_barrier_items_map;
    /**
     * @brief Barrier items.
     */
    std::map<int, std::vector<InterChiplet::SyncCommand> > m_barrier_write_items_map;
};

/**
 * @brief Data structure of process configuration.
 */
class ProcessStruct {
   public:
    ProcessStruct(const InterChiplet::ProcessConfig& __config)
        : m_command(__config.m_command),
          m_args(__config.m_args),
          m_log_file(__config.m_log_file),
          m_to_stdout(__config.m_to_stdout),
          m_clock_rate(__config.m_clock_rate),
          m_pre_copy(__config.m_pre_copy),
          m_unfinished_line(),
          m_thread_id(),
          m_pid(-1),
          m_pid2(-1),
          m_sync_struct(NULL) {}

   public:
    // Configuration.
    std::string m_command;
    std::vector<std::string> m_args;
    std::string m_log_file;
    bool m_to_stdout;
    double m_clock_rate;
    std::string m_pre_copy;

    std::string m_unfinished_line;

    // Indentify
    int m_round;
    int m_phase;
    int m_thread;

    pthread_t m_thread_id;
    int m_pid;
    int m_pid2;

    /**
     * @brief Pointer to synchronize structure.
     */
    SyncStruct* m_sync_struct;
};

/**
 * @brief  Create FIFO with specified name.
 * @param __fifo_name Specified name for PIPE.
 * @retval 0 Operation success. PIPE file is existed or created.
 * @retval -1 Operation fail. PIPE file is missing.
 */
int create_fifo(std::string __fifo_name) {
    if (access(__fifo_name.c_str(), F_OK) == -1) {
        // Report error if FIFO file does not exist and mkfifo error.
        if (mkfifo(__fifo_name.c_str(), 0664) == -1) {
            return -1;
        }
        // Report success.
        else {
            return 0;
        }
    }
    // Reuse exist FIFO and reports.
    else {
        return 0;
    }
}

static std::string cmdToDebug(const InterChiplet::SyncCommand& __cmd) {
    std::stringstream ss;
    if (__cmd.m_type == InterChiplet::SC_LOCK) {
        ss << "LOCK command from " << __cmd.m_src_x << "," << __cmd.m_src_y << " to "
           << __cmd.m_dst_x << "," << __cmd.m_dst_y << ".";
    } else if (__cmd.m_type == InterChiplet::SC_WAITLOCK) {
        ss << "WAITLOCK command from " << __cmd.m_src_x << "," << __cmd.m_src_y << " to "
           << __cmd.m_dst_x << "," << __cmd.m_dst_y << ".";
    } else if (__cmd.m_type == InterChiplet::SC_UNLOCK) {
        ss << "UNLOCK command from " << __cmd.m_src_x << "," << __cmd.m_src_y << " to "
           << __cmd.m_dst_x << "," << __cmd.m_dst_y << ".";
    } else if (__cmd.m_type == InterChiplet::SC_BARRIER) {
        ss << "BARRIER command from " << __cmd.m_src_x << "," << __cmd.m_src_y << " to "
           << __cmd.m_dst_x << ".";
    } else if (__cmd.m_type == InterChiplet::SC_PIPE) {
        ss << "PIPE command from " << __cmd.m_src_x << "," << __cmd.m_src_y << " to "
           << __cmd.m_dst_x << "," << __cmd.m_dst_y << ".";
    } else if (__cmd.m_type == InterChiplet::SC_READ) {
        ss << "READ command at " << static_cast<InterChiplet::TimeType>(__cmd.m_cycle)
           << " cycle from " << __cmd.m_src_x << "," << __cmd.m_src_y << " to " << __cmd.m_dst_x
           << "," << __cmd.m_dst_y << ".";
    } else if (__cmd.m_type == InterChiplet::SC_WRITE) {
        ss << "WRITE command at " << static_cast<InterChiplet::TimeType>(__cmd.m_cycle)
           << " cycle from " << __cmd.m_src_x << "," << __cmd.m_src_y << " to " << __cmd.m_dst_x
           << "," << __cmd.m_dst_y << ".";
    } else if (__cmd.m_type == InterChiplet::SC_CYCLE) {
        ss << "CYCLE command at " << static_cast<InterChiplet::TimeType>(__cmd.m_cycle)
           << " cycle.";
    }
    return ss.str();
}

void handle_lock_cmd(const InterChiplet::SyncCommand& __cmd, SyncStruct* __sync_struct) {
    // Check for unconfirmed waitlock command.
    bool has_waitlock_cmd = false;
    InterChiplet::SyncCommand waitlock_cmd;

    for (std::size_t i = 0; i < __sync_struct->m_waitlock_cmd_list.size(); i++) {
        InterChiplet::SyncCommand& __waitlock_cmd = __sync_struct->m_waitlock_cmd_list[i];
        // If there is waitlock command, confirm the lock.
        bool lock_match = false;
        if (__waitlock_cmd.m_src_x < 0 || __waitlock_cmd.m_src_y < 0) {
            lock_match =
                __cmd.m_dst_x == __waitlock_cmd.m_dst_x && __cmd.m_dst_y == __waitlock_cmd.m_dst_y;
        } else {
            lock_match = __cmd.m_src_x == __waitlock_cmd.m_src_x &&
                         __cmd.m_src_y == __waitlock_cmd.m_src_y &&
                         __cmd.m_dst_x == __waitlock_cmd.m_dst_x &&
                         __cmd.m_dst_y == __waitlock_cmd.m_dst_y;
        }
        if (lock_match) {
            has_waitlock_cmd = true;
            waitlock_cmd = __waitlock_cmd;
            __sync_struct->m_waitlock_cmd_list.erase(__sync_struct->m_waitlock_cmd_list.begin() +
                                                     i);
            break;
        }
    }

    // If there is not waitlock command, waitlock command.
    if (!has_waitlock_cmd) {
        spdlog::debug("{} Register LOCK command to pair with WAITLOCK command.", cmdToDebug(__cmd));
        __sync_struct->m_pending_lock_cmd_list.push_back(__cmd);
        // If there is waitlock command, response lock and waitlock command.
    } else {
        spdlog::debug("{} Pair with WAITLOCK command.", cmdToDebug(__cmd));

        // Append to lock queue.
        __sync_struct->m_lock_cmd_list.push_back(__cmd);

        // Send SYNC to response LOCK command.
        InterChiplet::SyncProtocol::sendSyncCmd(__cmd.m_stdin_fd, 0);

        // Send LOCK to response WAITLOCK command.
        InterChiplet::SyncProtocol::sendLockCmd(waitlock_cmd.m_stdin_fd, __cmd.m_src_x,
                                                __cmd.m_src_y, __cmd.m_dst_x, __cmd.m_dst_y);
    }
}

void handle_waitlock_cmd(const InterChiplet::SyncCommand& __cmd, SyncStruct* __sync_struct) {
    // Check for unconfirmed waitlock command.
    bool has_lock_cmd = false;
    InterChiplet::SyncCommand lock_cmd;

    // Check lock order and remove item..
    InterChiplet::SyncCommand waitlock_cmd = __cmd;
    std::multimap<InterChiplet::InnerTimeType, InterChiplet::NetworkDelayItem>::iterator it =
        __sync_struct->m_lock_delay_list.find_first_item(waitlock_cmd.m_dst_x,
                                                         waitlock_cmd.m_dst_y);
    if (it != __sync_struct->m_lock_delay_list.end()) {
        waitlock_cmd.m_src_x = it->second.m_src_x;
        waitlock_cmd.m_src_y = it->second.m_src_y;
        __sync_struct->m_lock_delay_list.erase(it);
    }

    // Try to pick with lock command.
    for (std::size_t i = 0; i < __sync_struct->m_pending_lock_cmd_list.size(); i++) {
        InterChiplet::SyncCommand& __lock_cmd = __sync_struct->m_pending_lock_cmd_list[i];
        // If there is lock command, confirm the lock.
        bool lock_match = false;
        if (waitlock_cmd.m_src_x < 0 || waitlock_cmd.m_src_y < 0) {
            lock_match = __lock_cmd.m_dst_x == waitlock_cmd.m_dst_x &&
                         __lock_cmd.m_dst_y == waitlock_cmd.m_dst_y;
        } else {
            lock_match = __lock_cmd.m_src_x == waitlock_cmd.m_src_x &&
                         __lock_cmd.m_src_y == waitlock_cmd.m_src_y &&
                         __lock_cmd.m_dst_x == waitlock_cmd.m_dst_x &&
                         __lock_cmd.m_dst_y == waitlock_cmd.m_dst_y;
        }
        if (lock_match) {
            has_lock_cmd = true;
            lock_cmd = __lock_cmd;
            __sync_struct->m_pending_lock_cmd_list.erase(
                __sync_struct->m_pending_lock_cmd_list.begin() + i);
            break;
        }
    }

    // If there is not waitlock command, waitlock command.
    if (!has_lock_cmd) {
        spdlog::debug("{} Register WAITLOCK command to pair with LOCK command.",
                      cmdToDebug(waitlock_cmd));
        __sync_struct->m_waitlock_cmd_list.push_back(waitlock_cmd);
    } else {
        spdlog::debug("{} Pair with LOCK command from {},{} to {},{}.", cmdToDebug(waitlock_cmd),
                      lock_cmd.m_src_x, lock_cmd.m_src_y, lock_cmd.m_dst_x, lock_cmd.m_dst_y);

        // Append to lock queue.
        __sync_struct->m_lock_cmd_list.push_back(lock_cmd);

        // Send SYNC to response LOCK command.
        InterChiplet::SyncProtocol::sendSyncCmd(lock_cmd.m_stdin_fd, 0);

        // Send LOCK to response WAITLOCK command.
        InterChiplet::SyncProtocol::sendLockCmd(waitlock_cmd.m_stdin_fd, lock_cmd.m_src_x,
                                                lock_cmd.m_src_y, lock_cmd.m_dst_x,
                                                lock_cmd.m_dst_y);
    }
}

void handle_unlock_cmd(const InterChiplet::SyncCommand& __cmd, SyncStruct* __sync_struct) {
    // Unlock resources.
    for (std::size_t i = 0; i < __sync_struct->m_lock_cmd_list.size(); i++) {
        InterChiplet::SyncCommand& __lock_cmd = __sync_struct->m_lock_cmd_list[i];
        // Remove lock_cmd from lock command queue.
        if (__lock_cmd.m_src_x == __cmd.m_src_x && __lock_cmd.m_src_y == __cmd.m_src_y &&
            __lock_cmd.m_dst_x == __cmd.m_dst_x && __lock_cmd.m_dst_y == __cmd.m_dst_y) {
            __sync_struct->m_lock_cmd_list.erase(__sync_struct->m_lock_cmd_list.begin() + i);
            break;
        }
    }

    spdlog::debug("{}", cmdToDebug(__cmd));

    // Send synchronize command.
    InterChiplet::SyncProtocol::sendSyncCmd(__cmd.m_stdin_fd, 0);
}

void handle_barrier_cmd(const InterChiplet::SyncCommand& __cmd, SyncStruct* __sync_struct) {
    int uid = __cmd.m_dst_x;
    int count = __cmd.m_nbytes;

    // New barrier.
    if (__sync_struct->m_barrier_count_map.find(uid) == __sync_struct->m_barrier_count_map.end()) {
        // Not specify the count of items, return sync directly.
        if (count == 0) {
            spdlog::debug("{} Barrier overflow.", cmdToDebug(__cmd));
            InterChiplet::SyncProtocol::sendSyncCmd(__cmd.m_stdin_fd, 0);
        }
        // If the count of items is 1, register barrier, return sync.
        else if (count == 1) {
            spdlog::debug("{} Register new barrier. Barrier overflow.", cmdToDebug(__cmd));
            __sync_struct->m_barrier_count_map[uid] = count;
            __sync_struct->m_barrier_items_map[uid] = std::vector<InterChiplet::SyncCommand>();

            InterChiplet::SyncProtocol::sendSyncCmd(__cmd.m_stdin_fd, 0);
        }
        // If the count of items is greater than 1, register barrier, add barrier item.
        else {
            spdlog::debug("{} Register new barrier. Add barrier item.", cmdToDebug(__cmd));
            __sync_struct->m_barrier_count_map[uid] = count;
            __sync_struct->m_barrier_items_map[uid] = std::vector<InterChiplet::SyncCommand>();
            __sync_struct->m_barrier_items_map[uid].push_back(__cmd);
        }
    }
    // Exist barrier.
    else {
        // Add barrier item.
        if (__sync_struct->m_barrier_items_map.find(uid) ==
            __sync_struct->m_barrier_items_map.end()) {
            __sync_struct->m_barrier_items_map[uid] = std::vector<InterChiplet::SyncCommand>();
        }
        __sync_struct->m_barrier_items_map[uid].push_back(__cmd);

        // Update counter.
        if (count > 0) {
            __sync_struct->m_barrier_count_map[uid] = count;
        }

        // If barrier override.
        if (__sync_struct->m_barrier_items_map[uid].size() >=
            __sync_struct->m_barrier_count_map[uid]) {
            spdlog::debug("{} Add barrier item. Barrier overflow.", cmdToDebug(__cmd));
            for (InterChiplet::SyncCommand item : __sync_struct->m_barrier_items_map[uid]) {
                InterChiplet::SyncProtocol::sendSyncCmd(item.m_stdin_fd, 0);
            }
            __sync_struct->m_barrier_items_map[uid].clear();
        } else {
            spdlog::debug("{} Add barrier item.", cmdToDebug(__cmd));
        }
    }
}

/**
 * @brief Handle PIPE command.
 * @param __cmd Command to handle.
 * @param __sync_struct Pointer to global synchronize structure.
 */
void handle_pipe_cmd(const InterChiplet::SyncCommand& __cmd, SyncStruct* __sync_struct) {
    // Create Pipe file and add the file to
    std::string file_name = InterChiplet::SyncProtocol::pipeNameMaster(
        __cmd.m_src_x, __cmd.m_src_y, __cmd.m_dst_x, __cmd.m_dst_y);
    if (create_fifo(file_name.c_str()) == 0) {
        __sync_struct->m_fifo_list.push_back(file_name);
    }
    spdlog::debug("{} Create/Reuse pipe file {}.", cmdToDebug(__cmd), file_name);

    // Send synchronize command.
    InterChiplet::SyncProtocol::sendSyncCmd(__cmd.m_stdin_fd, 0);
}

/**
 * @brief Handle READ command.
 * @param __cmd Command to handle.
 * @param __sync_struct Pointer to global synchronize structure.
 */
void handle_read_cmd(const InterChiplet::SyncCommand& __cmd, SyncStruct* __sync_struct) {
    // Check for paired write command.
    bool has_write_cmd = false;
    InterChiplet::SyncCommand write_cmd;
    for (std::size_t i = 0; i < __sync_struct->m_write_cmd_list.size(); i++) {
        InterChiplet::SyncCommand& __write_cmd = __sync_struct->m_write_cmd_list[i];
        if (__write_cmd.m_src_x == __cmd.m_src_x && __write_cmd.m_src_y == __cmd.m_src_y &&
            __write_cmd.m_dst_x == __cmd.m_dst_x && __write_cmd.m_dst_y == __cmd.m_dst_y &&
            __write_cmd.m_nbytes == __cmd.m_nbytes) {
            has_write_cmd = true;
            write_cmd = __write_cmd;
            __sync_struct->m_write_cmd_list.erase(__sync_struct->m_write_cmd_list.begin() + i);
            break;
        }
    }

    if (!has_write_cmd) {
        // If there is no paired write command, add this command to read command queue to wait.
        __sync_struct->m_read_cmd_list.push_back(__cmd);
        spdlog::debug("{} Register READ command to pair with WRITE command.", cmdToDebug(__cmd));
    } else {
        // If there is a paired write command, get the end cycle of transaction.
        std::tuple<InterChiplet::InnerTimeType, InterChiplet::InnerTimeType> end_cycle =
            __sync_struct->m_delay_list.getEndCycle(write_cmd, __cmd);
        InterChiplet::InnerTimeType write_end_cycle = std::get<0>(end_cycle);
        InterChiplet::InnerTimeType read_end_cycle = std::get<1>(end_cycle);
        spdlog::debug("{} Pair with WRITE command. Transation ends at [{},{}] cycle.",
                      cmdToDebug(__cmd), static_cast<InterChiplet::TimeType>(write_end_cycle),
                      static_cast<InterChiplet::TimeType>(read_end_cycle));

        // Insert event to benchmark list.
        InterChiplet::NetworkBenchItem bench_item(write_cmd, __cmd);
        __sync_struct->m_bench_list.insert(bench_item);

        // Send synchronize command to response READ command.
        InterChiplet::SyncProtocol::sendSyncCmd(
            __cmd.m_stdin_fd,
            static_cast<InterChiplet::TimeType>(read_end_cycle * __cmd.m_clock_rate));

        // Send synchronize command to response WRITE command.
        InterChiplet::SyncProtocol::sendSyncCmd(
            write_cmd.m_stdin_fd,
            static_cast<InterChiplet::TimeType>(write_end_cycle * write_cmd.m_clock_rate));
    }
}

void handle_barrier_write_cmd(const InterChiplet::SyncCommand& __cmd, SyncStruct* __sync_struct) {
    int uid = __cmd.m_dst_x;
    int count = __cmd.m_desc & 0xFFFF;

    // Insert event to benchmark list.
    InterChiplet::NetworkBenchItem bench_item(__cmd);
    __sync_struct->m_bench_list.insert(bench_item);

    if (__sync_struct->m_barrier_count_map.find(uid) == __sync_struct->m_barrier_count_map.end()) {
        InterChiplet::InnerTimeType end_cycle = __sync_struct->m_delay_list.getEndCycle(__cmd);
        spdlog::debug("{} Transation ends at {},{} cycle.", cmdToDebug(__cmd),
                      static_cast<InterChiplet::TimeType>(end_cycle));

        // Send synchronize command to response WRITE command.
        InterChiplet::SyncProtocol::sendSyncCmd(
            __cmd.m_stdin_fd, static_cast<InterChiplet::TimeType>(end_cycle * __cmd.m_clock_rate));
    } else {
        // Add barrier item.
        if (__sync_struct->m_barrier_write_items_map.find(uid) ==
            __sync_struct->m_barrier_write_items_map.end()) {
            __sync_struct->m_barrier_write_items_map[uid] =
                std::vector<InterChiplet::SyncCommand>();
        }
        __sync_struct->m_barrier_write_items_map[uid].push_back(__cmd);

        // If barrier override.
        if (__sync_struct->m_barrier_write_items_map[uid].size() >=
            __sync_struct->m_barrier_count_map[uid]) {
            // Get barrier overflow time.
            InterChiplet::InnerTimeType barrier_cycle = __sync_struct->m_delay_list.getBarrierCycle(
                __sync_struct->m_barrier_write_items_map[uid]);
            spdlog::debug("{} Barrier overflow at {} cycle.", cmdToDebug(__cmd),
                          static_cast<InterChiplet::TimeType>(barrier_cycle));
            // Generate a command as read command.
            InterChiplet::SyncCommand sync_cmd = __cmd;
            sync_cmd.m_cycle = barrier_cycle;

            // Send synchronization command to all barrier items.
            for (InterChiplet::SyncCommand item : __sync_struct->m_barrier_write_items_map[uid]) {
                std::tuple<InterChiplet::InnerTimeType, InterChiplet::InnerTimeType> end_cycle =
                    __sync_struct->m_delay_list.getEndCycle(item, sync_cmd);
                InterChiplet::InnerTimeType write_end_cycle = std::get<0>(end_cycle);
                spdlog::debug("\t{} Transaction ends at {} cycle.", cmdToDebug(item),
                              static_cast<InterChiplet::TimeType>(write_end_cycle));

                // Send synchronization comand to response WRITE command.
                InterChiplet::SyncProtocol::sendSyncCmd(
                    item.m_stdin_fd,
                    static_cast<InterChiplet::TimeType>(write_end_cycle * __cmd.m_clock_rate));
            }
            __sync_struct->m_barrier_write_items_map[uid].clear();
        } else {
            spdlog::debug("{} Wait for other barrier items.", cmdToDebug(__cmd));
        }
    }
}

/**
 * @brief Handle WRITE command.
 * @param __cmd Command to handle.
 * @param __sync_struct Pointer to global synchronize structure.
 */
void handle_write_cmd(const InterChiplet::SyncCommand& __cmd, SyncStruct* __sync_struct) {
    // Special handle WRITE cmmand after BARRIER.
    if (__cmd.m_desc & InterChiplet::SPD_BARRIER) {
        return handle_barrier_write_cmd(__cmd, __sync_struct);
    }

    // Check for paired read command.
    bool has_read_cmd = false;
    InterChiplet::SyncCommand read_cmd;
    for (std::size_t i = 0; i < __sync_struct->m_read_cmd_list.size(); i++) {
        InterChiplet::SyncCommand& __read_cmd = __sync_struct->m_read_cmd_list[i];
        if (__read_cmd.m_src_x == __cmd.m_src_x && __read_cmd.m_src_y == __cmd.m_src_y &&
            __read_cmd.m_dst_x == __cmd.m_dst_x && __read_cmd.m_dst_y == __cmd.m_dst_y &&
            __read_cmd.m_nbytes == __cmd.m_nbytes) {
            has_read_cmd = true;
            read_cmd = __read_cmd;
            __sync_struct->m_read_cmd_list.erase(__sync_struct->m_read_cmd_list.begin() + i);
            break;
        }
    }

    if (!has_read_cmd) {
        // If there is no paired read command, add this command to write command queue to wait.
        __sync_struct->m_write_cmd_list.push_back(__cmd);
        spdlog::debug("{} Register WRITE command to pair with READ command.", cmdToDebug(__cmd));
    } else {
        // If there is a paired read command, get the end cycle of transaction.
        std::tuple<InterChiplet::InnerTimeType, InterChiplet::InnerTimeType> end_cycle =
            __sync_struct->m_delay_list.getEndCycle(__cmd, read_cmd);
        InterChiplet::InnerTimeType write_end_cycle = std::get<0>(end_cycle);
        InterChiplet::InnerTimeType read_end_cycle = std::get<1>(end_cycle);
        spdlog::debug("{} Pair with READ command. Transation ends at [{},{}] cycle.",
                      cmdToDebug(__cmd), static_cast<InterChiplet::TimeType>(write_end_cycle),
                      static_cast<InterChiplet::TimeType>(read_end_cycle));

        // Insert event to benchmark list.
        InterChiplet::NetworkBenchItem bench_item(__cmd, read_cmd);
        __sync_struct->m_bench_list.insert(bench_item);

        // Send synchronize command to response WRITE command.
        InterChiplet::SyncProtocol::sendSyncCmd(
            __cmd.m_stdin_fd,
            static_cast<InterChiplet::TimeType>(write_end_cycle * __cmd.m_clock_rate));

        // Send synchronize command to response READ command.
        InterChiplet::SyncProtocol::sendSyncCmd(
            read_cmd.m_stdin_fd,
            static_cast<InterChiplet::TimeType>(read_end_cycle * read_cmd.m_clock_rate));
    }
}

/**
 * @brief Handle CYCLE command.
 * @param __cmd Command to handle.
 * @param __sync_struct Pointer to global synchronize structure.
 */
void handle_cycle_cmd(const InterChiplet::SyncCommand& __cmd, SyncStruct* __sync_struct) {
    spdlog::debug("{}", cmdToDebug(__cmd));

    // Update global cycle.
    InterChiplet::InnerTimeType new_cycle = __cmd.m_cycle;
    if (__sync_struct->m_cycle < new_cycle) {
        __sync_struct->m_cycle = new_cycle;
    }
}

void parse_command(char* __pipe_buf, ProcessStruct* __proc_struct, int __stdin_fd) {
    // Split line by '\n'
    std::string line = std::string(__pipe_buf);
    std::vector<std::string> lines;

    int start_idx = 0;
    for (std::size_t i = 0; i < line.size(); i++) {
        if (line[i] == '\n') {
            std::string l = line.substr(start_idx, i + 1 - start_idx);
            start_idx = i + 1;
            lines.push_back(l);
        }
    }
    if (start_idx < line.size()) {
        std::string l = line.substr(start_idx, line.size() - start_idx);
        lines.push_back(l);
    }

    // Unfinished line.
    if (__proc_struct->m_unfinished_line.size() > 0) {
        lines[0] = __proc_struct->m_unfinished_line + lines[0];
        __proc_struct->m_unfinished_line = "";
    }
    if (lines[lines.size() - 1].find('\n') == -1) {
        __proc_struct->m_unfinished_line = lines[lines.size() - 1];
        lines.pop_back();
    }

    // Get line start with [INTERCMD]
    for (std::size_t i = 0; i < lines.size(); i++) {
        std::string l = lines[i];
        if (l.substr(0, 10) == "[INTERCMD]") {
            InterChiplet::SyncCommand cmd = InterChiplet::SyncProtocol::parseCmd(l);
            cmd.m_stdin_fd = __stdin_fd;
            cmd.m_clock_rate = __proc_struct->m_clock_rate;
            cmd.m_cycle = cmd.m_cycle / __proc_struct->m_clock_rate;

            pthread_mutex_lock(&__proc_struct->m_sync_struct->m_mutex);

            // Call functions to handle corresponding command.
            switch (cmd.m_type) {
                case InterChiplet::SC_CYCLE:
                    handle_cycle_cmd(cmd, __proc_struct->m_sync_struct);
                    break;
                case InterChiplet::SC_BARRIER:
                    handle_barrier_cmd(cmd, __proc_struct->m_sync_struct);
                    break;
                case InterChiplet::SC_PIPE:
                    handle_pipe_cmd(cmd, __proc_struct->m_sync_struct);
                    break;
                case InterChiplet::SC_LOCK:
                    handle_lock_cmd(cmd, __proc_struct->m_sync_struct);
                    break;
                case InterChiplet::SC_UNLOCK:
                    handle_unlock_cmd(cmd, __proc_struct->m_sync_struct);
                    break;
                case InterChiplet::SC_WAITLOCK:
                    handle_waitlock_cmd(cmd, __proc_struct->m_sync_struct);
                    break;
                case InterChiplet::SC_READ:
                    handle_read_cmd(cmd, __proc_struct->m_sync_struct);
                    break;
                case InterChiplet::SC_WRITE:
                    handle_write_cmd(cmd, __proc_struct->m_sync_struct);
                    break;
                default:
                    break;
            }

            pthread_mutex_unlock(&__proc_struct->m_sync_struct->m_mutex);
        }
    }
}

#define PIPE_BUF_SIZE 1024

void* bridge_thread(void* __args_ptr) {
    ProcessStruct* proc_struct = (ProcessStruct*)__args_ptr;

    int pipe_stdin[2];   // Pipe to send data to child process
    int pipe_stdout[2];  // Pipe to receive data from child process
    int pipe_stderr[2];  // Pipe to receive data from child process

    // Create pipes
    if (pipe(pipe_stdin) == -1 || pipe(pipe_stdout) == -1 || pipe(pipe_stderr) == -1) {
        perror("pipe");
        exit(EXIT_FAILURE);
    }

    // Create sub directory for subprocess.
    char* sub_dir_path = new char[128];
    sprintf(sub_dir_path, "./proc_r%d_p%d_t%d", proc_struct->m_round, proc_struct->m_phase,
            proc_struct->m_thread);
    if (access(sub_dir_path, F_OK) == -1) {
        mkdir(sub_dir_path, 0775);
    }

    // Fork a child process
    pid_t pid = fork();
    if (pid == -1) {
        perror("fork");
        exit(EXIT_FAILURE);
    }

    if (pid == 0) {  // Child process
        // Close unnecessary pipe ends
        close(pipe_stdin[1]);
        close(pipe_stdout[0]);
        close(pipe_stderr[0]);

        // Redirect stdin and stdout to pipes
        dup2(pipe_stdin[0], STDIN_FILENO);
        dup2(pipe_stdout[1], STDOUT_FILENO);
        dup2(pipe_stderr[1], STDERR_FILENO);

        // Change directory to sub-process.
        if (chdir(sub_dir_path) < 0) {
            perror("chdir");
        }
        // TODO: Copy necessary configuration file.
        if (!proc_struct->m_pre_copy.empty()) {
            std::string cp_cmd = std::string("cp ") + proc_struct->m_pre_copy + " .";
            if (system(cp_cmd.c_str()) != 0) {
                perror("system");
            }
        }

        std::cout << "CWD: " << get_current_dir_name() << std::endl;

        // Build arguments.
        int argc = proc_struct->m_args.size();
        char** args_list = new char*[argc + 2];
        args_list[0] = new char[proc_struct->m_command.size() + 1];
        strcpy(args_list[0], proc_struct->m_command.c_str());
        args_list[0][proc_struct->m_command.size()] = '\0';
        for (int i = 0; i < argc; i++) {
            int arg_len = proc_struct->m_args[i].size();
            args_list[i + 1] = new char[arg_len + 1];
            strcpy(args_list[i + 1], proc_struct->m_args[i].c_str());
            args_list[i + 1][arg_len] = '\0';
        }
        args_list[argc + 1] = NULL;

        // Execute the child program
        std::cout << "Exec: ";
        for (int i = 0; i < proc_struct->m_args.size() + 1; i++) {
            std::cout << " " << args_list[i];
        }
        std::cout << std::endl;
        execvp(args_list[0], args_list);

        // If execl fails, it means the child process couldn't be started
        perror("execvp");
        exit(EXIT_FAILURE);
    } else {  // Parent process
        spdlog::info("Start simulation process {}. Command: {}", pid, proc_struct->m_command);
        proc_struct->m_pid2 = pid;

        // Close unnecessary pipe ends
        close(pipe_stdin[0]);
        close(pipe_stdout[1]);
        close(pipe_stderr[1]);
        int stdin_fd = pipe_stdin[1];
        int stdout_fd = pipe_stdout[0];
        int stderr_fd = pipe_stderr[0];

        pollfd fd_list[2] = {{fd : stdout_fd, events : POLL_IN},
                             {fd : stderr_fd, events : POLL_IN}};

        // Move log to subfolder.
        std::ofstream log_file(std::string(sub_dir_path) + "/" + proc_struct->m_log_file);

        // Write execution start time to log file.
        std::time_t t = std::time(0);
        std::tm* now = std::localtime(&t);
        log_file << "Execution starts at " << (now->tm_year + 1900) << "-" << (now->tm_mon + 1)
                 << "-" << now->tm_mday << "  " << (now->tm_hour) << ":" << (now->tm_min) << ":"
                 << (now->tm_sec) << std::endl;

        char* pipe_buf = new char[PIPE_BUF_SIZE + 1];
        bool to_stdout = proc_struct->m_to_stdout;
        int res = 0;
        while (true) {
            int res = poll(fd_list, 2, 1000);
            if (res == -1) {
                perror("poll");
                break;
            }

            bool has_stdout = false;
            if (fd_list[0].revents & POLL_IN) {
                has_stdout = true;
                int res = read(stdout_fd, pipe_buf, PIPE_BUF_SIZE);
                if (res <= 0) break;
                pipe_buf[res] = '\0';
                // log redirection.
                log_file.write(pipe_buf, res).flush();
                if (to_stdout) {
                    std::cout.write(pipe_buf, res);
                    std::cout.flush();
                }
                // Parse command in pipe_buf
                parse_command(pipe_buf, proc_struct, stdin_fd);
            }
            if (fd_list[1].revents & POLL_IN) {
                has_stdout = true;
                int res = read(stderr_fd, pipe_buf, PIPE_BUF_SIZE);
                if (res <= 0) break;
                pipe_buf[res] = '\0';
                // log redirection.
                log_file.write(pipe_buf, res).flush();
                if (to_stdout) {
                    std::cerr.write(pipe_buf, res);
                    std::cerr.flush();
                }
                // Parse command in pipe_buf
                parse_command(pipe_buf, proc_struct, stdin_fd);
            }

            // Check the status of child process and quit.
            int status;
            if (!has_stdout && (waitpid(pid, &status, WNOHANG) > 0)) {
                // Optionally handle child process termination status
                if (status == 0) {
                    spdlog::info("Simulation process {} terminate with status = {}.",
                                 proc_struct->m_pid2, status);
                } else {
                    spdlog::error("Simulation process {} terminate with status = {}.",
                                  proc_struct->m_pid2, status);
                }
                break;
            }
        }

        delete pipe_buf;
    }

    return 0;
}

InterChiplet::InnerTimeType __loop_phase_one(
    int __round, const std::vector<InterChiplet::ProcessConfig>& __proc_phase1_cfg_list,
    const std::vector<InterChiplet::ProcessConfig>& __proc_phase2_cfg_list) {
    // Create synchronize data structure.
    SyncStruct* g_sync_structure = new SyncStruct();

    // Load delay record.
    g_sync_structure->m_delay_list.load_delay("delayInfo.txt",
                                              __proc_phase2_cfg_list[0].m_clock_rate);
    spdlog::info("Load {} delay records.", g_sync_structure->m_delay_list.size());

    // Trace lock record.
    for (auto& pair : g_sync_structure->m_delay_list) {
        if (pair.second.m_desc & InterChiplet::SPD_LOCKER) {
            g_sync_structure->m_lock_delay_list.insert(
                pair.second.m_cycle + pair.second.m_delay_list[1], pair.second);
        }
    }

    // Create multi-thread.
    int thread_i = 0;
    std::vector<ProcessStruct*> proc_struct_list;
    for (auto& proc_cfg : __proc_phase1_cfg_list) {
        ProcessStruct* proc_struct = new ProcessStruct(proc_cfg);
        proc_struct->m_round = __round;
        proc_struct->m_phase = 1;
        proc_struct->m_thread = thread_i;
        proc_struct->m_sync_struct = g_sync_structure;
        int res =
            pthread_create(&(proc_struct->m_thread_id), NULL, bridge_thread, (void*)proc_struct);
        if (res < 0) {
            perror("pthread_create");
            exit(EXIT_FAILURE);
        }

        proc_struct_list.push_back(proc_struct);
        thread_i++;
    }

    // Wait threads to finish.
    for (auto& proc_struct : proc_struct_list) {
        pthread_join(proc_struct->m_thread_id, NULL);
        delete proc_struct;
    }
    spdlog::info("All process has exit.");

    // Remove file.
    for (auto& __name : g_sync_structure->m_fifo_list) {
        remove(__name.c_str());
    }

    // Dump benchmark record.
    g_sync_structure->m_bench_list.dump_bench("bench.txt", __proc_phase2_cfg_list[0].m_clock_rate);
    spdlog::info("Dump {} bench records.", g_sync_structure->m_bench_list.size());

    // Destory global synchronize structure, and return total cycle.
    InterChiplet::InnerTimeType res_cycle = g_sync_structure->m_cycle;
    delete g_sync_structure;
    return res_cycle;
}

void __loop_phase_two(int __round,
                      const std::vector<InterChiplet::ProcessConfig>& __proc_cfg_list) {
    // Create synchronize data structure.
    SyncStruct* g_sync_structure = new SyncStruct();

    // Create multi-thread.
    int thread_i = 0;
    std::vector<ProcessStruct*> proc_struct_list;
    for (auto& proc_cfg : __proc_cfg_list) {
        ProcessStruct* proc_struct = new ProcessStruct(proc_cfg);
        proc_struct->m_round = __round;
        proc_struct->m_phase = 2;
        proc_struct->m_thread = thread_i;
        thread_i++;
        proc_struct->m_sync_struct = g_sync_structure;
        int res =
            pthread_create(&(proc_struct->m_thread_id), NULL, bridge_thread, (void*)proc_struct);
        if (res < 0) {
            perror("pthread_create");
            exit(EXIT_FAILURE);
        }

        proc_struct_list.push_back(proc_struct);
        thread_i++;
    }

    // Wait threads to finish.
    for (auto& proc_struct : proc_struct_list) {
        pthread_join(proc_struct->m_thread_id, NULL);
        delete proc_struct;
    }
    spdlog::info("All process has exit.");

    // Destory global synchronize structure.
    delete g_sync_structure;
}

int main(int argc, const char* argv[]) {
    // Parse command line.
    InterChiplet::CmdLineOptions options;
    if (options.parse(argc, argv) != 0) {
        return 0;
    };

    // Initializate logging
    if (options.m_debug) {
        spdlog::set_level(spdlog::level::debug);
    }
    spdlog::info("==== LegoSim Chiplet Simulator ====");

    // Change working directory if --cwd is specified.
    if (options.m_cwd != "") {
        if (access(options.m_cwd.c_str(), F_OK) == 0) {
            if (chdir(options.m_cwd.c_str()) < 0) {
                perror("chdir");
            }
            spdlog::info("Change working directory {}.", get_current_dir_name());
        }
    }

    // Check exist of benchmark configuration yaml.
    if (access(options.m_bench.c_str(), F_OK) < 0) {
        spdlog::error("Cannot find benchmark {}.", options.m_bench);
        exit(EXIT_FAILURE);
    }

    // Load benchmark configuration.
    InterChiplet::BenchmarkConfig configs(options.m_bench);
    spdlog::info("Load benchmark configuration from {}.", options.m_bench);

    // Get start time of simulation.
    struct timeval simstart, simend, roundstart, roundend;
    gettimeofday(&simstart, 0);

    InterChiplet::InnerTimeType sim_cycle = 0;
    for (int round = 1; round <= options.m_timeout_threshold; round++) {
        // Get start time of one round.
        gettimeofday(&roundstart, 0);
        spdlog::info("**** Round {} Phase 1 ****", round);
        InterChiplet::InnerTimeType round_cycle =
            __loop_phase_one(round, configs.m_phase1_proc_cfg_list, configs.m_phase2_proc_cfg_list);

        // Get simulation cycle.
        // If simulation cycle this round is close to the previous one, quit iteration.
        spdlog::info("Benchmark elapses {} cycle.",
                     static_cast<InterChiplet::TimeType>(round_cycle));
        if (round > 1) {
            // Calculate error of simulation cycle.
            double err_rate = ((double)round_cycle - (double)sim_cycle) / (double)round_cycle;
            err_rate = err_rate < 0 ? -err_rate : err_rate;
            spdlog::info("Difference related to pervious round is {}%.", err_rate * 100);
            // If difference is small enough, quit loop.
            if (err_rate < options.m_err_rate_threshold) {
                spdlog::info("Quit simulation because simulation cycle has converged.");
                sim_cycle = round_cycle;
                break;
            }
        }
        sim_cycle = round_cycle;

        spdlog::info("**** Round {} Phase 2 ***", round);
        __loop_phase_two(round, configs.m_phase2_proc_cfg_list);

        // Get end time of one round.
        gettimeofday(&roundend, 0);
        unsigned long elaped_sec = roundend.tv_sec - roundstart.tv_sec;
        spdlog::info("Round {} elapses {}d {}h {}m {}s.", round, elaped_sec / 3600 / 24,
                     (elaped_sec / 3600) % 24, (elaped_sec / 60) % 60, elaped_sec % 60);
    }

    // Get end time of simulation.
    spdlog::info("**** End of Simulation ****");
    gettimeofday(&simend, 0);
    unsigned long elaped_sec = simend.tv_sec - simstart.tv_sec;
    spdlog::info("Benchmark elapses {} cycle.", static_cast<InterChiplet::TimeType>(sim_cycle));
    spdlog::info("Simulation elapseds {}d {}h {}m {}s.", elaped_sec / 3600 / 24,
                 (elaped_sec / 3600) % 24, (elaped_sec / 60) % 60, elaped_sec % 60);
}
