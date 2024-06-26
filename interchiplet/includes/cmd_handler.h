#pragma

#include <list>
#include <map>
#include <set>
#include <string>
#include <vector>

#include "net_bench.h"
#include "net_delay.h"
#include "sync_protocol.h"

/**
 * @defgroup cmd_handler_struct
 * @brief Structures for Command handler.
 * @{
 */
/**
 * @brief List of synchronization commands.
 */
typedef std::vector<InterChiplet::SyncCommand> SyncCmdList;

/**
 * @brief Structure for Clock synchronization.
 */
class SyncClockStruct {
   private:
    InterChiplet::InnerTimeType m_cycle;

   public:
    SyncClockStruct() : m_cycle(0) {}

    inline void update(InterChiplet::InnerTimeType __cycle) {
        if (m_cycle < __cycle) {
            m_cycle = __cycle;
        }
    }

    inline InterChiplet::InnerTimeType cycle() { return m_cycle; }
};

/**
 * @brief Structure for Pipe synchronization.
 */
class SyncPipeStruct {
   private:
    std::set<std::string> m_pipe_set;

   public:
    inline void insert(std::string& __name) { m_pipe_set.insert(__name); }

    inline void insert(const std::string& __name) { m_pipe_set.insert(__name); }

    inline std::set<std::string> pipeSet() { return m_pipe_set; }

    inline const std::set<std::string>& pipeSet() const { return m_pipe_set; }
};

/**
 * @brief Structure for Barrier synchronization.
 */
class SyncBarrierStruct {
   private:
    /**
     * @brief Barrier count.
     */
    std::map<int, int> m_barrier_count_map;
    /**
     * @brief Barrier items.
     */
    std::map<int, SyncCmdList> m_barrier_items_map;

   public:
    inline void insertBarrier(int __uid, int __count) {
        // Exist barrier
        if (m_barrier_count_map.find(__uid) != m_barrier_count_map.end()) {
            // Update barrier count
            if (__count > 0) {
                m_barrier_count_map[__uid] = __count;
            }
        }
        // New barrier
        else {
            m_barrier_count_map[__uid] = __count;
            m_barrier_items_map[__uid] = SyncCmdList();
        }
    }

    inline void insertBarrier(int __uid, int __count, const InterChiplet::SyncCommand& __cmd) {
        // Exist barrier
        if (m_barrier_count_map.find(__uid) != m_barrier_count_map.end()) {
            // Update barrier count
            if (__count > 0) {
                m_barrier_count_map[__uid] = __count;
            }
        }
        // New barrier
        else {
            m_barrier_count_map[__uid] = __count;
            m_barrier_items_map[__uid] = SyncCmdList();
        }
        m_barrier_items_map[__uid].push_back(__cmd);
    }

    inline bool overflow(int __uid) {
        // New Barrier
        if (m_barrier_count_map.find(__uid) == m_barrier_count_map.end()) {
            return true;
        }
        // Exist barrier
        else {
            return m_barrier_items_map[__uid].size() >= m_barrier_count_map[__uid];
        }
    }

    inline SyncCmdList& barrierCmd(int __uid) { return m_barrier_items_map[__uid]; }

    inline void reset(int __uid) {
        if (m_barrier_count_map.find(__uid) != m_barrier_count_map.end()) {
            m_barrier_items_map[__uid].clear();
        }
    }
};

/**
 * @brief Structure for Lock and Unlock synchronization.
 */
class SyncLockStruct {
   private:
    /**
     * @brief Barrier count.
     */
    std::set<int> m_lock_set;
    /**
     * @brief Last command for each barrier.
     */
    std::map<int, InterChiplet::SyncCommand> m_last_cmd_map;
    /**
     * @brief Barrier items.
     */
    std::map<int, SyncCmdList> m_lock_cmd_list;

   public:
    inline bool isLocked(int __uid) { return m_lock_set.find(__uid) != m_lock_set.end(); }

    inline void lock(int __uid, const InterChiplet::SyncCommand& __cmd) {
        if (!isLocked(__uid)) {
            m_lock_set.insert(__uid);
            m_last_cmd_map[__uid] = __cmd;
        }
    }

    inline void unlock(int __uid, const InterChiplet::SyncCommand& __cmd) {
        if (isLocked(__uid)) {
            m_lock_set.erase(__uid);
            m_last_cmd_map[__uid] = __cmd;
        }
    }

    inline bool hasLastCmd(int __uid) {
        return m_last_cmd_map.find(__uid) != m_last_cmd_map.end();
    }

    inline InterChiplet::SyncCommand getLastCmd(int __uid) {
        if (hasLastCmd(__uid)) {
            return m_last_cmd_map[__uid];
        } else {
            return InterChiplet::SyncCommand();
        }
    }

    inline void insertLockCmd(int __uid, const InterChiplet::SyncCommand& __cmd) {
        if (m_lock_cmd_list.find(__uid) == m_lock_cmd_list.end()) {
            m_lock_cmd_list[__uid] = SyncCmdList();
        }
        m_lock_cmd_list[__uid].push_back(__cmd);
    }

    inline bool hasLockCmd(int __uid) {
        if (m_lock_cmd_list.find(__uid) == m_lock_cmd_list.end()) {
            return false;
        }
        return m_lock_cmd_list[__uid].size() > 0;
    }

    inline bool hasLockCmd(int __uid, const InterChiplet::AddrType& __src) {
        if (m_lock_cmd_list.find(__uid) == m_lock_cmd_list.end()) {
            return false;
        }
        if (m_lock_cmd_list[__uid].size() == 0) {
            return false;
        }
        SyncCmdList& cmd_list = m_lock_cmd_list[__uid];
        for (SyncCmdList::iterator it = cmd_list.begin(); it != cmd_list.end(); it++) {
            if (it->m_src == __src) {
                return true;
            }
        }
        return false;
    }

    inline InterChiplet::SyncCommand popLockCmd(int __uid) {
        if (m_lock_cmd_list.find(__uid) == m_lock_cmd_list.end()) {
            return InterChiplet::SyncCommand();
        }
        if (m_lock_cmd_list[__uid].size() == 0) {
            return InterChiplet::SyncCommand();
        }
        InterChiplet::SyncCommand command = m_lock_cmd_list[__uid].front();
        m_lock_cmd_list[__uid].erase(m_lock_cmd_list[__uid].begin());
        return command;
    }

    inline InterChiplet::SyncCommand popLockCmd(int __uid, const InterChiplet::AddrType& __src) {
        if (m_lock_cmd_list.find(__uid) == m_lock_cmd_list.end()) {
            return InterChiplet::SyncCommand();
        }
        if (m_lock_cmd_list[__uid].size() == 0) {
            return InterChiplet::SyncCommand();
        }
        SyncCmdList& cmd_list = m_lock_cmd_list[__uid];
        for (SyncCmdList::iterator it = cmd_list.begin(); it != cmd_list.end(); it++) {
            if (it->m_src == __src) {
                InterChiplet::SyncCommand cmd = *it;
                cmd_list.erase(it);
                return cmd;
            }
        }
        return InterChiplet::SyncCommand();
    }
};

/**
 * @brief Structure for Launch and Wait-launch synchronization.
 */
class SyncLaunchStruct {
   private:
    /**
     * @brief List of Pending Pipe commands.
     */
    std::map<InterChiplet::AddrType, SyncCmdList> m_launch_cmd_list;
    /**
     * @brief List of Pending Pipe commands.
     */
    std::map<InterChiplet::AddrType, SyncCmdList> m_waitlaunch_cmd_list;

   public:
    inline bool hasMatchWaitlaunch(const InterChiplet::SyncCommand& __cmd) {
        if (m_waitlaunch_cmd_list.find(__cmd.m_dst) == m_waitlaunch_cmd_list.end()) {
            return false;
        }

        SyncCmdList& cmd_list = m_waitlaunch_cmd_list[__cmd.m_dst];

        for (std::size_t i = 0; i < cmd_list.size(); i++) {
            InterChiplet::SyncCommand& __waitlaunch_cmd = cmd_list[i];
            if (UNSPECIFIED_ADDR(__waitlaunch_cmd.m_src)) {
                if (__cmd.m_dst == __waitlaunch_cmd.m_dst) {
                    return true;
                }
            } else {
                if (__cmd.m_src == __waitlaunch_cmd.m_src &&
                    __cmd.m_dst == __waitlaunch_cmd.m_dst) {
                    return true;
                }
            }
        }
        return false;
    }

    inline InterChiplet::SyncCommand popMatchWaitlaunch(const InterChiplet::SyncCommand& __cmd) {
        SyncCmdList& cmd_list = m_waitlaunch_cmd_list[__cmd.m_dst];

        for (std::size_t i = 0; i < cmd_list.size(); i++) {
            InterChiplet::SyncCommand& __waitlaunch_cmd = cmd_list[i];
            if (UNSPECIFIED_ADDR(__waitlaunch_cmd.m_src)) {
                if (__cmd.m_dst == __waitlaunch_cmd.m_dst) {
                    InterChiplet::SyncCommand match_cmd = cmd_list[i];
                    cmd_list.erase(cmd_list.begin() + i);
                    return match_cmd;
                }
            } else {
                if (__cmd.m_src == __waitlaunch_cmd.m_src &&
                    __cmd.m_dst == __waitlaunch_cmd.m_dst) {
                    InterChiplet::SyncCommand match_cmd = cmd_list[i];
                    cmd_list.erase(cmd_list.begin() + i);
                    return match_cmd;
                }
            }
        }
        return InterChiplet::SyncCommand();
    }

    inline void insertWaitlaunch(InterChiplet::SyncCommand& __cmd) {
        if (m_waitlaunch_cmd_list.find(__cmd.m_dst) == m_waitlaunch_cmd_list.end()) {
            m_waitlaunch_cmd_list[__cmd.m_dst] = SyncCmdList();
        }
        m_waitlaunch_cmd_list[__cmd.m_dst].push_back(__cmd);
    }

    inline void insertWaitlaunch(const InterChiplet::SyncCommand& __cmd) {
        if (m_waitlaunch_cmd_list.find(__cmd.m_dst) == m_waitlaunch_cmd_list.end()) {
            m_waitlaunch_cmd_list[__cmd.m_dst] = SyncCmdList();
        }
        m_waitlaunch_cmd_list[__cmd.m_dst].push_back(__cmd);
    }

    inline bool hasMatchLaunch(const InterChiplet::SyncCommand& __cmd) {
        if (m_launch_cmd_list.find(__cmd.m_dst) == m_launch_cmd_list.end()) {
            return false;
        }

        SyncCmdList& cmd_list = m_launch_cmd_list[__cmd.m_dst];

        for (std::size_t i = 0; i < cmd_list.size(); i++) {
            InterChiplet::SyncCommand& __launch_cmd = cmd_list[i];
            if (UNSPECIFIED_ADDR(__cmd.m_src)) {
                if (__cmd.m_dst == __launch_cmd.m_dst) {
                    return true;
                }
            } else {
                if (__cmd.m_src == __launch_cmd.m_src && __cmd.m_dst == __launch_cmd.m_dst) {
                    return true;
                }
            }
        }
        return false;
    }

    inline InterChiplet::SyncCommand popMatchLaunch(const InterChiplet::SyncCommand& __cmd) {
        SyncCmdList& cmd_list = m_launch_cmd_list[__cmd.m_dst];

        for (std::size_t i = 0; i < cmd_list.size(); i++) {
            InterChiplet::SyncCommand& __launch_cmd = cmd_list[i];
            if (UNSPECIFIED_ADDR(__cmd.m_src)) {
                if (__cmd.m_dst == __launch_cmd.m_dst) {
                    InterChiplet::SyncCommand match_cmd = cmd_list[i];
                    cmd_list.erase(cmd_list.begin() + i);
                    return match_cmd;
                }
            } else {
                if (__cmd.m_src == __launch_cmd.m_src && __cmd.m_dst == __launch_cmd.m_dst) {
                    InterChiplet::SyncCommand match_cmd = cmd_list[i];
                    cmd_list.erase(cmd_list.begin() + i);
                    return match_cmd;
                }
            }
        }
        return InterChiplet::SyncCommand();
    }

    inline void insertLaunch(InterChiplet::SyncCommand& __cmd) {
        if (m_launch_cmd_list.find(__cmd.m_dst) == m_launch_cmd_list.end()) {
            m_launch_cmd_list[__cmd.m_dst] = SyncCmdList();
        }
        m_launch_cmd_list[__cmd.m_dst].push_back(__cmd);
    }

    inline void insertLaunch(const InterChiplet::SyncCommand& __cmd) {
        if (m_launch_cmd_list.find(__cmd.m_dst) == m_launch_cmd_list.end()) {
            m_launch_cmd_list[__cmd.m_dst] = SyncCmdList();
        }
        m_launch_cmd_list[__cmd.m_dst].push_back(__cmd);
    }
};

/**
 * @brief Structure for Communication synchronization.
 */
class SyncCommStruct {
   private:
    /**
     * @brief List of Read commands, used to pair with write commands.
     */
    std::map<InterChiplet::AddrType, SyncCmdList> m_read_cmd_list;
    /**
     * @brief List of Write commands, used to pair with write commands.
     */
    std::map<InterChiplet::AddrType, SyncCmdList> m_write_cmd_list;

   public:
    inline bool hasMatchWrite(const InterChiplet::SyncCommand& __cmd) {
        if (m_write_cmd_list.find(__cmd.m_dst) == m_write_cmd_list.end()) {
            return false;
        }

        SyncCmdList& cmd_list = m_write_cmd_list[__cmd.m_dst];

        for (std::size_t i = 0; i < cmd_list.size(); i++) {
            InterChiplet::SyncCommand& __write_cmd = cmd_list[i];
            if (__cmd.m_src == __write_cmd.m_src && __cmd.m_dst == __write_cmd.m_dst &&
                __cmd.m_nbytes == __write_cmd.m_nbytes) {
                return true;
            }
        }
        return false;
    }

    inline InterChiplet::SyncCommand popMatchWrite(const InterChiplet::SyncCommand& __cmd) {
        SyncCmdList& cmd_list = m_write_cmd_list[__cmd.m_dst];

        for (std::size_t i = 0; i < cmd_list.size(); i++) {
            InterChiplet::SyncCommand& __write_cmd = cmd_list[i];
            if (__cmd.m_src == __write_cmd.m_src && __cmd.m_dst == __write_cmd.m_dst &&
                __cmd.m_nbytes == __write_cmd.m_nbytes) {
                InterChiplet::SyncCommand match_cmd = cmd_list[i];
                cmd_list.erase(cmd_list.begin() + i);
                return match_cmd;
            }
        }
        return InterChiplet::SyncCommand();
    }

    inline void insertWrite(InterChiplet::SyncCommand& __cmd) {
        if (m_write_cmd_list.find(__cmd.m_dst) == m_write_cmd_list.end()) {
            m_write_cmd_list[__cmd.m_dst] = std::vector<InterChiplet::SyncCommand>();
        }
        m_write_cmd_list[__cmd.m_dst].push_back(__cmd);
    }

    inline void insertWrite(const InterChiplet::SyncCommand& __cmd) {
        if (m_write_cmd_list.find(__cmd.m_dst) == m_write_cmd_list.end()) {
            m_write_cmd_list[__cmd.m_dst] = std::vector<InterChiplet::SyncCommand>();
        }
        m_write_cmd_list[__cmd.m_dst].push_back(__cmd);
    }

    inline bool hasMatchRead(const InterChiplet::SyncCommand& __cmd) {
        if (m_read_cmd_list.find(__cmd.m_dst) == m_read_cmd_list.end()) {
            return false;
        }

        SyncCmdList& cmd_list = m_read_cmd_list[__cmd.m_dst];

        for (std::size_t i = 0; i < cmd_list.size(); i++) {
            InterChiplet::SyncCommand& __read_cmd = cmd_list[i];
            if (__cmd.m_src == __read_cmd.m_src && __cmd.m_dst == __read_cmd.m_dst &&
                __cmd.m_nbytes == __read_cmd.m_nbytes) {
                return true;
            }
        }
        return false;
    }

    inline InterChiplet::SyncCommand popMatchRead(const InterChiplet::SyncCommand& __cmd) {
        SyncCmdList& cmd_list = m_read_cmd_list[__cmd.m_dst];

        for (std::size_t i = 0; i < cmd_list.size(); i++) {
            InterChiplet::SyncCommand& __read_cmd = cmd_list[i];
            if (__cmd.m_src == __read_cmd.m_src && __cmd.m_dst == __read_cmd.m_dst &&
                __cmd.m_nbytes == __read_cmd.m_nbytes) {
                InterChiplet::SyncCommand match_cmd = cmd_list[i];
                cmd_list.erase(cmd_list.begin() + i);
                return match_cmd;
            }
        }
        return InterChiplet::SyncCommand();
    }

    inline void insertRead(InterChiplet::SyncCommand& __cmd) {
        if (m_read_cmd_list.find(__cmd.m_dst) == m_read_cmd_list.end()) {
            m_read_cmd_list[__cmd.m_dst] = std::vector<InterChiplet::SyncCommand>();
        }
        m_read_cmd_list[__cmd.m_dst].push_back(__cmd);
    }

    inline void insertRead(const InterChiplet::SyncCommand& __cmd) {
        if (m_read_cmd_list.find(__cmd.m_dst) == m_read_cmd_list.end()) {
            m_read_cmd_list[__cmd.m_dst] = std::vector<InterChiplet::SyncCommand>();
        }
        m_read_cmd_list[__cmd.m_dst].push_back(__cmd);
    }
};

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
    SyncStruct() {
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
     * @brief Benchmark list, recording the communication transactions have sent out.
     */
    NetworkBenchList m_bench_list;
    /**
     * @brief Delay list, recording the delay of each communication transactions
     */
    NetworkDelayStruct m_delay_list;

    /**
     * @brief Global simulation cycle, which is the largest notified cycle count.
     */
    SyncClockStruct m_cycle_struct;

    /**
     * @brief Pipe behavior.
     */
    SyncPipeStruct m_pipe_struct;
    /**
     * @brief Barrier behavior.
     */
    SyncBarrierStruct m_barrier_struct;
    /**
     * @brief Lock behavior.
     */
    SyncLockStruct m_lock_struct;
    /**
     * @brief Launch behavior.
     */
    SyncLaunchStruct m_launch_struct;

    /**
     * @brief Communication behavior.
     */
    SyncCommStruct m_comm_struct;
    /**
     * @brief Barrier timing behavior.
     */
    SyncBarrierStruct m_barrier_timing_struct;
    /**
     * @brief Lock behavior.
     */
    SyncLockStruct m_lock_timing_struct;
};
/**
 * @}
 */

/**
 * @defgroup cmd_handler_func
 * @brief Functions to handle commands.
 * @{ 
 */
/**
 * @brief Handle CYCLE command.
 * @param __cmd Command to handle.
 * @param __sync_struct Pointer to global synchronize structure.
 */
void handle_cycle_cmd(const InterChiplet::SyncCommand& __cmd, SyncStruct* __sync_struct);

/**
 * @brief Handle PIPE command.
 * @param __cmd Command to handle.
 * @param __sync_struct Pointer to global synchronize structure.
 */
void handle_pipe_cmd(const InterChiplet::SyncCommand& __cmd, SyncStruct* __sync_struct);

/**
 * @brief Handle BARRIER command.
 * @param __cmd Command to handle.
 * @param __sync_struct Pointer to global synchronize structure.
 */
void handle_barrier_cmd(const InterChiplet::SyncCommand& __cmd, SyncStruct* __sync_struct);

/**
 * @brief Handle LOCK command.
 * @param __cmd Command to handle.
 * @param __sync_struct Pointer to global synchronize structure.
 */
void handle_lock_cmd(const InterChiplet::SyncCommand& __cmd, SyncStruct* __sync_struct);

/**
 * @brief Handle UNLOCK command.
 * @param __cmd Command to handle.
 * @param __sync_struct Pointer to global synchronize structure.
 */
void handle_unlock_cmd(const InterChiplet::SyncCommand& __cmd, SyncStruct* __sync_struct);

/**
 * @brief Handle LAUNCH command.
 * @param __cmd Command to handle.
 * @param __sync_struct Pointer to global synchronize structure.
 */
void handle_launch_cmd(const InterChiplet::SyncCommand& __cmd, SyncStruct* __sync_struct);

/**
 * @brief Handle WAITLAUNCH command.
 * @param __cmd Command to handle.
 * @param __sync_struct Pointer to global synchronize structure.
 */
void handle_waitlaunch_cmd(const InterChiplet::SyncCommand& __cmd, SyncStruct* __sync_struct);

/**
 * @brief Handle READ command.
 * @param __cmd Command to handle.
 * @param __sync_struct Pointer to global synchronize structure.
 */
void handle_read_cmd(const InterChiplet::SyncCommand& __cmd, SyncStruct* __sync_struct);

/**
 * @brief Handle WRITE command.
 * @param __cmd Command to handle.
 * @param __sync_struct Pointer to global synchronize structure.
 */
void handle_write_cmd(const InterChiplet::SyncCommand& __cmd, SyncStruct* __sync_struct);
/**
 * @}
 */
