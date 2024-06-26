
# File cmd\_handler.h

[**File List**](files.md) **>** [**includes**](dir_943fa6db2bfb09b7dcf1f02346dde40e.md) **>** [**cmd\_handler.h**](cmd__handler_8h.md)

[Go to the documentation of this file.](cmd__handler_8h.md) 

```C++

#pragma

#include <list>
#include <map>
#include <set>
#include <string>
#include <vector>

#include "net_bench.h"
#include "net_delay.h"
#include "sync_protocol.h"

typedef std::vector<InterChiplet::SyncCommand> SyncCmdList;

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

class SyncPipeStruct {
   private:
    std::set<std::string> m_pipe_set;

   public:
    inline void insert(std::string& __name) { m_pipe_set.insert(__name); }

    inline void insert(const std::string& __name) { m_pipe_set.insert(__name); }

    inline std::set<std::string> pipeSet() { return m_pipe_set; }

    inline const std::set<std::string>& pipeSet() const { return m_pipe_set; }
};

class SyncBarrierStruct {
   private:
    std::map<int, int> m_barrier_count_map;
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

class SyncLockStruct {
   private:
    std::set<int> m_lock_set;
    std::map<int, InterChiplet::SyncCommand> m_last_cmd_map;
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

class SyncLaunchStruct {
   private:
    std::map<InterChiplet::AddrType, SyncCmdList> m_launch_cmd_list;
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

class SyncCommStruct {
   private:
    std::map<InterChiplet::AddrType, SyncCmdList> m_read_cmd_list;
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

class SyncStruct {
   public:
    SyncStruct() {
        if (pthread_mutex_init(&m_mutex, NULL) < 0) {
            perror("pthread_mutex_init");
            exit(EXIT_FAILURE);
        }
    }

    ~SyncStruct() { pthread_mutex_destroy(&m_mutex); }

   public:
    pthread_mutex_t m_mutex;

    NetworkBenchList m_bench_list;
    NetworkDelayStruct m_delay_list;

    SyncClockStruct m_cycle_struct;

    SyncPipeStruct m_pipe_struct;
    SyncBarrierStruct m_barrier_struct;
    SyncLockStruct m_lock_struct;
    SyncLaunchStruct m_launch_struct;

    SyncCommStruct m_comm_struct;
    SyncBarrierStruct m_barrier_timing_struct;
    SyncLockStruct m_lock_timing_struct;
};
void handle_cycle_cmd(const InterChiplet::SyncCommand& __cmd, SyncStruct* __sync_struct);

void handle_pipe_cmd(const InterChiplet::SyncCommand& __cmd, SyncStruct* __sync_struct);

void handle_barrier_cmd(const InterChiplet::SyncCommand& __cmd, SyncStruct* __sync_struct);

void handle_lock_cmd(const InterChiplet::SyncCommand& __cmd, SyncStruct* __sync_struct);

void handle_unlock_cmd(const InterChiplet::SyncCommand& __cmd, SyncStruct* __sync_struct);

void handle_launch_cmd(const InterChiplet::SyncCommand& __cmd, SyncStruct* __sync_struct);

void handle_waitlaunch_cmd(const InterChiplet::SyncCommand& __cmd, SyncStruct* __sync_struct);

void handle_read_cmd(const InterChiplet::SyncCommand& __cmd, SyncStruct* __sync_struct);

void handle_write_cmd(const InterChiplet::SyncCommand& __cmd, SyncStruct* __sync_struct);

```