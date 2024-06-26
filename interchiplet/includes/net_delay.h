#pragma once

#include <fstream>
#include <map>

#include "global_define.h"
#include "spdlog/spdlog.h"

#define PAC_PAYLOAD_BIT 512
#define PAC_PAYLOAD_BYTE (PAC_PAYLOAD_BIT / 8)

/**
 * @defgroup net_delay
 * @brief Network latency information interface.
 * @{
 */
/**
 * @brief Structure presents delay of one package in network.
 */
class NetworkDelayItem {
   public:
    /**
     * @brief Package injection cycle. Used to order packages.
     */
    InterChiplet::InnerTimeType m_cycle;
    /**
     * @brief Packate id. (Not used yet.)
     */
    uint64_t m_id;
    /**
     * @brief Source address.
     */
    InterChiplet::AddrType m_src;
    /**
     * @brief Destination address.
     */
    InterChiplet::AddrType m_dst;
    /**
     * @brief Synchronization protocol descriptor.
     */
    long m_desc;
    /**
     * @brief Delay of packages.
     *
     * Each package has two delay values. The first value is the delay from the write side, and the
     * second value is the delay from the read side.
     */
    std::vector<InterChiplet::InnerTimeType> m_delay_list;

   public:
    /**
     * @brief Construct Empty NetworkDelayItem.
     */
    NetworkDelayItem() {}

    /**
     * @brief Construct NetworkDelayItem.
     * @param __cycle Package injection cycle.
     * @param __src Source address.
     * @param __dst Destination address.
     * @param __desc Synchronization protocol descriptor.
     * @param __delay_list List of package delays.
     */
    NetworkDelayItem(InterChiplet::InnerTimeType __cycle, const InterChiplet::AddrType& __src,
                     const InterChiplet::AddrType& __dst, long __desc,
                     const std::vector<InterChiplet::InnerTimeType>& __delay_list)
        : m_cycle(__cycle), m_dst(__dst), m_src(__src), m_delay_list(__delay_list) {}

    /**
     * @brief Overloading operator <<.
     *
     * Write NetworkDelayItem to output stream.
     */
    friend std::ostream& operator<<(std::ostream& os, const NetworkDelayItem& __item) {
        os << __item.m_cycle << " " << DIM_X(__item.m_src) << " " << DIM_Y(__item.m_src) << " "
           << DIM_Y(__item.m_dst) << " " << DIM_X(__item.m_dst) << " " << __item.m_desc << " "
           << __item.m_delay_list.size();
        for (auto& delay : __item.m_delay_list) {
            os << " " << delay;
        }
        return os;
    }

    /**
     * @brief Overloading operator >>.
     *
     * Read NetworkDelayItem from input stream.
     */
    friend std::istream& operator>>(std::istream& os, NetworkDelayItem& __item) {
        os >> __item.m_cycle;
        long src_x, src_y, dst_x, dst_y;
        os >> src_x >> src_y >> dst_x >> dst_y;
        __item.m_src.push_back(src_x);
        __item.m_src.push_back(src_y);
        __item.m_dst.push_back(dst_x);
        __item.m_dst.push_back(dst_y);
        os >> __item.m_desc;
        int delay_cnt = 0;
        os >> delay_cnt;
        for (int i = 0; i < delay_cnt; i++) {
            InterChiplet::TimeType delay;
            os >> delay;
            __item.m_delay_list.push_back(delay);
        }
        return os;
    }
};

typedef std::tuple<InterChiplet::InnerTimeType, InterChiplet::InnerTimeType> CmdDelayPair;
#define SRC_DELAY(pair) std::get<0>(pair)
#define DST_DELAY(pair) std::get<1>(pair)
typedef std::multimap<InterChiplet::InnerTimeType, NetworkDelayItem> NetworkDelayOrder;

/**
 * @brief Map for network delay information.
 */
class NetworkDelayMap : public std::map<InterChiplet::AddrType, NetworkDelayOrder> {
   public:
    /**
     * @brief Insert delay information.
     * @param __addr Address as key.
     * @param __cycle Event cycle.
     * @param __item Delay information structure.
     */
    void insert(const InterChiplet::AddrType& __addr, InterChiplet::InnerTimeType __cycle,
                const NetworkDelayItem& __item) {
        if (find(__addr) == end()) {
            (*this)[__addr] = NetworkDelayOrder();
        }
        (*this)[__addr].insert(
            std::pair<InterChiplet::InnerTimeType, NetworkDelayItem>(__cycle, __item));
    }

    /**
     * @brief Check whether there is delay information for the specified address.
     * @param __addr Address.
     * @return If there is delay information for the specified address, return True.
     */
    bool hasAddr(const InterChiplet::AddrType& __addr) {
        // If there is no address, return false.
        if (find(__addr) == end()) {
            return false;
        }
        // If there is no delay information for the address, return false.
        return at(__addr).size() > 0;
    }

    /**
     * @brief Check whether there is delay information for the specified address.
     * @param __addr Address as key.
     * @param __src Source address.
     * @param __dst Destination address.
     * @return If there is delay information for the specified address, return True.
     */
    bool hasAddr(const InterChiplet::AddrType& __addr, const InterChiplet::AddrType& __src,
                 const InterChiplet::AddrType& __dst) {
        // If there is no address, return false.
        if (find(__addr) == end()) {
            return false;
        }
        // If there is no delay information for the address, return false.
        for (NetworkDelayOrder::iterator it = (*this)[__addr].begin(); it != (*this)[__addr].end();
             it++) {
            if (it->second.m_src == __src && it->second.m_dst == __dst) {
                return true;
            }
        }
        return false;
    }

    /**
     * @brief Return the first delay information for the specified address.
     * @param __addr Address.
     * @return The first delay information to the specified address.
     */
    NetworkDelayItem front(const InterChiplet::AddrType& __addr) {
        // If there is no destination address, return empty.
        if (find(__addr) == end()) {
            return NetworkDelayItem();
        }
        // If there is no delay information for the address, return false.
        if (at(__addr).size() == 0) {
            return NetworkDelayItem();
        }

        return at(__addr).begin()->second;
    }

    /**
     * @brief Return the first delay information for the specified address.
     * @param __addr Address as key.
     * @param __src Source address.
     * @param __dst Destination address.
     * @return The first delay information to the specified address.
     */
    NetworkDelayItem front(const InterChiplet::AddrType& __addr,
                           const InterChiplet::AddrType& __src,
                           const InterChiplet::AddrType& __dst) {
        // If there is no destination address, return empty.
        if (find(__addr) == end()) {
            return NetworkDelayItem();
        }
        // If there is no delay information for the address, return false.
        for (NetworkDelayOrder::iterator it = (*this)[__addr].begin(); it != (*this)[__addr].end();
             it++) {
            if (it->second.m_src == __src && it->second.m_dst == __dst) {
                return it->second;
            }
        }
        return NetworkDelayItem();
    }

    /**
     * @brief Pop the first delay information for the specified address.
     * @param __addr Asdress.
     */
    void pop(const InterChiplet::AddrType& __addr) {
        // If there is no destination address, do nothing.
        if (find(__addr) == end()) {
            return;
        }
        // If there is no delay information for the address, do nothing.
        if (at(__addr).size() == 0) {
            return;
        }

        at(__addr).erase(at(__addr).begin());
    }

    /**
     * @brief Pop the first delay information for the specified address.
     * @param __addr Address as key.
     * @param __src Source address.
     * @param __dst Destination address.
     */
    void pop(const InterChiplet::AddrType& __addr, const InterChiplet::AddrType& __src,
             const InterChiplet::AddrType& __dst) {
        // If there is no destination address, do nothing.
        if (find(__addr) == end()) {
            return;
        }
        // If there is no delay information for the address, return false.
        for (NetworkDelayOrder::iterator it = (*this)[__addr].begin(); it != (*this)[__addr].end();
             it++) {
            if (it->second.m_src == __src && it->second.m_dst == __dst) {
                (*this)[__addr].erase(it);
                return;
            }
        }
    }

    /**
     * @brief Check the order of write/read commands.
     * @param __cmd Command to check.
     * @return If the order of command matches the order of delay infomation, return True.
     * Otherwise return False.
     */
    bool checkOrderOfCommand(const InterChiplet::SyncCommand& __cmd) {
        // If the source does not exist, return false.
        if (find(__cmd.m_src) == end()) {
            return true;
        }

        // If the source has no packet, return false.
        if ((*this)[__cmd.m_src].size() == 0) {
            return true;
        }

        NetworkDelayItem& delay_item = (*this)[__cmd.m_src].begin()->second;
        // Return true if command matches the first item in delay information list.
        if (delay_item.m_src == __cmd.m_src && delay_item.m_dst == __cmd.m_dst &&
            delay_item.m_desc == __cmd.m_desc) {
            return true;
        } else {
            spdlog::warn("Delay info from {},{} to {},{} with flag {}.", delay_item.m_src[0],
                         delay_item.m_src[1], delay_item.m_dst[0], delay_item.m_dst[1],
                         delay_item.m_desc);
            spdlog::warn("Command    from {},{} to {},{} with flag {}.", __cmd.m_src[0],
                         __cmd.m_src[1], __cmd.m_dst[0], __cmd.m_dst[1], __cmd.m_desc);
            return false;
        }
    }
};

/**
 * @brief List of network delay item.
 */
class NetworkDelayStruct {
   public:
    NetworkDelayStruct() : m_item_count(0) {}

    int size() const { return m_item_count; }

   public:
    /**
     * @brief Load package delay list from specified file.
     * @param __file_name Path to benchmark file.
     * @param __clock_rate Clock ratio (Simulator clock/Interchiplet clock).
     */
    void loadDelayInfo(const std::string& __file_name, double __clock_rate) {
        std::ifstream bench_if(__file_name, std::ios::in);
        m_item_count = 0;

        while (bench_if) {
            // Load item from file.
            NetworkDelayItem item;
            bench_if >> item;
            if (!bench_if) break;
            item.m_cycle = item.m_cycle / __clock_rate;
            for (std::size_t i = 0; i < item.m_delay_list.size(); i++) {
                item.m_delay_list[i] = item.m_delay_list[i] / __clock_rate;
            }
            m_item_count += 1;

            // Source map.
            m_src_delay_map.insert(item.m_src, item.m_cycle, item);
            // Ordering of barrier, launch, lock and unlock.
            if (item.m_desc & 0xF0000) {
                InterChiplet::InnerTimeType end_cycle = item.m_cycle + item.m_delay_list[1];
                if (item.m_desc & InterChiplet::SPD_BARRIER) {
                    m_barrier_delay_map.insert(item.m_dst, end_cycle, item);
                } else if (item.m_desc & InterChiplet::SPD_LAUNCH) {
                    m_launch_delay_map.insert(item.m_dst, end_cycle, item);
                } else if (item.m_desc & InterChiplet::SPD_LOCK) {
                    m_lock_delay_map.insert(item.m_dst, end_cycle, item);
                } else if (item.m_desc & InterChiplet::SPD_UNLOCK) {
                    m_unlock_delay_map.insert(item.m_dst, end_cycle, item);
                }
            }
        }
    }

    /**
     * @brief Check the order of write/read commands.
     * @param __cmd Command to check.
     * @return If the order of command matches the order of delay infomation, return True.
     * Otherwise return False.
     */
    bool checkOrderOfCommand(const InterChiplet::SyncCommand& __cmd) {
        return m_src_delay_map.checkOrderOfCommand(__cmd);
    }

    /**
     * @brief Clear delay information.
     */
    void clearDelayInfo() {
        // Launch delay list
        m_src_delay_map.clear();
        // Launch order.
        m_launch_delay_map.clear();
        // Barrier order.
        m_barrier_delay_map.clear();
        // Lock order.
        m_lock_delay_map.clear();
        // Unlock order.
        m_unlock_delay_map.clear();
    }

   public:
    /**
     * @brief Check whether there is LAUNCH command in order.
     * @param __dst Destination address.
     * @return If there is LAUNCH command to the specified destination address, return True.
     */
    inline bool hasLaunch(const InterChiplet::AddrType& __dst) {
        return m_launch_delay_map.hasAddr(__dst);
    }

    /**
     * @brief Return the source address of the first LAUNCH command to the specified destination
     * address.
     *
     * @param __dst Destination address.
     * @return The source address of the first LAUNCH command.
     */
    inline InterChiplet::AddrType frontLaunchSrc(const InterChiplet::AddrType& __dst) {
        return m_launch_delay_map.front(__dst).m_src;
    }

    /**
     * @brief Pop the first LAUNCH command to the specified destination address.
     * @param __dst Destination address.
     */
    inline void popLaunch(const InterChiplet::AddrType& __dst) { m_launch_delay_map.pop(__dst); }

    /**
     * @brief Check whether there is LOCK command in order.
     * @param __dst Destination address.
     * @return If there is LOCK command to the specified destination address, return True.
     */
    inline bool hasLock(const InterChiplet::AddrType& __dst) {
        return m_lock_delay_map.hasAddr(__dst);
    }

    /**
     * @brief Return the source address of the first LOCK command to the specified destination
     * address.
     *
     * @param __dst Destination address.
     * @return The source address of the first LOCK command.
     */
    inline InterChiplet::AddrType frontLockSrc(const InterChiplet::AddrType& __dst) {
        return m_lock_delay_map.front(__dst).m_src;
    }

    /**
     * @brief Pop the first LOCK command to the specified destination address.
     * @param __dst Destination address.
     */
    inline void popLock(const InterChiplet::AddrType& __dst) { m_lock_delay_map.pop(__dst); }

   public:
    /**
     * @brief Get end cycle of one communication.
     * @param __write_cmd Write command
     * @param __read_cmd  Read command.
     * @return End cycle of this communication, used to acknowledge SYNC command.
     */
    CmdDelayPair getEndCycle(const InterChiplet::SyncCommand& __write_cmd,
                             const InterChiplet::SyncCommand& __read_cmd) {
        if (!m_src_delay_map.hasAddr(__write_cmd.m_src, __write_cmd.m_src, __write_cmd.m_dst)) {
            return getDefaultEndCycle(__write_cmd, __read_cmd);
        }

        NetworkDelayItem delay_info =
            m_src_delay_map.front(__write_cmd.m_src, __write_cmd.m_src, __write_cmd.m_dst);

        m_src_delay_map.pop(__write_cmd.m_src, __write_cmd.m_src, __write_cmd.m_dst);
        // Launch/Barrier/Lock/Unlock communication.
        if (__write_cmd.m_desc & (InterChiplet::SPD_LAUNCH | InterChiplet::SPD_BARRIER |
                                  InterChiplet::SPD_LOCK | InterChiplet::SPD_UNLOCK)) {
            // Forward packet.
            InterChiplet::InnerTimeType pac_delay_src = delay_info.m_delay_list[0];
            InterChiplet::InnerTimeType pac_delay_dst = delay_info.m_delay_list[1];
            InterChiplet::InnerTimeType write_end_time = __write_cmd.m_cycle + pac_delay_src;
            InterChiplet::InnerTimeType read_end_time = __write_cmd.m_cycle + pac_delay_dst;
            if (__read_cmd.m_cycle > read_end_time) {
                read_end_time = __read_cmd.m_cycle;
            }
            // Acknowledge packet.
            InterChiplet::InnerTimeType ack_delay_src = delay_info.m_delay_list[2];
            InterChiplet::InnerTimeType ack_delay_dst = delay_info.m_delay_list[3];
            read_end_time = read_end_time + ack_delay_src;
            write_end_time = read_end_time - ack_delay_src + ack_delay_dst;

            return CmdDelayPair(write_end_time, read_end_time);
        }
        // Normal communication.
        else {
            // Forward packet.
            InterChiplet::InnerTimeType pac_delay_src = delay_info.m_delay_list[0];
            InterChiplet::InnerTimeType pac_delay_dst = delay_info.m_delay_list[1];
            InterChiplet::InnerTimeType write_end_time = __write_cmd.m_cycle + pac_delay_src;
            InterChiplet::InnerTimeType read_end_time = __write_cmd.m_cycle + pac_delay_dst;
            if (__read_cmd.m_cycle > read_end_time) {
                read_end_time = __read_cmd.m_cycle;
            }
            return CmdDelayPair(write_end_time, read_end_time);
        }
    }

    InterChiplet::InnerTimeType getBarrierCycle(
        const std::vector<InterChiplet::SyncCommand>& barrier_items) {
        InterChiplet::InnerTimeType barrier_cycle = 0;
        for (auto& item : barrier_items) {
            InterChiplet::InnerTimeType t_cycle = 0;
            if (!m_src_delay_map.hasAddr(item.m_src, item.m_src, item.m_dst)) {
                t_cycle = getDefaultEndCycle(item);
            } else {
                NetworkDelayItem delay_info =
                    m_src_delay_map.front(item.m_src, item.m_src, item.m_dst);
                t_cycle = delay_info.m_cycle + delay_info.m_delay_list[1];
            }

            if (t_cycle > barrier_cycle) {
                barrier_cycle = t_cycle;
            }
        }
        return barrier_cycle;
    }

   public:
    /**
     * @brief Get end cycle of one communication if cannot find this communication from delay
     * list.
     * @param __write_cmd Write command
     * @return End cycle of this communication, used to acknowledge SYNC command.
     */
    InterChiplet::InnerTimeType getDefaultEndCycle(const InterChiplet::SyncCommand& write_cmd) {
        // TODO: Get more accurate end cycle.
        int pac_size = write_cmd.m_nbytes / PAC_PAYLOAD_BYTE +
                       ((write_cmd.m_nbytes % PAC_PAYLOAD_BYTE) > 0 ? 1 : 0) + 1;

        return write_cmd.m_cycle + pac_size;
    }
    /**
     * @brief Get end cycle of one communication if cannot find this communication from delay
     * list.
     * @param __write_cmd Write command
     * @param __read_cmd  Read command.
     * @return End cycle of this communication, used to acknowledge SYNC command.
     */
    CmdDelayPair getDefaultEndCycle(const InterChiplet::SyncCommand& write_cmd,
                                    const InterChiplet::SyncCommand& read_cmd) {
        // TODO: Get more accurate end cycle.
        int pac_size = write_cmd.m_nbytes / PAC_PAYLOAD_BYTE +
                       ((write_cmd.m_nbytes % PAC_PAYLOAD_BYTE) > 0 ? 1 : 0) + 1;

        if (write_cmd.m_cycle >= read_cmd.m_cycle) {
            return CmdDelayPair(write_cmd.m_cycle + pac_size, write_cmd.m_cycle + pac_size);
        } else {
            return CmdDelayPair(read_cmd.m_cycle + pac_size, read_cmd.m_cycle + pac_size);
        }
    }

   private:
    int m_item_count;

    /**
     * @brief Launch delay list. Key is the destination address and value is a list of delay item.
     */
    NetworkDelayMap m_src_delay_map;
    /**
     * @brief Launch order. Key is the destination address and value is a list of delay item.
     */
    NetworkDelayMap m_launch_delay_map;
    /**
     * @brief Barrier order. Key is the destination address and value is a list of delay item.
     */
    NetworkDelayMap m_barrier_delay_map;
    /**
     * @brief Lock order. Key is the destination address and value is a list of delay item.
     */
    NetworkDelayMap m_lock_delay_map;
    /**
     * @brief Unlock order. Key is the destination address and value is a list of delay item.
     */
    NetworkDelayMap m_unlock_delay_map;
};
/**
 * @}
 */
