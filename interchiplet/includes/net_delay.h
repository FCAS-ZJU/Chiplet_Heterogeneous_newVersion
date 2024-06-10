#pragma once

#include <fstream>
#include <map>

#include "global_define.h"

#define PAC_PAYLOAD_BIT 512
#define PAC_PAYLOAD_BYTE (PAC_PAYLOAD_BIT / 8)

namespace InterChiplet {
/**
 * @brief Structure presents delay of one package in network.
 */
class NetworkDelayItem {
   public:
    /**
     * @brief Package injection cycle. Used to order packages.
     */
    InnerTimeType m_cycle;
    /**
     * @brief Packate id. (Not used yet.)
     */
    uint64_t m_id;
    /**
     * @brief Source address in X-axis.
     */
    int m_src_x;
    /**
     * @brief Source address in Y-axis.
     */
    int m_src_y;
    /**
     * @brief Destination address in X-axis.
     */
    int m_dst_x;
    /**
     * @brief Destination address in Y-axis.
     */
    int m_dst_y;
    /**
     * @brief Synchronization protocol descriptor.
     */
    long m_desc;
    /**
     * @brief Package delay on the source side.
     */
    std::vector<InnerTimeType> m_delay_list;

   public:
    /**
     * @brief Construct Empty NetworkDelayItem.
     */
    NetworkDelayItem() {}

    /**
     * @brief Construct NetworkDelayItem.
     * @param __cycle Package injection cycle.
     * @param __src_x Source address in X-axis.
     * @param __src_y Source address in Y-axis.
     * @param __dst_x Destination address in X-axis.
     * @param __dst_y Destination address in Y-axis.
     * @param __desc Synchronization protocol descriptor.
     * @param __delay_list List of package delays.
     */
    NetworkDelayItem(InnerTimeType __cycle, int __src_x, int __src_y, int __dst_x, int __dst_y,
                     long __desc, const std::vector<InnerTimeType>& __delay_list)
        : m_cycle(__cycle),
          m_dst_x(__dst_x),
          m_dst_y(__dst_y),
          m_src_x(__src_x),
          m_src_y(__src_y),
          m_delay_list(__delay_list) {}

    /**
     * @brief Overloading operator <<.
     *
     * Write NetworkDelayItem to output stream.
     */
    friend std::ostream& operator<<(std::ostream& os, const NetworkDelayItem& __item) {
        os << __item.m_cycle << " " << __item.m_src_x << " " << __item.m_src_y << " "
           << __item.m_dst_x << " " << __item.m_dst_y << " " << __item.m_desc;
        os << " " << __item.m_delay_list.size();
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
        os >> __item.m_cycle >> __item.m_src_x >> __item.m_src_y >> __item.m_dst_x >>
            __item.m_dst_y >> __item.m_desc;
        int delay_cnt = 0;
        os >> delay_cnt;
        for (int i = 0; i < delay_cnt; i++) {
            TimeType delay;
            os >> delay;
            __item.m_delay_list.push_back(delay);
        }
        return os;
    }
};

/**
 * @brief List of network delay item.
 */
class NetworkDelayList : public std::multimap<InnerTimeType, NetworkDelayItem> {
   public:
    /**
     * @brief Construct NetworkDelayList.
     */
    NetworkDelayList() : std::multimap<InnerTimeType, NetworkDelayItem>() {}

    /**
     * @brief Insert item into list.
     */
    void insert(const NetworkDelayItem& __item) {
        std::multimap<InnerTimeType, NetworkDelayItem>::insert(
            std::pair<InnerTimeType, NetworkDelayItem>(__item.m_cycle, __item));
    }

    /**
     * @brief Insert item into list.
     */
    void insert(InnerTimeType __cycle, const NetworkDelayItem& __item) {
        std::multimap<InnerTimeType, NetworkDelayItem>::insert(
            std::pair<InnerTimeType, NetworkDelayItem>(__cycle, __item));
    }

    /**
     * @brief Load package delay list from specified file.
     * @param file_name Path to benchmark file.
     */
    void load_delay(const std::string& __file_name, const double __clock_rate) {
        std::ifstream bench_if(__file_name, std::ios::in);
        while (bench_if) {
            NetworkDelayItem item;
            bench_if >> item;
            if (!bench_if) break;
            item.m_cycle = item.m_cycle / __clock_rate;
            for (std::size_t i = 0; i < item.m_delay_list.size(); i++) {
                item.m_delay_list[i] = item.m_delay_list[i] / __clock_rate;
            }
            insert(item);
        }
    }

    /**
     * @brief Get end cycle of one communication.
     * @param __write_cmd Write command
     * @param __read_cmd  Read command.
     * @return End cycle of this communication, used to acknowledge SYNC command.
     */
    std::tuple<InnerTimeType, InnerTimeType> getEndCycle(
        const InterChiplet::SyncCommand& __write_cmd, const InterChiplet::SyncCommand& __read_cmd) {
        std::multimap<InnerTimeType, NetworkDelayItem>::iterator it = find_first_item(
            __write_cmd.m_src_x, __write_cmd.m_src_y, __write_cmd.m_dst_x, __write_cmd.m_dst_y);

        if (it == end()) {
            return getDefaultEndCycle(__write_cmd, __read_cmd);
        } else {
            erase(it);
            // Locker communication.
            if (__write_cmd.m_desc & SPD_LOCKER) {
                // Forward packet.
                InnerTimeType pac_delay_src = it->second.m_delay_list[0];
                InnerTimeType pac_delay_dst = it->second.m_delay_list[1];
                InnerTimeType write_end_time = __write_cmd.m_cycle + pac_delay_src;
                InnerTimeType read_end_time = __write_cmd.m_cycle + pac_delay_dst;
                if (__read_cmd.m_cycle > read_end_time) {
                    read_end_time = __read_cmd.m_cycle;
                }
                // Acknowledge packet.
                InnerTimeType ack_delay_src = it->second.m_delay_list[2];
                InnerTimeType ack_delay_dst = it->second.m_delay_list[3];
                read_end_time = read_end_time + ack_delay_src;
                write_end_time = read_end_time - ack_delay_src + ack_delay_dst;

                return std::tuple<InnerTimeType, InnerTimeType>(write_end_time, read_end_time);
            }
            // Normal communication.
            else {
                // Forward packet.
                InnerTimeType pac_delay_src = it->second.m_delay_list[0];
                InnerTimeType pac_delay_dst = it->second.m_delay_list[1];
                InnerTimeType write_end_time = __write_cmd.m_cycle + pac_delay_src;
                InnerTimeType read_end_time = __write_cmd.m_cycle + pac_delay_dst;
                if (__read_cmd.m_cycle > read_end_time) {
                    read_end_time = __read_cmd.m_cycle;
                }
                return std::tuple<InnerTimeType, InnerTimeType>(write_end_time, read_end_time);
            }
        }
    }

    /**
     * @brief Get end cycle of one communication.
     * @param __write_cmd Write command
     * @return End cycle of this communication, used to acknowledge SYNC command.
     */
    InnerTimeType getEndCycle(const InterChiplet::SyncCommand& __write_cmd) {
        std::multimap<InnerTimeType, NetworkDelayItem>::iterator it = find_first_item(
            __write_cmd.m_src_x, __write_cmd.m_src_y, __write_cmd.m_dst_x, __write_cmd.m_dst_y);

        if (it == end()) {
            return getDefaultEndCycle(__write_cmd);
        } else {
            erase(it);
            // Forward packet.
            InnerTimeType pac_delay_src = it->second.m_delay_list[0];
            InnerTimeType write_end_time = __write_cmd.m_cycle + pac_delay_src;
            return pac_delay_src;
        }
    }

   public:
    /**
     * @brief Return the pointer to the first item match specified source and destination.
     */
    std::multimap<InnerTimeType, NetworkDelayItem>::iterator find_first_item(int __src_x,
                                                                             int __src_y,
                                                                             int __dst_x,
                                                                             int __dst_y) {
        for (std::multimap<InnerTimeType, NetworkDelayItem>::iterator it = begin(); it != end();
             it++) {
            if (it->second.m_src_x == __src_x && it->second.m_src_y == __src_y &&
                it->second.m_dst_x == __dst_x && it->second.m_dst_y == __dst_y) {
                return it;
            }
        }
        return end();
    }
    /**
     * @brief Return the pointer to the first item match specified source and destination.
     */
    std::multimap<InnerTimeType, NetworkDelayItem>::iterator find_first_item(int __dst_x,
                                                                             int __dst_y) {
        for (std::multimap<InnerTimeType, NetworkDelayItem>::iterator it = begin(); it != end();
             it++) {
            if (it->second.m_dst_x == __dst_x && it->second.m_dst_y == __dst_y) {
                return it;
            }
        }
        return end();
    }

    /**
     * @brief Get end cycle of one communication if cannot find this communication from delay
     * list.
     * @param __write_cmd Write command
     * @return End cycle of this communication, used to acknowledge SYNC command.
     */
    InnerTimeType getDefaultEndCycle(const InterChiplet::SyncCommand& write_cmd) {
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
    std::tuple<InnerTimeType, InnerTimeType> getDefaultEndCycle(
        const InterChiplet::SyncCommand& write_cmd, const InterChiplet::SyncCommand& read_cmd) {
        // TODO: Get more accurate end cycle.
        int pac_size = write_cmd.m_nbytes / PAC_PAYLOAD_BYTE +
                       ((write_cmd.m_nbytes % PAC_PAYLOAD_BYTE) > 0 ? 1 : 0) + 1;

        if (write_cmd.m_cycle >= read_cmd.m_cycle) {
            return std::tuple<InnerTimeType, InnerTimeType>(write_cmd.m_cycle + pac_size,
                                                            write_cmd.m_cycle + pac_size);
        } else {
            return std::tuple<InnerTimeType, InnerTimeType>(read_cmd.m_cycle + pac_size,
                                                            read_cmd.m_cycle + pac_size);
        }
    }
};
}  // namespace InterChiplet
