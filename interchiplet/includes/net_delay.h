#pragma once

#include <map>
#include <fstream>

#include "global_define.h"

#define PAC_PAYLOAD_BIT 512
#define PAC_PAYLOAD_BYTE (PAC_PAYLOAD_BIT / 8)

namespace InterChiplet
{
    /**
     * @brief Structure of one package in network.
     */
    class NetworkBenchItem
    {
    public:
        /**
         * @brief Package injection cycle.
         */
        TimeType m_cycle;
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
         * @brief Size of package in bytes.
         */
        int m_pac_size;

    public:
        /**
         * @brief Construct NetworkBenchItem.
         * @param __cycle Package injection cycle.
         * @param __src_x Source address in X-axis.
         * @param __src_y Source address in Y-axis.
         * @param __dst_x Destination address in X-axis.
         * @param __dst_y Destination address in Y-axis.
         * @param __pac_size Size of package in bytes.
         */
        NetworkBenchItem(
            TimeType __cycle, int __src_x, int __src_y, int __dst_x, int __dst_y, int __pac_size)
            : m_cycle(__cycle)
            , m_dst_x(__dst_x)
            , m_dst_y(__dst_y)
            , m_src_x(__src_x)
            , m_src_y(__src_y)
            , m_pac_size(__pac_size)
        {}

        /**
         * @brief Construct NetworkBenchItem from SyncCommand.
         * @param __cmd Structure of synchronization command.
         */
        NetworkBenchItem(const InterChiplet::SyncCommand& __cmd)
            : m_cycle(__cmd.m_cycle)
            , m_dst_x(__cmd.m_dst_x)
            , m_dst_y(__cmd.m_dst_y)
            , m_src_x(__cmd.m_src_x)
            , m_src_y(__cmd.m_src_y)
            , m_pac_size(1)
        {
            // Calculate the number of flit. 
            // One head flit is required any way.
            m_pac_size = __cmd.m_nbytes / PAC_PAYLOAD_BYTE
                + ((__cmd.m_nbytes % PAC_PAYLOAD_BYTE) > 0 ? 1 : 0) + 1;
        }
    };

    /**
     * @brief List of network benchmark item.
     */
    class NetworkBenchList
        : public std::multimap<TimeType, NetworkBenchItem>
    {
    public:
        /**
         * @brief Construct NetworkBenchList.
         */
        NetworkBenchList()
            : std::multimap<TimeType, NetworkBenchItem>()
        {}

        /**
         * @brief Insert item into list.
         */
        void insert(const NetworkBenchItem& __item)
        {
            std::multimap<TimeType, NetworkBenchItem>::insert(
                std::pair<TimeType, NetworkBenchItem>(__item.m_cycle, __item));
        }

        /**
         * @brief Dump benchmark list to specified file.
         * @param file_name Path to benchmark file.
         */
        void dump_bench(const std::string& __file_name)
        {
            std::ofstream bench_of(__file_name, std::ios::out);
            for (auto& it: *this)
            {
                bench_of << it.second.m_cycle << " "
                    << it.second.m_src_x << " " << it.second.m_src_y << " "
                    << it.second.m_dst_x << " " << it.second.m_dst_y << " "
                    << it.second.m_pac_size << std::endl;
            }
            bench_of.flush();
            bench_of.close();
        }
    };

    /**
     * @brief Structure presents delay of one package in network.
     */
    class NetworkDelayItem
    {
    public:
        /**
         * @brief Package injection cycle. Used to order packages.
         */
        TimeType m_cycle;
        /**
         * @brief Packate id. (Not used yet.)
         */
        TimeType m_id;
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
         * @brief Package delay: from injection of the first flit to ejection of the last flit.
         */
        TimeType m_delay;

    public:
        /**
         * @brief Construct NetworkDelayItem.
         * @param __cycle Package injection cycle.
         * @param __src_x Source address in X-axis.
         * @param __src_y Source address in Y-axis.
         * @param __dst_x Destination address in X-axis.
         * @param __dst_y Destination address in Y-axis.
         * @param __delay Package delay.
         */
        NetworkDelayItem(
            TimeType __cycle, int __src_x, int __src_y, int __dst_x, int __dst_y, int __delay)
            : m_cycle(__cycle)
            , m_dst_x(__dst_x)
            , m_dst_y(__dst_y)
            , m_src_x(__src_x)
            , m_src_y(__src_y)
            , m_delay(__delay)
        {}
    };

    /**
     * @brief List of network delay item.
     */
    class NetworkDelayList
        : public std::multimap<TimeType, NetworkDelayItem>
    {
    public:
        /**
         * @brief Construct NetworkDelayList.
         */
        NetworkDelayList()
            : std::multimap<TimeType, NetworkDelayItem>()
        {}

        /**
         * @brief Insert item into list.
         */
        void insert(const NetworkDelayItem& __item)
        {
            std::multimap<TimeType, NetworkDelayItem>::insert(
                std::pair<TimeType, NetworkDelayItem>(__item.m_cycle, __item));
        }

        /**
         * @brief Load package delay list from specified file.
         * @param file_name Path to benchmark file.
         */
        void load_delay(const std::string& __file_name)
        {
            std::ifstream bench_if(__file_name, std::ios::in);
            while (bench_if)
            {
                TimeType cycle;
                int src_x;
                int src_y;
                int dst_x;
                int dst_y;
                TimeType delay;
                bench_if >> cycle >> src_x >> src_y >> dst_x >> dst_y >> delay;

                if (!bench_if) break;

                NetworkDelayItem item(cycle, src_x, src_y, dst_x, dst_y, delay);
                insert(item);
            }
        }

        /**
         * @brief Get end cycle of one communication.
         * @param __write_cmd Write command 
         * @param __read_cmd  Read command.
         * @return End cycle of this communication, used to acknowledge SYNC command.
         */
        TimeType getEndCycle(const InterChiplet::SyncCommand& __write_cmd,
                             const InterChiplet::SyncCommand& __read_cmd)
        {
            std::multimap<TimeType, NetworkDelayItem>::iterator it = find_first_item(
                __write_cmd.m_src_x, __write_cmd.m_src_y, __write_cmd.m_dst_x, __write_cmd.m_dst_y);

            if (it == end())
            {
                return getDefaultEndCycle(__write_cmd, __read_cmd);
            }
            else
            {
                TimeType delay = it->second.m_delay;
                erase(it);

                if (__write_cmd.m_cycle >= __read_cmd.m_cycle)
                {
                    return __write_cmd.m_cycle + delay;
                }
                else
                {
                    return __read_cmd.m_cycle + delay;
                }
            }
        }

    private:
        /**
         * @brief Return the pointer to the first item match specified source and destination.
         */
        std::multimap<TimeType, NetworkDelayItem>::iterator find_first_item(
            int __src_x, int __src_y, int __dst_x, int __dst_y)
        {
            for (std::multimap<TimeType, NetworkDelayItem>::iterator it = begin();
                    it != end(); it ++)
            {
                if (it->second.m_src_x == __src_x && it->second.m_src_y == __src_y
                    && it->second.m_dst_x == __dst_x && it->second.m_dst_y == __dst_y)
                {

                    return it;
                }
            }
            return end();
        }

        /**
         * @brief Get end cycle of one communication if cannot find this communication from delay
         * list.
         * @param __write_cmd Write command 
         * @param __read_cmd  Read command.
         * @return End cycle of this communication, used to acknowledge SYNC command.
         */
        TimeType getDefaultEndCycle(const InterChiplet::SyncCommand& write_cmd,
                                    const InterChiplet::SyncCommand& read_cmd)
        {
            // TODO: Get more accurate end cycle.
            int pac_size = write_cmd.m_nbytes / PAC_PAYLOAD_BYTE
                + ((write_cmd.m_nbytes % PAC_PAYLOAD_BYTE) > 0 ? 1 : 0) + 1;

            return (write_cmd.m_cycle >= read_cmd.m_cycle ? write_cmd.m_cycle : read_cmd.m_cycle)
                + pac_size;
        }
    };
}
