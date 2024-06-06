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
         * @brief Package injection cycle from the source side.
         */
        InnerTimeType m_src_cycle;
        /**
         * @brief Package injection cycle from the destination side.
         */
        InnerTimeType m_dst_cycle;
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
        /**
         * @brief Synchronization protocol descriptor.
         */
        long m_desc;

    public:
        /**
         * @brief Construct Empty NetworkBenchItem.
         */
        NetworkBenchItem()
        {}

        /**
         * @brief Construct NetworkBenchItem.
         * @param __src_cycle Package injection cycle from the source side.
         * @param __dst_cycle Package injection cycle from the destination side.
         * @param __src_x Source address in X-axis.
         * @param __src_y Source address in Y-axis.
         * @param __dst_x Destination address in X-axis.
         * @param __dst_y Destination address in Y-axis.
         * @param __pac_size Size of package in bytes.
         * @param __desc Synchronization protocol descriptor.
         */
        NetworkBenchItem(TimeType __src_cycle,
                         TimeType __dst_cycle,
                         int __src_x,
                         int __src_y,
                         int __dst_x,
                         int __dst_y,
                         int __pac_size,
                         long __desc)
            : m_src_cycle(__src_cycle)
            , m_dst_cycle(__dst_cycle)
            , m_dst_x(__dst_x)
            , m_dst_y(__dst_y)
            , m_src_x(__src_x)
            , m_src_y(__src_y)
            , m_pac_size(__pac_size)
            , m_desc(__desc)
        {}

        /**
         * @brief Construct NetworkBenchItem from SyncCommand.
         * @param __src_cmd Structure of source command.
         * @param __dst_cmd Structure of destination command.
         */
        NetworkBenchItem(const InterChiplet::SyncCommand& __src_cmd,
                         const InterChiplet::SyncCommand& __dst_cmd)
            : m_src_cycle(__src_cmd.m_cycle)
            , m_dst_cycle(__dst_cmd.m_cycle)
            , m_dst_x(__src_cmd.m_dst_x)
            , m_dst_y(__src_cmd.m_dst_y)
            , m_src_x(__src_cmd.m_src_x)
            , m_src_y(__src_cmd.m_src_y)
            , m_pac_size(1)
            , m_desc(__src_cmd.m_desc | __dst_cmd.m_desc)
        {
            // Calculate the number of flit. 
            // One head flit is required any way.
            m_pac_size = __src_cmd.m_nbytes / PAC_PAYLOAD_BYTE
                + ((__src_cmd.m_nbytes % PAC_PAYLOAD_BYTE) > 0 ? 1 : 0) + 1;
        }

        /**
         * @brief Overloading operator <<.
         * 
         * Write NetworkBenchItem to output stream.
         */
        friend std::ostream& operator<<(std::ostream& os, const NetworkBenchItem& __item)
        {
            os << __item.m_src_cycle << " " << __item.m_dst_cycle << " "
                << __item.m_src_x << " " << __item.m_src_y << " "
                << __item.m_dst_x << " " << __item.m_dst_y << " "
                << __item.m_pac_size << " " << __item.m_desc;
            return os;
        }

        /**
         * @brief Overloading operator >>.
         * 
         * Read NetworkBenchItem from input stream.
         */
        friend std::istream& operator>>(std::istream& os, NetworkBenchItem& __item)
        {
            os >> __item.m_src_cycle >> __item.m_dst_cycle
                >> __item.m_src_x >> __item.m_src_y
                >> __item.m_dst_x >> __item.m_dst_y
                >> __item.m_pac_size >> __item.m_desc;
            return os;
        }
    };

    /**
     * @brief List of network benchmark item.
     */
    class NetworkBenchList
        : public std::multimap<InnerTimeType, NetworkBenchItem>
    {
    public:
        /**
         * @brief Construct NetworkBenchList.
         */
        NetworkBenchList()
            : std::multimap<InnerTimeType, NetworkBenchItem>()
        {}

        /**
         * @brief Insert item into list.
         */
        void insert(const NetworkBenchItem& __item)
        {
            std::multimap<InnerTimeType, NetworkBenchItem>::insert(
                std::pair<InnerTimeType, NetworkBenchItem>(__item.m_src_cycle, __item));
        }

        /**
         * @brief Dump benchmark list to specified file.
         * @param file_name Path to benchmark file.
         */
        void dump_bench(const std::string& __file_name, const double __clock_rate)
        {
            std::ofstream bench_of(__file_name, std::ios::out);
            for (auto& it: *this)
            {
                bench_of << static_cast<TimeType>(it.second.m_src_cycle * __clock_rate) << " "
                    << static_cast<TimeType>(it.second.m_dst_cycle * __clock_rate) << " "
                    << it.second.m_src_x << " " << it.second.m_src_y << " "
                    << it.second.m_dst_x << " " << it.second.m_dst_y << " "
                    << it.second.m_pac_size << " " << it.second.m_desc << std::endl;
            }
            bench_of.flush();
            bench_of.close();
        }
    };
}
