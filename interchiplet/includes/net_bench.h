#pragma once

#include <fstream>
#include <map>

#include "global_define.h"

#define PAC_PAYLOAD_BIT 512
#define PAC_PAYLOAD_BYTE (PAC_PAYLOAD_BIT / 8)

/**
 * @defgroup net_bench
 * @brief Network benchmark interface.
 * @{
 */
/**
 * @brief Structure of one package in network.
 */
class NetworkBenchItem {
   public:
    /**
     * @brief Package injection cycle from the source side.
     */
    InterChiplet::InnerTimeType m_src_cycle;
    /**
     * @brief Package injection cycle from the destination side.
     */
    InterChiplet::InnerTimeType m_dst_cycle;
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
    NetworkBenchItem() {}

    /**
     * @brief Construct NetworkBenchItem from SyncCommand.
     * @param __src_cmd Structure of source command.
     * @param __dst_cmd Structure of destination command.
     */
    NetworkBenchItem(const InterChiplet::SyncCommand& __src_cmd,
                     const InterChiplet::SyncCommand& __dst_cmd)
        : m_src_cycle(__src_cmd.m_cycle),
          m_dst_cycle(__dst_cmd.m_cycle),
          m_dst(__src_cmd.m_dst),
          m_src(__src_cmd.m_src),
          m_pac_size(1),
          m_desc(__src_cmd.m_desc | __dst_cmd.m_desc) {
        // Calculate the number of flit.
        // One head flit is required any way.
        m_pac_size = __src_cmd.m_nbytes / PAC_PAYLOAD_BYTE +
                     ((__src_cmd.m_nbytes % PAC_PAYLOAD_BYTE) > 0 ? 1 : 0) + 1;
    }

    /**
     * @brief Construct NetworkBenchItem from SyncCommand.
     * @param __src_cmd Structure of source command.
     */
    NetworkBenchItem(const InterChiplet::SyncCommand& __src_cmd)
        : m_src_cycle(__src_cmd.m_cycle),
          m_dst_cycle(__src_cmd.m_cycle),
          m_dst(__src_cmd.m_dst),
          m_src(__src_cmd.m_src),
          m_pac_size(1),
          m_desc(__src_cmd.m_desc) {
        // Calculate the number of flit.
        // One head flit is required any way.
        m_pac_size = __src_cmd.m_nbytes / PAC_PAYLOAD_BYTE +
                     ((__src_cmd.m_nbytes % PAC_PAYLOAD_BYTE) > 0 ? 1 : 0) + 1;
    }

    /**
     * @brief Overloading operator <<.
     *
     * Write NetworkBenchItem to output stream.
     */
    friend std::ostream& operator<<(std::ostream& os, const NetworkBenchItem& __item) {
        os << __item.m_src_cycle << " " << __item.m_dst_cycle << " " << DIM_X(__item.m_src) << " "
           << DIM_Y(__item.m_src) << " " << DIM_X(__item.m_dst) << " " << DIM_Y(__item.m_dst) << " "
           << __item.m_pac_size << " " << __item.m_desc;
        return os;
    }

    /**
     * @brief Overloading operator >>.
     *
     * Read NetworkBenchItem from input stream.
     */
    friend std::istream& operator>>(std::istream& os, NetworkBenchItem& __item) {
        os >> __item.m_src_cycle >> __item.m_dst_cycle;
        long src_x, src_y, dst_x, dst_y;
        os >> src_x >> src_y >> dst_x >> dst_y;
        __item.m_src.push_back(src_x);
        __item.m_src.push_back(src_y);
        __item.m_dst.push_back(dst_x);
        __item.m_dst.push_back(dst_y);
        os >> __item.m_pac_size >> __item.m_desc;
        return os;
    }
};

/**
 * @brief List of network benchmark item.
 */
class NetworkBenchList : public std::multimap<InterChiplet::InnerTimeType, NetworkBenchItem> {
   public:
    /**
     * @brief Construct NetworkBenchList.
     */
    NetworkBenchList() : std::multimap<InterChiplet::InnerTimeType, NetworkBenchItem>() {}

    /**
     * @brief Insert item into list.
     *
     * Take the start cycle on source side as ordering key.
     */
    void insert(const NetworkBenchItem& __item) {
        std::multimap<InterChiplet::InnerTimeType, NetworkBenchItem>::insert(
            std::pair<InterChiplet::InnerTimeType, NetworkBenchItem>(__item.m_src_cycle, __item));
    }

    /**
     * @brief Dump benchmark list to specified file.
     * @param __file_name Path to benchmark file.
     * @param __clock_rate Clock ratio (Simulator clock/Interchiplet clock).
     */
    void dumpBench(const std::string& __file_name, double __clock_rate) {
        std::ofstream bench_of(__file_name, std::ios::out);
        for (auto& it : *this) {
            bench_of << static_cast<InterChiplet::TimeType>(it.second.m_src_cycle * __clock_rate)
                     << " "
                     << static_cast<InterChiplet::TimeType>(it.second.m_dst_cycle * __clock_rate)
                     << " " << DIM_X(it.second.m_src) << " " << DIM_Y(it.second.m_src) << " "
                     << DIM_X(it.second.m_dst) << " " << DIM_Y(it.second.m_dst) << " "
                     << it.second.m_pac_size << " " << it.second.m_desc << std::endl;
        }
        bench_of.flush();
        bench_of.close();
    }
};
/**
 * @}
 */
