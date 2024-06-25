
# File net\_bench.h

[**File List**](files.md) **>** [**includes**](dir_943fa6db2bfb09b7dcf1f02346dde40e.md) **>** [**net\_bench.h**](net__bench_8h.md)

[Go to the documentation of this file.](net__bench_8h.md) 

```C++

#pragma once

#include <fstream>
#include <map>

#include "global_define.h"

#define PAC_PAYLOAD_BIT 512
#define PAC_PAYLOAD_BYTE (PAC_PAYLOAD_BIT / 8)

namespace InterChiplet {
class NetworkBenchItem {
   public:
    InnerTimeType m_src_cycle;
    InnerTimeType m_dst_cycle;
    uint64_t m_id;
    AddrType m_src;
    AddrType m_dst;
    int m_pac_size;
    long m_desc;

   public:
    NetworkBenchItem() {}

    NetworkBenchItem(const SyncCommand& __src_cmd, const SyncCommand& __dst_cmd)
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

    NetworkBenchItem(const SyncCommand& __src_cmd)
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

    friend std::ostream& operator<<(std::ostream& os, const NetworkBenchItem& __item) {
        os << __item.m_src_cycle << " " << __item.m_dst_cycle << " " << DIM_X(__item.m_src) << " "
           << DIM_Y(__item.m_src) << " " << DIM_X(__item.m_dst) << " " << DIM_Y(__item.m_dst) << " "
           << __item.m_pac_size << " " << __item.m_desc;
        return os;
    }

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

class NetworkBenchList : public std::multimap<InnerTimeType, NetworkBenchItem> {
   public:
    NetworkBenchList() : std::multimap<InnerTimeType, NetworkBenchItem>() {}

    void insert(const NetworkBenchItem& __item) {
        std::multimap<InnerTimeType, NetworkBenchItem>::insert(
            std::pair<InnerTimeType, NetworkBenchItem>(__item.m_src_cycle, __item));
    }

    void dumpBench(const std::string& __file_name, double __clock_rate) {
        std::ofstream bench_of(__file_name, std::ios::out);
        for (auto& it : *this) {
            bench_of << static_cast<TimeType>(it.second.m_src_cycle * __clock_rate) << " "
                     << static_cast<TimeType>(it.second.m_dst_cycle * __clock_rate) << " "
                     << DIM_X(it.second.m_src) << " " << DIM_Y(it.second.m_src) << " "
                     << DIM_X(it.second.m_dst) << " " << DIM_Y(it.second.m_dst) << " "
                     << it.second.m_pac_size << " " << it.second.m_desc << std::endl;
        }
        bench_of.flush();
        bench_of.close();
    }
};
}  // namespace InterChiplet

```