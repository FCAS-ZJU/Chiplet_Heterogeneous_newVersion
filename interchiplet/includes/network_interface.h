#ifndef NETWORK_INTERFACE_H
#define NETWORK_INTERFACE_H

#include "intercomm.h"

#define PAC_PAYLOAD_BIT 512
#define PAC_PAYLOAD_BYTE (PAC_PAYLOAD_BIT / 8)

class BenchItem
{
public:
    long long int m_cycle;
    long long int m_id;
    int m_src_x;
    int m_src_y;
    int m_dst_x;
    int m_dst_y;
    int m_pac_size;
    double m_delay;

public:
    BenchItem(long long int __cycle, int __src_x, int __src_y, int __dst_x, int __dst_y, int __pac_size)
        : m_cycle(__cycle)
        , m_dst_x(__dst_x)
        , m_dst_y(__dst_y)
        , m_src_x(__src_x)
        , m_src_y(__src_y)
        , m_pac_size(__pac_size)
    {}

    BenchItem(const nsInterchiplet::SyncCommand& __cmd)
        : m_cycle(__cmd.m_cycle)
        , m_dst_x(__cmd.m_dst_x)
        , m_dst_y(__cmd.m_dst_y)
        , m_src_x(__cmd.m_src_x)
        , m_src_y(__cmd.m_src_y)
        , m_pac_size(__cmd.m_nbytes / PAC_PAYLOAD_BYTE + ((__cmd.m_nbytes % PAC_PAYLOAD_BYTE) > 0 ? 1 : 0) + 1)
    {}
};

class BenchList
    : public std::multimap<long long int, BenchItem>
{
public:
    BenchList()
        : std::multimap<long long int, BenchItem>()
    {}

    void insert(const BenchItem& item)
    {
        std::multimap<long long int, BenchItem>::insert(
            std::pair<long long int, BenchItem>(item.m_cycle, item));
    }

    void dump_bench()
    {
        std::ofstream bench_of("bench.txt", std::ios::out);
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

    void load_delay()
    {
        std::ifstream bench_if("delayInfo.txt", std::ios::in);
        while (bench_if)
        {
            long long int cycle;
            int src_x;
            int src_y;
            int dst_x;
            int dst_y;
            long long int delay;
            bench_if >> cycle >> src_x >> src_y >> dst_x >> dst_y >> delay;

            if (!bench_if) break;

            BenchItem item(cycle, src_x, src_y, dst_x, dst_y, 0);
            item.m_delay = delay;
            insert(item);
        }
    }

    long long int getEndCycle(const nsInterchiplet::SyncCommand& write_cmd,
                              const nsInterchiplet::SyncCommand& read_cmd)
    {
        std::multimap<long long int, BenchItem>::iterator it = find_first_item(
            write_cmd.m_src_x, write_cmd.m_src_y, write_cmd.m_dst_x, write_cmd.m_dst_y);

        if (it == end())
        {
            return getDefaultEndCycle(write_cmd, read_cmd);
        }
        else
        {
            long long int delay = it->second.m_delay;
            long long int end_time = (write_cmd.m_cycle >= read_cmd.m_cycle ? write_cmd.m_cycle : read_cmd.m_cycle) + delay;

            erase(it);
            return end_time;
        }
    }

private:
    std::multimap<long long int, BenchItem>::iterator find_first_item(
        int __src_x, int __src_y, int __dst_x, int __dst_y)
    {
        for (std::multimap<long long int, BenchItem>::iterator it = begin(); it != end(); it ++)
        {
            if (it->second.m_src_x == __src_x && it->second.m_src_y == __src_y
                && it->second.m_dst_x == __dst_x && it->second.m_dst_y == __dst_y)
            {

                return it;
            }
        }
        return end();
    }

    long long int getDefaultEndCycle(const nsInterchiplet::SyncCommand& write_cmd,
                                     const nsInterchiplet::SyncCommand& read_cmd)
    {
        // TODO: Get more accurate end cycle.
        int pac_size = write_cmd.m_nbytes / PAC_PAYLOAD_BYTE + ((write_cmd.m_nbytes % PAC_PAYLOAD_BYTE) > 0 ? 1 : 0) + 1;

        return (write_cmd.m_cycle >= read_cmd.m_cycle ? write_cmd.m_cycle : read_cmd.m_cycle) + pac_size;
    }
};

#endif
