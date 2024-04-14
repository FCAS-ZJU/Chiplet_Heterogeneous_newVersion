#ifndef NETWORK_INTERFACE_H
#define NETWORK_INTERFACE_H

#include "intercomm.h"

#define PAC_PAYLOAD_BIT 512
#define PAC_PAYLOAD_BYTE (PAC_PAYLOAD_BIT / 8)

long long int getCommEndCycle(const nsInterchiplet::SyncCommand& write_cmd,
                              const nsInterchiplet::SyncCommand& read_cmd)
{
    // TODO: Get more accurate end cycle.
    int pac_size = write_cmd.m_nbytes / PAC_PAYLOAD_BYTE + ((write_cmd.m_nbytes % PAC_PAYLOAD_BYTE) > 0 ? 1 : 0) + 1;

    return (write_cmd.m_cycle >= read_cmd.m_cycle ? write_cmd.m_cycle : read_cmd.m_cycle) + pac_size;
}

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
};

BenchItem syncCommandToBenchItem(const nsInterchiplet::SyncCommand& __cmd)
{
    BenchItem bench_item;
    bench_item.m_cycle = __cmd.m_cycle;
    bench_item.m_dst_x = __cmd.m_dst_x;
    bench_item.m_dst_y = __cmd.m_dst_y;
    bench_item.m_src_x = __cmd.m_src_x;
    bench_item.m_src_y = __cmd.m_src_y;
    bench_item.m_pac_size = __cmd.m_nbytes / PAC_PAYLOAD_BYTE + ((__cmd.m_nbytes % PAC_PAYLOAD_BYTE) > 0 ? 1 : 0) + 1;

    return bench_item;
}

#endif
