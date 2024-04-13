#ifndef NETWORK_INTERFACE_H
#define NETWORK_INTERFACE_H

#include "intercomm.h"

long long int getCommEndCycle(const nsInterchiplet::SyncCommand& write_cmd,
                              const nsInterchiplet::SyncCommand& read_cmd)
{
    // TODO: Get more accurate end cycle.
    return write_cmd.m_cycle >= read_cmd.m_cycle ? write_cmd.m_cycle : read_cmd.m_cycle;
}

#endif
