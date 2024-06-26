
# File apis\_c.h

[**File List**](files.md) **>** [**includes**](dir_943fa6db2bfb09b7dcf1f02346dde40e.md) **>** [**apis\_c.h**](apis__c_8h.md)

[Go to the documentation of this file.](apis__c_8h.md) 

```C++

#pragma once

#include <unistd.h>

#include <cstdint>

namespace InterChiplet {
typedef decltype(syscall(0)) syscall_return_t;

syscall_return_t launch(int64_t __dst_x, int64_t __dst_y, int64_t __src_x, int64_t __src_y);

syscall_return_t waitLaunch(int64_t __dst_x, int64_t __dst_y, int64_t* __src_x, int64_t* __src_y);

syscall_return_t barrier(int64_t __uid, int64_t __src_x, int64_t __src_y, int64_t __count = 0);

syscall_return_t lock(int64_t __uid, int64_t __src_x, int64_t __src_y);

syscall_return_t unlock(int64_t __uid, int64_t __src_x, int64_t __src_y);

syscall_return_t sendMessage(int64_t __dst_x, int64_t __dst_y, int64_t __src_x, int64_t __src_y,
                             void* __addr, int64_t __nbyte);

syscall_return_t receiveMessage(int64_t __dst_x, int64_t __dst_y, int64_t __src_x, int64_t __src_y,
                                void* __addr, int64_t __nbyte);

}  // namespace InterChiplet

```