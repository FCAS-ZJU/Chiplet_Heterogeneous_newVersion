
# File apis\_c.cpp

[**File List**](files.md) **>** [**interchiplet**](dir_a2025b34133129e5724d121abe9a4a4a.md) **>** [**srcs**](dir_b94c70d771af9f161858c2c4e7b3d1c5.md) **>** [**apis\_c.cpp**](apis__c_8cpp.md)

[Go to the documentation of this file.](apis__c_8cpp.md) 

```C++


#include "apis_c.h"

#include "global_define.h"

namespace InterChiplet {
syscall_return_t barrier(int64_t __uid, int64_t __src_x, int64_t __src_y, int64_t __count) {
    int ret_code = syscall(SYSCALL_BARRIER, __uid, __src_x, __src_y, __count);
    return ret_code;
}

syscall_return_t lock(int64_t __uid, int64_t __src_x, int64_t __src_y) {
    int ret_code = syscall(SYSCALL_LOCK, __uid, __src_x, __src_y);
    return ret_code;
}

syscall_return_t unlock(int64_t __uid, int64_t __src_x, int64_t __src_y) {
    int ret_code = syscall(SYSCALL_UNLOCK, __uid, __src_x, __src_y);
    return ret_code;
}

syscall_return_t launch(int64_t __dst_x, int64_t __dst_y, int64_t __src_x,
                                int64_t __src_y) {
    int ret_code = syscall(SYSCALL_LAUNCH, __dst_x, __dst_y, __src_x, __src_y);
    return ret_code;
}

syscall_return_t waitLaunch(int64_t __dst_x, int64_t __dst_y, int64_t* __src_x,
                              int64_t* __src_y) {
    int ret_code = syscall(SYSCALL_WAITLAUNCH, __dst_x, __dst_y, __src_x, __src_y);
    return ret_code;
}

syscall_return_t sendMessage(int64_t __dst_x, int64_t __dst_y, int64_t __src_x, int64_t __src_y,
                             void* __addr, int64_t __nbyte) {
    int ret_code =
        syscall(SYSCALL_REMOTE_WRITE, __dst_x, __dst_y, __src_x, __src_y, __addr, __nbyte);
    return ret_code;
}
syscall_return_t receiveMessage(int64_t __dst_x, int64_t __dst_y, int64_t __src_x, int64_t __src_y,
                                void* __addr, int64_t __nbyte) {
    int ret_code =
        syscall(SYSCALL_REMOTE_READ, __dst_x, __dst_y, __src_x, __src_y, __addr, __nbyte);
    return ret_code;
}
}  // namespace InterChiplet

```