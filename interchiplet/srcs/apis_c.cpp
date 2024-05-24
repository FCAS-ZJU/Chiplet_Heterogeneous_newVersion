
#include "global_define.h"
#include "apis_c.h"

namespace InterChiplet
{
    syscall_return_t sendMessage(int64_t __dst_x,
                                    int64_t __dst_y,
                                    int64_t __src_x,
                                    int64_t __src_y,
                                    void* __addr,
                                    int64_t __nbyte)
    {
        int ret_code = syscall(SYSCALL_SEND_TO_GPU,
                                __dst_x,
                                __dst_y,
                                __src_x,
                                __src_y,
                                __addr,
                                __nbyte);
        return ret_code;
    }
    syscall_return_t receiveMessage(int64_t __dst_x,
                                    int64_t __dst_y,
                                    int64_t __src_x,
                                    int64_t __src_y,
                                    void* __addr,
                                    int64_t __nbyte)
    {
        int ret_code = syscall(SYSCALL_READ_FROM_GPU,
                                __dst_x,
                                __dst_y,
                                __src_x,
                                __src_y,
                                __addr,
                                __nbyte);
        return ret_code;
    }
}
