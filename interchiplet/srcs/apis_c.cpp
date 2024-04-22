
#include "global_define.h"
#include "apis_c.h"

namespace InterChiplet
{
    syscall_return_t sendGpuMessage(int64_t __dst_x,
                                    int64_t __dst_y,
                                    int64_t __src_x,
                                    int64_t __src_y,
                                    int64_t* __addr,
                                    int64_t __size)
    {
        // Convert size to byte.
        int64_t byte_size = __size * sizeof(int64_t);
        int ret_code = syscall(SYSCALL_SEND_TO_GPU,
                                __dst_x,
                                __dst_y,
                                __src_x,
                                __src_y,
                                (void*)__addr,
                                __size * sizeof(int64_t));
        return ret_code;
    }
    syscall_return_t readGpuMessage(int64_t __dst_x,
                                    int64_t __dst_y,
                                    int64_t __src_x,
                                    int64_t __src_y,
                                    int64_t* __addr,
                                    int64_t __size)
    {
        int64_t byte_size = __size * sizeof(int64_t);
        int ret_code = syscall(SYSCALL_READ_FROM_GPU,
                                __dst_x,
                                __dst_y,
                                __src_x,
                                __src_y,
                                (void*)__addr,
                                __size * sizeof(int64_t));
        return ret_code;
    }
}
