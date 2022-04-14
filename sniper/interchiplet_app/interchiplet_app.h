#include<cstdint>
//#include<tuple>
#include<cstddef>
#include<syscall.h>
#include<unistd.h>

namespace nsInterchiplet
{
    typedef decltype(syscall(0)) syscall_return_t;
    syscall_return_t sendGpuMessage(int64_t dstX,int64_t dstY,int64_t srcX,int64_t srcY,int64_t* a,int64_t size);
    syscall_return_t readGpuMessage(int64_t dstX,int64_t dstY,int64_t srcX,int64_t srcY,int64_t* a, int64_t size);
}
