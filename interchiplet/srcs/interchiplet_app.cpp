#include"interchiplet_app.h"

//#include"interchiplet_app_core.h"
#include"sniper_change.h"

#include<syscall.h>
#include<unistd.h>
#include<mutex>
#include<thread>

using namespace std;
using namespace nsChange;
namespace nsInterchiplet
{
    static mutex mtx;
    syscall_return_t sendGpuMessage(int64_t dstX,int64_t dstY,int64_t srcX,int64_t srcY,int64_t *a, int64_t size)
    {
		return syscall(SYSCALL_SEND_TO_GPU,dstX,dstY,srcX,srcY,a,size);
    }
    syscall_return_t readGpuMessage(int64_t dstX,int64_t dstY,int64_t srcX,int64_t srcY,int64_t *a, int64_t size)
    {
        return syscall(SYSCALL_READ_FROM_GPU,dstX,dstY,srcX,srcY,a,size);
    }
}
