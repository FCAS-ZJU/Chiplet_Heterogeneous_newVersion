#include <cstdint>
#include <cstddef>

namespace nsChange
{
    static const int SYSCALL_TEST_CHANGE = 500;  //测试
    static const int SYSCALL_REMOTE_READ = 501;  //跨芯粒读
    static const int SYSCALL_REMOTE_WRITE = 502; //跨芯粒写
    static const int SYSCALL_REG_FUNC = 503;     //将函数地址传给pin（已弃用）
    static const int SYSCALL_CONNECT = 504;      //建立连接
    static const int SYSCALL_DISCONNECT = 505;   //中断连接
    static const int SYSCALL_GET_LOCAL_ADDR=506;//获得本机地址
    constexpr int SYSCALL_CHECK_REMOTE_READ=507;//检查跨芯粒读取是否有数据
    static const int SYSCALL_SEND_TO_GPU = 508;
    static const int SYSCALL_READ_FROM_GPU = 509;
}
