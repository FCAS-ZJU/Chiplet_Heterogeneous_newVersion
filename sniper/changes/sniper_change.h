//在接口中不要使用任何C++的类，尤其是标准库的类
//很可能使用了两套C++标准库

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
} // namespace nsChange
/* 
extern "C"
{
    int connectZmq(const char *addr);
    int disconnectZmq();
    int readAllMsg();
    const char *getBuf(int64_t socket);
    int popBuf(int64_t socket);
}
 */
