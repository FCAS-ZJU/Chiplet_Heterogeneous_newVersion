#include <zmq.hpp>
#include <zmq_addon.hpp>
#include<deque>
#include<string>

namespace nsInterchiplet
{
    extern const int MSG_LEN;

    int connectZmq(const std::string &addr);
    int closeZmq();
    int readMsg(std::string &str);
    int writeMsg(const std::string &str);
}
