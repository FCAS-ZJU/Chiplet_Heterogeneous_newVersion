#include "interchiplet_client.h"

#include<cstring>

//与popnet通信部分
namespace nsInterchiplet
{
using namespace std;

const int MSG_LEN = 1024;

static zmq::context_t *zmqContext;
static zmq::socket_t *zmqSocket;
static deque<zmq::message_t> msgBuf;
int connectZmq(const string &addr)
{
    zmqContext = new zmq::context_t(1);
    zmqSocket = new zmq::socket_t(*zmqContext, zmq::socket_type::pair);
    zmqSocket->connect(addr);
    zmqSocket->send(zmq::str_buffer("ready"));
    char buf[MSG_LEN + 1] = {0};
    zmqSocket->recv(buf, MSG_LEN);
    return strcmp(buf, "start") == 0 ? 0 : -1;
}
/* int connectZmq()
{
    return connectZmq(popnetAddr);
}
int disconnectZmq()
{
    zmqSocket->disconnect(popnetAddr);
    return 0;
} */
int closeZmq()
{
    zmqSocket->close();
    delete zmqSocket;
    delete zmqContext;
    return 0;
}
int readAllMsg()
{
    auto ret = zmq::recv_multipart(*zmqSocket, std::back_inserter(msgBuf));
    return (int)*ret;
}
int readMsg(string &str)
{
    if (msgBuf.empty())
        readAllMsg();
    //str = msgBuf.front().to_string() + '\0';
    char buf[MSG_LEN+1]={0};
    memcpy(buf,msgBuf.front().data(),msgBuf.front().size());
    str=buf;
    msgBuf.pop_front();
    return 0;
}
int writeMsg(const string &str)
{
    char tmp[MSG_LEN] = {0};
    strcpy(tmp, str.c_str());
    auto r = zmqSocket->send(zmq::str_buffer(tmp)/* , zmq::send_flags::dontwait */);
    return *r;
}
} // namespace nsInterchiplet
