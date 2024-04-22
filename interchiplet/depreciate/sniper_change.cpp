#include "sniper_change.h"

#include"zmq.hpp"
#include"zmq_addon.hpp"

#include <utility>
#include <list>
#include <atomic>
std::atomic_int64_t socketNumber(0);
using namespace std;
namespace nsChange
{
   //
}

typedef std::unordered_map<int64_t, std::list<std::string>> bufMap_t;
bufMap_t buf;
zmq::context_t zmqContext;
zmq::socket_t zmqSocket;
extern "C" int connectZmq(const char *addr)
{
   zmqSocket = zmq::socket_t(zmqContext, zmq::socket_type::pair);
   zmqSocket.bind(addr);
   return 0;
}
extern "C" int disconnectZmq()
{
   zmqSocket.close();
   return 0;
}
extern "C" int readAllMsg()
{
   std::vector<zmq::message_t> msgBuf;
   auto ret = zmq::recv_multipart(zmqSocket, std::back_inserter(msgBuf));
   for (auto &msg : msgBuf)
   {
      //将消息存入不同socket的buf上
   }
   return (int)*ret;
}
extern "C" const char *getBuf(int64_t socket)
{
   bufMap_t::iterator it = buf.find(socket);
   if (it == buf.end())
      return NULL;
   else if (it->second.empty())
      return NULL;
   else
      return it->second.front().c_str();
}
extern "C" int popBuf(int64_t socket)
{
   bufMap_t::iterator it = buf.find(socket);
   if (it == buf.end())
      return -1;
   else if (it->second.empty())
      return -1;
   else
   {
      it->second.pop_front();
      return 0;
   }
}
