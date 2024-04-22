#include"interchiplet_client.h"

#include <iostream>
#include <cstring>
#include <sys/socket.h>
#include <netinet/in.h>
#include <sys/types.h>
#include <deque>
#include <string>
#include <algorithm>
#include <cassert>
#include <iterator>
#include <zmq.hpp>
#include <zmq_addon.hpp>
#include <fstream>
#include <thread>
#include <errno.h>
#include <unistd.h>
#include<sstream>
#include<boost/property_tree/ini_parser.hpp>
#include<filesystem>
#include <boost/property_tree/ptree.hpp>
#include <boost/filesystem.hpp>
#include<queue>
using namespace std;

ofstream logfile("changes/zmq_pro.log");

//read config
int64_t subnet;
string popnetAddr;
int readConfig()
{
    const char CONFIG_PATH[]="changes/zmq_pro.ini";
    const char SUBNET_ITEM[]="subnet_id";
    const char INTER_ADDR_ITEM[]="inter_address";
    if(!boost::filesystem::exists(CONFIG_PATH)){
        return -1;
    }
    boost::property_tree::ptree root,tag;
    boost::property_tree::ini_parser::read_ini(CONFIG_PATH,root);
    tag=root.get_child("config");
    if(tag.count(SUBNET_ITEM)!=1||tag.count(INTER_ADDR_ITEM)!=1)return -2;
    subnet=tag.get<int64_t>(SUBNET_ITEM);
    popnetAddr=tag.get<string>(INTER_ADDR_ITEM);
    return 0;
}

//与sniper通信部分
namespace nsSniperConn
{
const size_t BUF_SIZE = 4096;
const int SNIPER_PORT_BASE = 7000;
int listenFd, connfd;
queue<string> lines;
int openPort(int port)
{
    const int LISTEN_QUEUE_LEN = 10;
    listenFd = socket(AF_INET, SOCK_STREAM, 0);
    if (listenFd < 0)
        return listenFd;
    sockaddr_in serverAddr;
    memset(&serverAddr, 0, sizeof(serverAddr));
    serverAddr.sin_family = AF_INET;
    serverAddr.sin_addr.s_addr = htonl(INADDR_ANY);
    serverAddr.sin_port = htons(port);
    int r = bind(listenFd, (sockaddr *)&serverAddr, sizeof(serverAddr));
    if (r < 0)
        return -1;
    r = listen(listenFd, LISTEN_QUEUE_LEN);
    if(r<0)return -2;
    return 0;
}
int getConnection()
{
    connfd = accept(listenFd, (sockaddr *)NULL, NULL);
    return connfd;
}
int receiveLine()
{
    static int remain = 0;
    static char buf[BUF_SIZE+1] = {0};
    if (lines.empty())
    {
        char *end = buf + remain;
        char *pos = end;
        int r;
        while (pos == end)
        {
            r = recv(connfd, end, BUF_SIZE - remain, 0);
            if (r <= 0)
                return r;
            remain += r;
            end = buf + remain;
            pos = find(buf, end, '\n');
            //assert(remain < BUF_SIZE);
            if(remain>=BUF_SIZE)throw "Message too long";
        }
        *pos = '\0';
        lines.push(buf);
        char *pos0 = pos + 1;
        for (;;)
        {
            pos = find(pos0, end, '\n');
            if (pos == end)
                break;
            *pos = '\0';
            lines.push(pos0);
            pos0 = pos + 1;
        }
        remain = distance(pos0, end);
        copy(pos0, end, buf);
        return lines.size();
    }
    return 0;
}
int disconnect()
{
    return close(connfd) | close(listenFd);
}
} // namespace nsSniperConn

int main()
{
    int r;
    r=readConfig();
    logfile<<"readConfig: "<<r<<endl;
    logfile<<"\tsubnet: "<<subnet<<endl
        <<"\tinter addr: "<<popnetAddr<<endl;
    if(r<0)return r;
    r=nsInterchiplet::connectZmq(popnetAddr);
    logfile<<"popnet connect: "<<r<<endl;
    r = nsSniperConn::openPort(nsSniperConn::SNIPER_PORT_BASE + subnet);
    //logfile<<"openport succeeds"<<endl;
    logfile << "sniper openport: " << r << endl;
    cout << subnet << endl;
    r = nsSniperConn::getConnection();
    //cout<<"ok"<<endl;
    //logfile<<"connection succeeds"<<endl;
    logfile << "sniper connection: " << r << endl;
    for (;;)
    {
        if (nsSniperConn::receiveLine() <= 0)
        {
            logfile << "Error: " << errno << endl;
            break;
        }
        logfile << "read from sniper: " << nsSniperConn::lines.front() << endl;
        stringstream ss(nsSniperConn::lines.front()+'\n');
        string cmd,d;
        ss>>cmd;
        if(cmd=="send"){
            getline(ss,d);
            nsInterchiplet::writeMsg("send "+d);
            //控制发送速率，避免丢消息
            this_thread::sleep_for(1ms);
            cout<<"send_ret"<<endl;
        }
        else if(cmd=="recv"){
            nsInterchiplet::writeMsg("recv\n");
            nsInterchiplet::readMsg(d);
            if(d.find("exit")<d.size()){
                logfile<<"interchiplet exits"<<endl;
                break;
            }
            cout<<d<<endl;
            logfile<<"read from ZMQ: "<<d<<endl;
        }
        nsSniperConn::lines.pop();
        //this_thread::sleep_for(1ms);
    }
    r = nsSniperConn::disconnect();
    logfile << "sniper disconnect: " << r << endl;
    logfile<<"popnet disconnect: "<<nsInterchiplet::closeZmq()<<endl;
    logfile << "exit" << endl;
    logfile.close();
    return 0;
}
