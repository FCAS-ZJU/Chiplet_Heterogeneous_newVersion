#include<iostream>
#include<fstream>
#include<boost/property_tree/ini_parser.hpp>
#include<filesystem>
#include <boost/property_tree/ptree.hpp>
#include <boost/filesystem.hpp>
#include<cstdint>
#include<string>
using namespace std;

const char CONFIG_PATH[]="changes/zmq_pro.ini";
const char SUBNET_ITEM[]="subnet_id";

int64_t subnet;
void readConfig()
{
    if(!boost::filesystem::exists(CONFIG_PATH)){
        cout<<"no config file\n";
        return;
    }
    boost::property_tree::ptree root,tag;
    boost::property_tree::ini_parser::read_ini(CONFIG_PATH,root);
    tag=root.get_child("config");
    if(tag.count(SUBNET_ITEM)!=1)return;
    subnet=tag.get<int64_t>(SUBNET_ITEM);
}

int main()
{
    readConfig();
    ifstream ifs("message_record.txt");
    ofstream ofs("record_"+to_string(subnet)+".txt");
    string cmd;
    int64_t localPort,remotePort,remoteAddr,localCore;
    uint64_t t;
    while (ifs>>cmd>>localPort>>remoteAddr>>remotePort>>localCore>>t)
    {
        //命令 本地地址  本地端口 远程地址 远程端口 本地核心 时间（纳秒）
        ofs<<cmd<<' '
            <<subnet<<' '
            <<localPort<<' '
            <<remoteAddr<<' '
            <<remotePort<<' '
            <<localCore<<' '
            <<t<<'\n';
    }
    ifs.close();
    ofs.flush();
    ofs.close();
    return 0;
}
