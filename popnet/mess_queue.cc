#include "mess_queue.h"
#include "sim_foundation.h"
#include <string>
#include <iostream>
#include <strstream>
#include <unistd.h>

//changed at 2020-5-23
ofstream logfile("popnet.log");

bool operator<(const mess_event & a, const mess_event & b) {
	return a.event_start() < b.event_start();
}

bool operator>(const mess_event&a,const mess_event&b)
{
	return a.event_start()>b.event_start();
}

mess_queue * mess_queue::m_pointer_ = nullptr;

mess_queue::mess_queue(time_type start_time):
    current_time_(0),
	last_time_(0),
	mess_counter_(0),
	m_q_(),
	total_finished_(0)
{
	current_time_ = start_time;
	m_pointer_ = this;
	//changed at 2020-5-8
	//add_message(mess_event(0, ROUTER_));
	add_message(mess_event(TIME_0, ROUTER_));
}

string mess_queue:: mess_error_ = 
	string("This message type is not supported.\n");

//commented at 2020-3
//主循环，不断循环处理消息
void mess_queue::simulator() {
	//changed at 2020-5-23
	//static uint64_t wireMsgCnt=0,creditMsgCnt=0;
	time_type report_t = 0;
	long total_incoming = 0;
	//修改此循环
	//when mess queue is not empty and simulation deadline has not reach
	while(m_q_.size() > 0 && (current_time_ <= (configuration
					::ap().sim_length()))) {
		//changed at 2021-10-26
		//mess_event current_message = * get_message();
		mess_event current_message=get_message();
		//changed at 2020-5-23
		//输出日志
		logfile<<current_message;
		if(current_message.event_type()==WIRE_||current_message.event_type()==CREDIT_){
			logfile<<"\tFrom Router "<<current_message.src()
				<<" to Router "<<current_message.des()
				<<" Port "<<current_message.pc()
				<<" Virtual Channel "<<current_message.vc()<<endl;
		}
		remove_top_message();
		//保证当前时间小于或等于事件开始时间
		Sassert(static_cast<bool>(current_time_ <= ((current_message.
					event_start()) + S_ELPS_)));
		//if(current_time_>current_message.event_start()+S_ELPS_)continue;
		current_time_ = current_message.event_start();

		if(current_time_ > report_t) {
		   cout<<"Current time: "<<current_time_<<" Incoming packets"
			   <<total_incoming<<" Finished packets"<<total_finished_<<endl;
			//changed at 2020-5-23
			//cout<<"Wire Message Count: "<<wireMsgCnt<<" Credit Message Count: "<<creditMsgCnt<<endl;
		   sim_foundation::wsf().simulation_results();
		   report_t += REPORT_PERIOD_;
		}
		
		switch(current_message.event_type()) {

			case EVG_ ://读取轨迹文件的下一条记录
				//TODO: 改为读取外部输入
				sim_foundation::wsf().receive_EVG_message(current_message);
				total_incoming ++;
			break;

			case ROUTER_ :
				sim_foundation::wsf().receive_ROUTER_message(current_message);
			break;

			case WIRE_ :
				sim_foundation::wsf().receive_WIRE_message(current_message);
				//changed at 2020-5-23
				//wireMsgCnt++;
			break;

			case CREDIT_ :
				sim_foundation::wsf().receive_CREDIT_message(current_message);
				//changed at 2020-5-23
				//creditMsgCnt++;
			break;

			default:
				throw pro_error(mess_error_);
			break;
		} 
	} 
}

