#include "mess_queue.h"
#include "sim_foundation.h"
#include <string>
#include <iostream>
#include <strstream>
#include <unistd.h>

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
	while(m_q_.size() > 0 && (current_time_ <= (configuration::ap().sim_length())))
	{
		//changed at 2021-10-26
		//mess_event current_message = * get_message();
		mess_event current_message = get_message();
		//changed at 2020-5-23
		//输出日志
		if(current_message.event_type() == WIRE_)
		{
			std::cout << current_time_ << "\tFrom Router " << current_message.src()
				<< " to Router " << current_message.des()
				<< " Port " << current_message.pc()
				<< " Virtual Channel " << current_message.vc() << endl;
		}
		remove_top_message();
		//保证当前时间小于或等于事件开始时间
		Sassert(static_cast<bool>(current_time_ <= ((current_message.event_start()) + S_ELPS_)));
		//if(current_time_>current_message.event_start()+S_ELPS_)continue;
		current_time_ = current_message.event_start();

		//if(current_time_ > report_t) {
		//   cout<<"Current time: "<<current_time_<<" Incoming packets"
		//	   <<total_incoming<<" Finished packets"<<total_finished_<<endl;
		//	//changed at 2020-5-23
		//	//cout<<"Wire Message Count: "<<wireMsgCnt<<" Credit Message Count: "<<creditMsgCnt<<endl;
		//   sim_foundation::wsf().simulation_results();
		//   report_t += REPORT_PERIOD_;
		//}
		
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

		// TODO: Quick path
		// If there is no package in the network, directly jump to next EVG event.
		if (total_incoming == total_finished_)
		{
			// Copy events to temp event queue, except ROUTER event.
			std::vector<mess_event> temp_event_queue;
			std::vector<mess_event> temp_router_event_queue;
			double first_event_time = -1;
			double first_router_event_time = -1;
			while (m_q_.size() > 0)
			{
				if (m_q_.top().event_type() != ROUTER_)
				{
					if (first_event_time < 0)
					{
						first_event_time = m_q_.top().event_start();
					}
					temp_event_queue.push_back(m_q_.top());
				}
				else
				{
					if (first_router_event_time < 0)
					{
						first_router_event_time = m_q_.top().event_start();
					}
					temp_router_event_queue.push_back(m_q_.top());
				}
				m_q_.pop();
			}

			std::cout << first_router_event_time << " " << first_event_time << std::endl;
			// If there are only ROUTER events in queue, quit simulation.
			if (first_event_time < 0)
			{
				break;
			}
			if (first_router_event_time < 0)
			{
				continue;
			}

			// Copy events back to message queue.
			while (temp_event_queue.size() > 0)
			{
				m_q_.push(temp_event_queue.front());
				temp_event_queue.erase(temp_event_queue.begin());
			}

			if (first_router_event_time < first_event_time)
			{
				// Add new ROUTER event.
				double t = (first_event_time - first_router_event_time) / PIPE_DELAY_;
				double new_router_event_time = first_router_event_time + round(t) * PIPE_DELAY_;

				m_q_.push(mess_event(new_router_event_time, ROUTER_));
				current_time_ = new_router_event_time;
				std::cout << "Direct Forward to cycle: " << current_time_ << " (" << t << ")" << std::endl;
			}
			else
			{
				// Copy router event back to event queue.
				while (temp_router_event_queue.size() > 0)
				{
					m_q_.push(temp_router_event_queue.front());
					temp_router_event_queue.erase(temp_router_event_queue.begin());
				}
			}
		}
	} 
}

