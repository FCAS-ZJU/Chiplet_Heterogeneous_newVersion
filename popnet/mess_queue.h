#ifndef NETWORK_MESS_QUEUE_H_
#define NETWORK_MESS_QUEUE_H_

#include "SStd.h"
#include "index.h"
#include "configuration.h"
#include "sim_foundation.h"
#include "mess_event.h"
#include <exception>
#include <set>
#include <cmath>
#include <utility>
#include <iostream>
#include <string>

//changed at 2021-10-26
#include<queue>

//changed at 2021-10-26
bool operator>(const mess_event&a,const mess_event&b);

bool operator<(const mess_event & a, const mess_event & b);


class mess_queue {
	friend ostream & operator<<(ostream& os, const mess_queue & sgq);
	private:
		//changed at 2021-10-26
		//改为优先队列以提速
		//typedef multiset <mess_event> message_q;
		typedef std::priority_queue<mess_event,std::vector<mess_event>,std::greater<mess_event> > message_q;
		time_type current_time_;
		time_type last_time_; 
		long mess_counter_;
		message_q m_q_;
		long total_finished_;
		static mess_queue * m_pointer_;
		static string mess_error_;

	public:

		class pro_error: public exception {
			public:
				pro_error(const string & err) : what_(err) {}
				virtual const char * what() const throw() {return what_.c_str();}
				virtual ~pro_error() throw() {};

			private:
				string what_;
		}; 
		
		typedef message_q::size_type size_type; 
		//typedef message_q::iterator iterator;
		static const mess_queue & m_pointer() {return * m_pointer_;}
		static mess_queue & wm_pointer() {return * m_pointer_;}
		mess_queue(time_type start_time);
		~mess_queue(){m_pointer_ = 0;}

		time_type current_time() const {return current_time_;}

		time_type last_time() const {return last_time_;}

		long mess_counter() const {return mess_counter_;}

		void simulator();

		long total_finished() {return total_finished_;}
		long total_finished() const {return total_finished_;}
		void TotFin_inc() {total_finished_++;}

		//changed at 2021-10-26
		//iterator get_message() {return m_q_.begin();}
		const mess_event&get_message()const
		{
			return m_q_.top();
		}
		//void remove_message(iterator pos) {m_q_.erase(pos);}
		//void remove_top_message() {m_q_.erase(m_q_.begin());}
		void remove_top_message()
		{
			m_q_.pop();
		}
		size_type message_queue_size() const {return m_q_.size();}
		void add_message(const mess_event & x) 
		{
			mess_counter_++;
			//m_q_.insert(x);
			m_q_.push(x);
			//cout<<"Router "
		}
};
#endif
