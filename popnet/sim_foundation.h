#ifndef SIM_FOUNDATION_
#define SIM_FOUNDATION_

#include "sim_router.h"
#include "flit.h"
#include "mess_event.h"
#include <vector>
#include "configuration.h"
#include "SRGen.h"
#include <exception>
#include <utility>
#include <map>
#include <functional>
#include <iostream>
#include <fstream>

class sim_foundation {

	friend ostream& operator<<(ostream&, const sim_foundation&);

	private:

		vector<sim_router_template> inter_network_;
		long ary_size_;//每个维度下的尺寸
		long cube_size_;//维度
		long router_counter_;//路由器总数
		long packet_counter_;

		//changed at 2021-10-26
		//将文件流改为队列，以提高速度
		//ifstream inFile_;
		std::queue<SPacket>inputTraces;

		static string file_name_;
		static sim_foundation * s_f_;

	public:
		class file_error: public exception {
			public:
				file_error(const string & err) : what_(err) {}
				virtual const char * what() const throw() {return what_.c_str();}
				virtual ~file_error() throw();

			private:
				string what_;
		};

		static const sim_foundation & sf() {return *s_f_;}
		static sim_foundation & wsf() {return *s_f_;}
		sim_foundation();
		~sim_foundation(){s_f_ = nullptr;}
		long ary_size() const {return ary_size_;}
		long cube_size() const {return cube_size_;}
		long packet_counter() {return packet_counter_;}
		long packet_counter() const {return packet_counter_;}

		vector<sim_router_template> & inter_network() {return inter_network_;}
		const vector<sim_router_template> & inter_network() const
	   												{return inter_network_;}
		sim_router_template & router(const add_type& a);
		const sim_router_template & router(const add_type& a) const;
		bool valid_address(const add_type & a) const;

		void receive_EVG_message(mess_event mesg);
		void receive_ROUTER_message(mess_event mesg);
		void receive_WIRE_message(mess_event mesg);
		void receive_CREDIT_message(mess_event mesg);

		void router_power_exchange();
		
		void simulation_results();
		void simulation_check();

		void init_file();

		//changed at 2021-10-26
		void readTraceFile();

};

void readPacket(SPacket&packet,std::ifstream&ifs,size_t dimension);
#endif
