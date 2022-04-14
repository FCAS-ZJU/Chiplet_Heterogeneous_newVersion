#ifndef NETWORK_CONFIG_H_
#define NETWORK_CONFIG_H_

/*******************************************************/
/*           Interconnection Network Simulator parse   */
/*******************************************************/

#include <string>
#include <exception>
#include "SStd.h"
#include "SRGen.h"
#include "index.h"
using namespace std;

//changed at 2020-5-8
extern const time_type TIME_0;

/*******************************************************/
class configuration {
	friend ostream & operator<<(ostream &, const configuration & cf);
	private:
		long ary_number_; //k-ary 每个维度上的路由器个数，比如一个二维的9*9网络的此参数就是9
		long cube_number_;  //n-cube 维度
		long virtual_channel_number_; //VC per physical port
		long buffer_size_; //buffer size of each vc
		long outbuffer_size_; //output buffer size
		long flit_size_; //flit size, i.e. num. of 64 bits
		double link_length_; //link length #um
		long ran_seed_; //random seed used for generate random number
		time_type sim_length_; //simulation length i.e. 10,000,000 cycles.
	 	string trace_fname_; //simulation input file name 轨迹文件
		long routing_alg_; //choose one of the routing algorithms
		vc_share_type vc_share_; //VC sharing, based on routing algorithms
		static string help_;
		static configuration * ap_;
	public:
		class init_error: public exception {
			public:
				init_error(const string & err) : what_(err) {}
				virtual const char * what() const throw() {return what_.c_str();}
				virtual ~init_error() throw() {};

			private:
				string what_;
		};

		class parse_error: public exception {
			public:
				parse_error(const string & err) : what_(err) {}
				virtual const char * what() const throw() { return what_.c_str();}
				virtual ~parse_error() throw() {};
			
			private:
				string what_;
		};

		configuration(int argc, char * const argv []);

		static const configuration  & ap() {return *ap_;}
		static configuration  & wap() {return *ap_;}

		long ary_number() const {return ary_number_;}
		long ary_number() {return ary_number_;}
		long cube_number() const {return cube_number_;}
		long cube_number() {return cube_number_;}
		long virtual_channel_number() const {return virtual_channel_number_;}
		long virtual_channel_number() {return virtual_channel_number_;}
		long buffer_size() const {return buffer_size_;}
		long buffer_size() {return buffer_size_;}
		long outbuffer_size() const {return outbuffer_size_;}
		long outbuffer_size() {return outbuffer_size_;}
		long flit_size() const {return flit_size_;}
		long flit_size() {return flit_size_;}
		double link_length() {return link_length_;}
		double link_length() const {return link_length_;}
		time_type sim_length() const {return sim_length_;}
		time_type sim_length() {return sim_length_;}
		string trace_fname() {return trace_fname_;}
		string trace_fname() const {return trace_fname_;}
		long routing_alg() {return routing_alg_;}
		long routing_alg() const {return routing_alg_;}
		vc_share_type vc_share() {return vc_share_;}
		vc_share_type vc_share() const {return vc_share_;}
		~configuration() { ap_ = 0;}
};
#endif
