#ifndef SIM_ROUTER_
#define SIM_ROUTER_

#include "flit.h"
#include "index.h"
#include "SStd.h"
#include "SRGen.h"
#include "configuration.h"
#include <vector>
#include <utility>
#include <map>
#include <functional>
#include <fstream>
#include <sstream>
#include <string>
#include <cstring>
#include <set>
extern "C" {
#include "SIM_power.h"
#include "SIM_router_power.h"
#include "SIM_power_router.h"
}

//changed at 2021-10-26
#include<queue>

// *****************************************************//
// data structure to model the structure and behavior   //
// of routers.                                          //
// *****************************************************//

class input_template {
	friend ostream& operator<<(ostream& os, const input_template & Ri);

	private:
		//input buffers: <phy<vc<buffer>>>
		vector<vector<vector<flit_template> > > input_;
		//the state of each input vc
		vector<vector<VC_state_type> > states_;
		//the candidate routing vcs
		vector<vector<vector<VC_type> > > routing_;
		//the chosen routing vc
		vector<vector<VC_type > > crouting_;
		//this is a flag to show that the buffer of injection is full
		//used to control the packet injection to decrease mem usage
		bool ibuff_full_;

//---------------------------------------------------------------------------//
//---------------------------------------------------------------------------//
	public:
		vector<vector<vector<flit_template> > > & input() {return input_;}
		const vector<vector<vector<flit_template> > > & input() const 
						   {return input_;}
		const vector<flit_template> & input(long a, long b) const 
							 {return input_[a][b];} 
		vector<flit_template> & input(long a, long b) {return input_[a][b];} 
//---------------------------------------------------------------------------//
		void add_flit(long a, long b, const flit_template & c)
		{	
			 input_[a][b].push_back(c);}
		void remove_flit(long a, long b)  
				 {input_[a][b].erase(input_[a][b].begin());}

		flit_template & get_flit(long a, long b) 
				{Sassert(input_[a][b].size() > 0);
				return(input_[a][b][0]);}
		const flit_template & get_flit(long a, long b) const{
				Sassert(input_[a][b].size() > 0);
				return(input_[a][b][0]);}
		flit_template & get_flit(long a, long b, long c) 
				{Sassert(input_[a][b].size() > c);
				return(input_[a][b][c]);}
		const flit_template & get_flit(long a, long b, long c) const{
				Sassert(input_[a][b].size() > c);
				return(input_[a][b][c]);}

//---------------------------------------------------------------------------//
		vector<vector<VC_state_type> > & states() {return states_;}
		const vector<vector<VC_state_type> > & states() const {return states_;}
		VC_state_type  state(long a, long b) {return states_[a][b];}
		const VC_state_type state(long a, long b) const {return states_[a][b];}

		void state_update(long a, long b, VC_state_type c){states_[a][b] = c;}
//---------------------------------------------------------------------------//
		vector<vector<VC_type > > & crouting() {return crouting_;}
		const vector<vector<VC_type > > & crouting() const {return crouting_;}
		VC_type & crouting(long a, long b) {return crouting_[a][b];}
		const VC_type & crouting(long a, long b) const {return crouting_[a][b];}
		void crouting_assign(long a, long b, VC_type c) {crouting_[a][b] = c;}
		void clear_crouting(long a, long b) {crouting_[a][b] = VC_NULL;}
//---------------------------------------------------------------------------//
		vector<vector<vector<VC_type > > > & routing() 
			{return routing_;}
		const vector<vector<vector<VC_type > > > & routing()  const
			{return routing_;}
		vector<VC_type > & routing(long a, long b)
	   		{return routing_[a][b];}
		const vector<VC_type > & routing(long a, long b) const
	   		{return routing_[a][b];}
		void add_routing(long a, long b, VC_type c) 
				{ routing_[a][b].push_back(c);}
		void clear_routing(long a, long b) {routing_[a][b].clear();}
//---------------------------------------------------------------------------//
		void ibuff_is_full() {ibuff_full_ = true;}
		void ibuff_not_full() {ibuff_full_ = false;}
		bool ibuff_full() {return ibuff_full_;}
		bool ibuff_full() const {return ibuff_full_;}
//---------------------------------------------------------------------------//

		input_template();
		//a: phy ports b: virtual channel
		input_template(long a, long b);
};

//****************************************************************************//
//****************************************************************************//
class output_template {
	private:
		//used for next input
		long buffer_size_;
		vector<vector<long> >counter_;
		vector<vector<VC_state_type> >flit_state_;
		//assigned for the input 
		vector<vector<VC_type> > assign_;
		//USED_ FREE_
		vector<vector<VC_usage_type> > usage_;
		//local output buffers
		vector<vector<flit_template> > outbuffers_;
		//output address
		vector<vector<VC_type> > outadd_;
		vector<long> localcounter_;
//---------------------------------------------------------------------------//
//---------------------------------------------------------------------------//
	public:
		long buffer_size() {return buffer_size_;}
		long buffer_size() const {return buffer_size_;}
		vector<vector<long> > & counter() {return counter_;}
		long counter(long a, long b) {return counter_[a][b];}
		long counter(long a, long b) const {return counter_[a][b];}
		void counter_inc(long a, long b) {counter_[a][b]++;}
		void counter_dec(long a, long b) {counter_[a][b]--;}
//---------------------------------------------------------------------------//
		vector<vector<VC_usage_type> > & usage() {return usage_;}
		VC_usage_type usage(long a,long b) {return usage_[a][b];}
		VC_usage_type usage(long a,long b) const {return usage_[a][b];}
		
		VC_type & assign(long a, long b) {return assign_[a][b];}
		const VC_type & assign(long a, long b) const {return assign_[a][b];}

		void acquire(long a, long b, VC_type c) 
			{usage_[a][b] = USED_; assign_[a][b] = c; }
		void release(long a, long b) 
			{usage_[a][b] = FREE_; assign_[a][b] = VC_NULL;}
//---------------------------------------------------------------------------//
		void add_flit(long a, const flit_template& b)
			{ outbuffers_[a].push_back(b); localcounter_[a]--;}

		void remove_flit(long a)
		{ 
			Sassert(outbuffers_[a].size() > 0);
			outbuffers_[a].erase(outbuffers_[a].begin()); 
			vector<flit_template>(outbuffers_[a]).swap(outbuffers_[a]);
			//Sassert(flit_state_[a].size() > 0);
			//flit_state_[a].erase(flit_state_[a].begin());
			//vector<VC_state_type>(flit_state_[a]).swap(flit_state_[a]);
			localcounter_[a]++;
		}

		VC_state_type flit_state(long a) {return flit_state_[a][0];}
		VC_state_type flit_state(long a) const {return flit_state_[a][0];}

		flit_template & get_flit(long a)
	   		{Sassert(outbuffers_[a].size() > 0);
		   	return outbuffers_[a][0];}

//---------------------------------------------------------------------------//
		vector<flit_template> & outbuffers(long a) 
			{return outbuffers_[a];}
		const vector<flit_template> & outbuffers(long a) const 
			{return outbuffers_[a];}

//---------------------------------------------------------------------------//
		vector<vector<VC_type> > outadd() {return outadd_;}
		const vector<vector<VC_type> > outadd() const {return outadd_;}
		vector<VC_type> outadd(long a) {return outadd_[a];}
		const vector<VC_type> outadd(long a) const {return outadd_[a];}
		VC_type get_add(long a) {return outadd_[a][0];}
		void remove_add(long a) {Sassert(outadd_[a].size() > 0); 
			outadd_[a].erase(outadd_[a].begin());}
		void add_add(long a, VC_type b) {outadd_[a].push_back(b);}

//---------------------------------------------------------------------------//
		vector<long> & localcounter() {return localcounter_;}
		long localcounter(long a) {return localcounter_[a];}
		long localcounter(long a) const {return localcounter_[a];}
		void localcounter_inc(long a) {localcounter_[a]++;}
		void localcounter_dec(long a) {localcounter_[a]--;}
//---------------------------------------------------------------------------//


		output_template();
		output_template(long a, long b, long c, long d);
};

//***************************************************************************//
class power_template {
	long flit_size_;
	SIM_power_router_info_t router_info_;
	SIM_power_router_t router_power_;
	SIM_power_arbiter_t arbiter_vc_power_;
	SIM_power_bus_t link_power_;
	vector<Data_type> buffer_write_;
	vector<Data_type> buffer_read_;
	vector<Data_type> crossbar_read_;
	vector<Data_type> crossbar_write_;
	vector<Data_type> link_traversal_;
	vector<long> crossbar_input_;
	vector<vector<Atom_type> > arbiter_vc_req_;
	vector<vector<unsigned long> > arbiter_vc_grant_;

	public:
	power_template();
	power_template(long a, long b, long c);
	void power_buffer_read(long in_port, Data_type & read_d);
	void power_buffer_write(long in_port, Data_type & write_d);
	void power_crossbar_trav(long in_port, long out_port, Data_type & trav_d);
	void power_vc_arbit(long pc, long vc, Atom_type req, unsigned long gra);
	void power_link_traversal(long in_port, Data_type & read_d);
	double power_buffer_report();
	double power_link_report();
	double power_crossbar_report();
	double power_arbiter_report();
};
//***************************************************************************//
//changed at 2021-10-26
//数据包结构体
struct SPacket
{
	time_type startTime;
	add_type sourceAddress,destinationAddress;
	long packetSize;

	SPacket()
	{
		size_t addSize=configuration::ap().cube_number();
		sourceAddress.reserve(addSize);
		destinationAddress.reserve(addSize);
	}
};
//physical port 0 is injection port
class sim_router_template {
	friend ostream& operator<<(ostream& os, const sim_router_template & sr);

	private:
		//router address
		add_type address_;
		//the input buffer module
		input_template   input_module_;
		//output module
		output_template output_module_;
		//power module
		power_template power_module_;
		//initial random number
		Data_type init_data_;
		long ary_size_;//表示网络是一个n^d网络，其中n就是ary_size_
		long flit_size_; //size of 64 bits
		long physic_ports_;
		long vc_number_;
		long buffer_size_; //each vc buffer size
		long outbuffer_size_; //each vc buffer size
		//the total delay of the flits
		time_type total_delay_;
		//the routing algorithm used
		long routing_alg_;
		//the routing algorithm is now used, this is a pointer, points to
		//the routing procedure
		void (sim_router_template::* curr_algorithm)(const add_type &,
			   const add_type &, long, long);
		void (sim_router_template::*curr_prevRouterFunc)(add_type&,long);
		long (sim_router_template::*curr_prevPortFunc)(long);
		time_type (sim_router_template::*curr_wireDelayFunc)(long);
		void (sim_router_template::*curr_nextAddFunc)(add_type&,long);
		long (sim_router_template::*curr_wirePcFunc)(long);

		//used for the next packet injection time
		time_type local_input_time_;
		//packet counter
		long packet_counter_;
		//changed at 2021-10-26
		//将文件改为队列，以提高速度
		//input trace file
		//ifstream * localinFile_;//本地轨迹文件流，本路由器的轨迹文件流
		std::queue<SPacket>localInputTraces;

		time_type getWireDelay(long port);
		time_type getWireDelay_mesh(long port);
		time_type getWireDelay_chipletMesh(long port);
		time_type getWireDelay_chipletStar(long port);
		
		void getNextAddress(add_type&nextAddress,long port);
		void getNextAddress_chipletMesh(add_type&nextAdd,long port);
		void getNextAddress_chipletStar(add_type &nextAddress, long port);
		void getNextAddress_mesh(add_type&nextAdd,long port);

		long getWirePc(long port);
		long getWirePc_mesh(long port);
		long getWirePc_chipletStar(long port);

		void getFromRouter(add_type&from,long port);
		void getFromRouter_mesh(add_type&from,long port);
		void getFromRouter_chipletStar(add_type&from,long port);
		
		long getFromPort(long port);
		long getFromPort_mesh(long port);
		long getFromPort_chipletStar(long port);
	public:
		vector<long> & address() {return address_;}
		const vector<long> & address() const {return address_;}

		input_template  & input_module() {return input_module_;}
		output_template & output_module() {return output_module_;}	
		long physic_ports() {return physic_ports_;}
		long vc_number() {return vc_number_;}
		long buffer_size() {return buffer_size_;}

		void accept_flit(time_type a, const flit_template & b);

	
		double power_buffer_report() { return power_module_.power_buffer_report();}
		double power_crossbar_report() { return power_module_.power_crossbar_report();}
		double power_arbiter_report() { return power_module_.power_arbiter_report();}
		double power_link_report() { return power_module_.power_link_report();}
		//simulation results and empty checking
		//empty check
		void empty_check() const;

		sim_router_template();
		//a: phy ports. b: vc number. c: buffer size. d: output buffer size.
		//e: its address  f: ary_size g: flit_size
		sim_router_template(long a, long b, long c, long d, 
				const add_type & e, long f, long g);
		//-------------------------------------------------------------------//
		//simulator evoking
		void router_sim_pwr();
		//-------------------------------------------------------------------//
	
		//calculate the total delay	
		void delay_update(time_type a) { total_delay_ += a;}
		time_type total_delay() const {return total_delay_;}
		time_type total_delay() {return total_delay_;}

		//receive credit processing
		void receive_credit(long a, long b);

		void receive_packet();
		void inject_packet(long a, add_type & b, add_type & c, time_type d,
				long e);
		void receive_flit(long a, long b, flit_template & c);

		void routing_decision();
		void XY_algorithm(const add_type & des_t, const add_type & sor_t, 
				long s_ph, long s_vc);
		void TXY_algorithm(const add_type & des_t, 
				const add_type & sor_t, long s_ph, long s_vc);

		VC_type vc_selection(long a, long b);
		void vc_arbitration();

		void sw_arbitration();

		void flit_outbuffer();

		void flit_traversal();
		void flit_traversal(long a);

		/* ifstream & localinFile() {return * localinFile_;}
		void init_local_file(); */

		//changed at 2020-5-6
		//新路由算法
		void chiplet_routing_alg(const add_type & des_t,const add_type & sor_t, long s_ph, long s_vc);
		//changed at 2020-5-19
		void chiplet_star_topo_routing_alg(const add_type & des_t,const add_type & src_t, long s_ph, long s_vc);

		//changed at 2021-10-26
		void inputTrace(const SPacket&packet);
};

#endif
