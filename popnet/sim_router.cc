#include <math.h>
#include <iomanip>
#include <sstream>
#include <cstdio>
#include "sim_router.h"
#include "sim_foundation.h"
#include "mess_queue.h"
#include "mess_event.h"
#include "SRGen.h"
#include "SStd.h"

//changed at 2020-5-12
#include <algorithm>
#include <functional>

//changed at 2021-12-9
#define USING_VOLA
#ifdef USING_VOLA
#define VOLA volatile
#else
#define VOLA
#endif

// *****************************************************//
// data structure to model the structure and behavior   //
// of routers.                                          //
// *****************************************************//
ostream &operator<<(ostream &os, const sim_router_template &sr)
{
	long k = sr.address_.size();
	for (long i = 0; i < k; i++)
	{
		os << sr.address_[i] << " ";
	}
	os << endl;
	return os;
}

//***************************************************************************//
sim_router_template::sim_router_template() : address_(),
											 input_module_(),
											 output_module_(),
											 power_module_(),
											 init_data_(),
											 ary_size_(),
											 flit_size_(),
											 physic_ports_(),
											 vc_number_(),
											 buffer_size_(),
											 outbuffer_size_(),
											 total_delay_(),
											 routing_alg_(),
											 curr_algorithm(),
											 local_input_time_(),
											 packet_counter_()/* ,
											 localinFile_() */

{
}

//***************************************************************************//
//a : physical ports, b: vc number, c: buffer size,  d: output
//buffer size, e: address  f: ary_size_ g: flit_size_
sim_router_template::sim_router_template(long a, long b, long c,
										 long d, const add_type &e, long f, long g) : address_(e),
																					  input_module_(a, b),
																					  output_module_(a, b, c, d),
																					  power_module_(a, b, g),
																					  init_data_(),
																					  ary_size_(f),
																					  flit_size_(g),
																					  physic_ports_(a),
																					  vc_number_(b),
																					  buffer_size_(c),
																					  outbuffer_size_(d),
																					  total_delay_(0),
																					  routing_alg_(0),
																					  curr_algorithm(0),
																					  local_input_time_(),
																					  packet_counter_(0)/* ,
																					  localinFile_() */
{
	init_data_.resize(g);
	for (long i = 0; i < g; i++)
	{
		init_data_[i] = SRGen::wrg().flat_ull(0, MAX_64_);
	}
	//get the trace file name
	//changed at 2021-10-26
	/* init_local_file();
	localinFile() >> local_input_time_; */
	local_input_time_=localInputTraces.front().startTime;

	routing_alg_ = configuration::ap().routing_alg();
	switch (routing_alg_)
	{

	case XY_:
		curr_algorithm = &sim_router_template::XY_algorithm;
		curr_prevRouterFunc= &sim_router_template::getFromRouter_mesh;
		curr_prevPortFunc= &sim_router_template::getFromPort_mesh;
		curr_wireDelayFunc= &sim_router_template::getWireDelay_mesh;
		curr_nextAddFunc= &sim_router_template::getNextAddress_mesh;
		curr_wirePcFunc= &sim_router_template::getWirePc_mesh;
		break;

	case TXY_:
		curr_algorithm = &sim_router_template::TXY_algorithm;
		curr_prevRouterFunc= &sim_router_template::getFromRouter_mesh;
		curr_prevPortFunc= &sim_router_template::getFromPort_mesh;
		curr_wireDelayFunc= &sim_router_template::getWireDelay_mesh;
		curr_nextAddFunc= &sim_router_template::getNextAddress_mesh;
		curr_wirePcFunc= &sim_router_template::getWirePc_mesh;
		break;

	//changed at 2020-5-6
	//添加了芯粒的路由算法
	case CHIPLET_ROUTING_MESH:
		curr_algorithm = &sim_router_template::chiplet_routing_alg;
		curr_prevRouterFunc= &sim_router_template::getFromRouter_mesh;
		curr_prevPortFunc= &sim_router_template::getFromPort_mesh;
		curr_wireDelayFunc= &sim_router_template::getWireDelay_chipletMesh;
		curr_nextAddFunc= &sim_router_template::getNextAddress_chipletMesh;
		curr_wirePcFunc= &sim_router_template::getWirePc_mesh;
		break;

	//changed at 2020-5-19
	//增加了星型拓扑的芯粒路由算法
	case CHIPLET_STAR_TOPO_ROUTING:
		curr_algorithm = &sim_router_template::chiplet_star_topo_routing_alg;
		curr_prevRouterFunc= &sim_router_template::getFromRouter_chipletStar;
		curr_prevPortFunc= &sim_router_template::getFromPort_chipletStar;
		curr_wireDelayFunc= &sim_router_template::getWireDelay_chipletStar;
		curr_nextAddFunc= &sim_router_template::getNextAddress_chipletStar;
		curr_wirePcFunc= &sim_router_template::getWirePc_chipletStar;
		break;

	default:
		Sassert(0);
		break;
	}
}

power_template::power_template() : flit_size_(),
								   router_info_(),
								   router_power_(),
								   arbiter_vc_power_(),
								   link_power_(),
								   buffer_write_(),
								   buffer_read_(),
								   crossbar_read_(),
								   crossbar_write_(),
								   link_traversal_(),
								   crossbar_input_(),
								   arbiter_vc_req_(),
								   arbiter_vc_grant_()
{
}
//a: physical ports, b: virtual ports c: flit size
power_template::power_template(long a, long b, long c) : flit_size_(c),
														 router_info_(),
														 router_power_(),
														 arbiter_vc_power_(),
														 link_power_(),
														 buffer_write_(),
														 buffer_read_(),
														 crossbar_read_(),
														 crossbar_write_(),
														 link_traversal_(),
														 crossbar_input_(),
														 arbiter_vc_req_(),
														 arbiter_vc_grant_()

{
	FUNC(SIM_router_power_init, &router_info_, &router_power_);
	buffer_write_.resize(a);
	buffer_read_.resize(a);
	crossbar_read_.resize(a);
	crossbar_write_.resize(a);
	link_traversal_.resize(a);
	crossbar_input_.resize(a, 0);
	for (long i = 0; i < a; i++)
	{
		buffer_write_[i].resize(flit_size_, 0);
		buffer_read_[i].resize(flit_size_, 0);
		crossbar_read_[i].resize(flit_size_, 0);
		crossbar_write_[i].resize(flit_size_, 0);
		link_traversal_[i].resize(flit_size_, 0);
	}
	SIM_arbiter_init(&arbiter_vc_power_, 1, 1, a * b, 0, NULL);
	arbiter_vc_req_.resize(a);
	arbiter_vc_grant_.resize(a);
	for (long i = 0; i < a; i++)
	{
		arbiter_vc_req_[i].resize(b, 1);
		arbiter_vc_grant_[i].resize(b, 1);
	}
	SIM_bus_init(&link_power_, GENERIC_BUS, IDENT_ENC, ATOM_WIDTH_,
				 0, 1, 1, configuration::wap().link_length(), 0);
}

void power_template::power_buffer_read(long in_port, Data_type &read_d)
{
	for (long i = 0; i < flit_size_; i++)
	{
		FUNC(SIM_buf_power_data_read, &(router_info_.in_buf_info),
			 &(router_power_.in_buf), read_d[i]);
		buffer_read_[in_port][i] = read_d[i];
	}
}
void power_template::power_link_traversal(long in_port, Data_type &read_d)
{
	for (long i = 0; i < flit_size_; i++)
	{
		Atom_type old_d = link_traversal_[in_port][i];
		Atom_type new_d = read_d[i];
		SIM_bus_record(&link_power_, old_d, new_d);
		link_traversal_[in_port][i] = read_d[i];
	}
}
int i = 0;
void power_template::power_buffer_write(long in_port, Data_type &write_d)
{

	for (i = 0; i < flit_size_; i++)
	{
		//changed at 2021-12-9
		//此处有越界问题，已暂时修复
		/* Atom_type old_d = buffer_write_[in_port][i];
		Atom_type new_d = write_d[i];
		Atom_type old_d2 = buffer_write_[in_port][i];
		Atom_type new_d2 = write_d[i]; */
		VOLA Atom_type ata[4]={0};
		VOLA Atom_type&old_d=ata[0];
		VOLA Atom_type&new_d=ata[1];
		VOLA Atom_type&old_d2=ata[2];
		VOLA Atom_type&new_d2=ata[3];
		old_d=buffer_write_[in_port][i];
		new_d=write_d[i];
		old_d2=buffer_write_[in_port][i];
		new_d2=new_d2 = write_d[i];

		FUNC(SIM_buf_power_data_write, &(router_info_.in_buf_info),
			 &(router_power_.in_buf), (char *)(&old_d),
			 (char *)(&old_d),
			 (char *)(&new_d));

		buffer_write_[in_port][i] = write_d[i];
	}
}

void power_template::power_crossbar_trav(long in_port, long out_port,
										 Data_type &trav_d)
{
	for (long i = 0; i < flit_size_; i++)
	{
		SIM_crossbar_record(&(router_power_.crossbar), 1, trav_d[i],
							crossbar_read_[in_port][i], 1, 1);
		SIM_crossbar_record(&(router_power_.crossbar), 0, trav_d[i],
							crossbar_write_[out_port][i], crossbar_input_[out_port],
							in_port);

		crossbar_read_[in_port][i] = trav_d[i];
		crossbar_write_[out_port][i] = trav_d[i];
		crossbar_input_[out_port] = in_port;
	}
}
void power_template::power_vc_arbit(long pc, long vc, Atom_type req,
									unsigned long gra)
{
	SIM_arbiter_record(&arbiter_vc_power_, req, arbiter_vc_req_[pc][vc],
					   gra, arbiter_vc_grant_[pc][vc]);
	arbiter_vc_req_[pc][vc] = req;
	arbiter_vc_grant_[pc][vc] = gra;
}

double power_template::power_arbiter_report()
{
	return SIM_arbiter_report(&arbiter_vc_power_);
}

double power_template::power_buffer_report()
{
	return SIM_array_power_report(&(router_info_.in_buf_info),
								  &(router_power_.in_buf));
}

double power_template::power_crossbar_report()
{
	return SIM_crossbar_report(&(router_power_.crossbar));
}

double power_template::power_link_report()
{
	return SIM_bus_report(&link_power_);
}

//***************************************************************************//
/* void sim_router_template::init_local_file()
{
	string name_t = configuration::wap().trace_fname();
	for (long i = 0; i < address_.size(); i++)
	{
		ostringstream n_t;
		n_t << address_[i];
		name_t = name_t + "." + n_t.str();
	}
	localinFile_ = new ifstream;
	localinFile_->open(name_t.c_str());
	if (!localinFile_)
	{
		cerr << "can not open trace file :" << name_t << endl;
		assert(0);
	}
} */
void sim_router_template::inputTrace(const SPacket&packet)
{
	localInputTraces.push(packet);
}
//***************************************************************************//
ostream &operator<<(ostream &os, const input_template &Ri)
{
	for (long i = 0; i < (Ri.input_).size(); i++)
	{
		for (long j = 0; j < ((Ri.input_)[i]).size(); j++)
		{
			for (long k = 0; k < ((Ri.input_)[i][j]).size(); k++)
			{
				os << Ri.input_[i][j][k] << " ";
			}
			os << "|";
		}
		os << endl;
	}
	return os;
}

//***************************************************************************//

input_template::input_template() : input_(),
								   states_(),
								   routing_(),
								   crouting_(),
								   ibuff_full_()
{
}

//***************************************************************************//
//a: phy ports. b: virtual number
input_template::input_template(long a, long b) : input_(),
												 states_(),
												 routing_(),
												 crouting_(),
												 ibuff_full_(false)
{
	input_.resize(a);
	for (long i = 0; i < a; i++)
	{
		input_[i].resize(b);
	}
	states_.resize(a);
	for (long i = 0; i < a; i++)
	{
		states_[i].resize(b, INIT_);
	}
	routing_.resize(a);
	for (long i = 0; i < a; i++)
	{
		routing_[i].resize(b);
	}
	crouting_.resize(a);
	for (long i = 0; i < a; i++)
	{
		crouting_[i].resize(b, VC_NULL);
	}
}

//***************************************************************************//
//output buffer counter module
output_template::output_template() : buffer_size_(),
									 counter_(),
									 flit_state_(),
									 assign_(),
									 usage_(),
									 outbuffers_(),
									 outadd_(),
									 localcounter_()
{
}
//a: phy size. b: vir number. c: input buffer size. d: output buffer size
output_template::output_template(long a, long b, long c, long d) : buffer_size_(c),
																   counter_(),
																   flit_state_(),
																   assign_(),
																   usage_(),
																   outbuffers_(),
																   outadd_(),
																   localcounter_()
{
	counter_.resize(a);
	for (long i = 0; i < a; i++)
	{
		counter_[i].resize(b, c);
	}
	localcounter_.resize(a, d);
	assign_.resize(a);
	for (long i = 0; i < a; i++)
	{
		assign_[i].resize(b, VC_NULL);
	}
	usage_.resize(a);
	for (long i = 0; i < a; i++)
	{
		usage_[i].resize(b, FREE_);
	}
	outbuffers_.resize(a);
	flit_state_.resize(a);
	outadd_.resize(a);
}

//***************************************************************************//
void sim_router_template::receive_credit(long a, long b)
{
	output_module_.counter_inc(a, b);
}

//***************************************************************************//
void sim_router_template::receive_packet()
{
	time_type event_time = mess_queue::m_pointer().current_time();
	long cube_s = sim_foundation::wsf().cube_size();
	//changed at 2021-10-26
	/* add_type sor_addr_t;
	add_type des_addr_t;
	long pack_size_t;
	long pack_c; */
	//changed at 2021-10-26
	//预留空间
	/* sor_addr_t.reserve(cube_s);
	des_addr_t.reserve(cube_s); */
	while ((input_module_.ibuff_full() == false) && (local_input_time_ <=
													 event_time + S_ELPS_))
	{
		/* sor_addr_t.clear();
		des_addr_t.clear(); */
		/* for (long i = 0; i < cube_s; i++)
		{
			long t;
			localinFile() >> t;
			if (localinFile().eof())
			{
				return;
			}
			sor_addr_t.push_back(t);
		}
		//read destination address
		for (long i = 0; i < cube_s; i++)
		{
			long t;
			localinFile() >> t;
			if (localinFile().eof())
			{
				return;
			}
			des_addr_t.push_back(t);
		}
		//read packet size
		localinFile() >> pack_size_t; */
		if(localInputTraces.empty())return;
		SPacket&p=localInputTraces.front();
		inject_packet(packet_counter_, /* sor_addr_t */p.sourceAddress,
					  /* des_addr_t */p.destinationAddress, local_input_time_, /* pack_size_t */p.packetSize);
		packet_counter_++;

		//second, create next EVG_ event
		/* if (!localinFile().eof())
		{
			localinFile() >> local_input_time_;
			if (localinFile().eof())
			{
				return;
			}
		} */
		localInputTraces.pop();
		if(!localInputTraces.empty()){
			local_input_time_=localInputTraces.front().startTime;
		}
	}
}
//***************************************************************************//
//a : flit id. b: sor add. c: des add. d: start time. e: size
void sim_router_template::inject_packet(long a, add_type &b, add_type &c,
										time_type d, long e)
{
	// if it is the HEADER_ flit choose the shortest waiting vc queue
	// next state, it should choose routing
	VC_type vc_t;
	//changed at 2020-5-8
	//大小小于2的包会不会有问题？
	//有问题……
	//if(e<2)e=2;
	//修正：用SINGLE，既代表头又代表尾
	/* if(e==1){
		Data_type flitData;
		long i;
		for(i=0;i<flit_size_;++i){
			init_data_[i]=(Atom_type)(init_data_[i]*CORR_EFF_+SRGen::wrg().flat_ull(0,MAX_64_));
			flitData.push_back(init_data_[i]);
		}
		vc_t 
		return;
	} */
	//changed at 2020-5-12
	//检查是否有不符合要求的输入
	const char WRONG_ADDRESS[] = "Coordinate out of range";
	/* for(auto&x:b){
		if(x>=ary_size_){
			cerr<<WRONG_ADDRESS<<endl;
			throw WRONG_ADDRESS;
		}
	}
	for(auto&x:c){
		if(x>=ary_size_){
			cerr<<WRONG_ADDRESS<<endl;
			throw WRONG_ADDRESS;
		}
	} */
	auto outOfRange = [&](long x) {
		return x < 0 || x >= ary_size_;
	};
	if (any_of(b.begin(), b.end(), outOfRange) || any_of(c.begin(), c.end(), outOfRange))
	{
		cerr << WRONG_ADDRESS << endl;
		throw WRONG_ADDRESS;
	}
	for (long l = 0; l < e; l++)
	{
		Data_type flit_data;
		for (long i = 0; i < flit_size_; i++)
		{
			init_data_[i] = static_cast<Atom_type>(init_data_[i] * CORR_EFF_ + SRGen::wrg().flat_ull(0, MAX_64_));
			flit_data.push_back(init_data_[i]);
		}
		if (l == 0)
		{
			vc_t = pair<long, long>(0, input_module_.input(0, 0).size());
			for (long i = 0; i < vc_number_; i++)
			{
				long t = input_module_.input(0, i).size();
				if (vc_t.second > t)
				{
					vc_t = pair<long, long>(i, t);
				}
			}
			//if the input buffer is empty, set it to be ROUTING_
			if (input_module_.input(0, vc_t.first).size() == 0)
			{
				input_module_.state_update(0, vc_t.first, ROUTING_);
			}
			//if the input buffer has more than predefined flits, then
			// add the flits and sign a flag
			if (input_module_.input(0, vc_t.first).size() > 100)
			{
				input_module_.ibuff_is_full();
			}
			//changed at 2020-5-9
			/* input_module_.add_flit(0, (vc_t.first),  
								flit_template(a, HEADER_, b, c, d, flit_data)); */
			input_module_.add_flit(0, vc_t.first, flit_template(a, e == 1 ? SINGLE : HEADER_, b, c, d, flit_data));
		}
		else if (l == (e - 1))
		{
			input_module_.add_flit(0, (vc_t.first),
								   flit_template(a, TAIL_, b, c, d, flit_data));
		}
		else
		{
			input_module_.add_flit(0, (vc_t.first),
								   flit_template(a, BODY_, b, c, d, flit_data));
		}
		power_module_.power_buffer_write(0, flit_data);
	}
}

//***************************************************************************//
//receive a flit from other router
//pc, vc, flit
void sim_router_template::receive_flit(long a, long b, flit_template &c)
{
	input_module_.add_flit(a, b, c);
	power_module_.power_buffer_write(a, c.data());
	//changed at 2020-5-9
	if (/* c.type() == HEADER_ */ isHeader(c.type()))
	{
		if (input_module_.input(a, b).size() == 1)
		{
			input_module_.state_update(a, b, ROUTING_);
		}
		else
		{
			//changed at 2020-5-8
			/* if (count_if(input_module_.input(a, b).begin(), input_module_.input(a, b).end(), [](const flit_template &flit) {
					return isHeader(flit.type());
				}) > 1)
			{
				cout << "input size: " << input_module_.input(a, b).size() << " pc: " << a << " vc: " << b << endl;
				cout << "Router ";
				for (auto &x : address_)
					cout << x << ' ';
				cout << endl;
				for (auto &x : input_module_.input(a, b))
				{
					cout << "From ";
					for (auto &y : x.sor_addr())
						cout << y << ' ';
					cout << "to ";
					for (auto &y : x.des_addr())
						cout << y << ' ';
					cout << endl;
				}
			} */
		}
	}
	else
	{
		if (input_module_.state(a, b) == INIT_)
		{
			input_module_.state_update(a, b, SW_AB_);
		}
	}
}

//***************************************************************************//
//switch arbitration pipeline stage
void sim_router_template::sw_arbitration()
{
	map<long, vector<VC_type>> vc_o_map;
	for (long i = 0; i < physic_ports_; i++)
	{
		vector<long> vc_i_t;
		for (long j = 0; j < vc_number_; j++)
		{
			if (input_module_.state(i, j) == SW_AB_)
			{
				VC_type out_t = input_module_.crouting(i, j);
				if ((output_module_.counter(out_t.first, out_t.second) > 0) &&
					(output_module_.localcounter(out_t.first) > 0))
				{
					vc_i_t.push_back(j);
				}else{
					//changed at 2020-5-22
					/* if(output_module_.counter(out_t.first, out_t.second)==0){
						cout<<"Full output buffer of Router ";
						for(auto&x:address_)cout<<x<<' ';
						cout<<"Port "<<i<<" Virtual Channel "<<j;
						cout<<endl;
					} */
				}
			}
		}
		long vc_size_t = vc_i_t.size();
		if (vc_size_t > 1)
		{
			long win_t = SRGen::wrg().flat_l(0, vc_size_t);
			VC_type r_t = input_module_.crouting(i, vc_i_t[win_t]);
			vc_o_map[r_t.first].push_back(VC_type(i, vc_i_t[win_t]));
		}
		else if (vc_size_t == 1)
		{
			VC_type r_t = input_module_.crouting(i, vc_i_t[0]);
			vc_o_map[r_t.first].push_back(VC_type(i, vc_i_t[0]));
		}
	}

	if (vc_o_map.size() == 0)
	{
		return;
	}

	for (long i = 0; i < physic_ports_; i++)
	{
		long vc_size_t = vc_o_map[i].size();
		if (vc_size_t > 0)
		{
			VC_type vc_win = vc_o_map[i][0];
			if (vc_size_t > 1)
			{
				vc_win = vc_o_map[i][SRGen::wrg().flat_l(0,
														 vc_size_t)];
			}
			input_module_.state_update(vc_win.first, vc_win.second,
									   SW_TR_);
			flit_template &flit_t = input_module_.get_flit(
				vc_win.first, vc_win.second);
		}
	}
}

inline bool isCentral(const add_type&address,long centerXY)
{
	return address[0]==centerXY&&address[1]==centerXY&&address[2]==0&&address[3]==0;
}
//获得微片的上一个路由器的地址
void sim_router_template::getFromRouter(add_type&from,long port)
{
	/* switch(routing_alg_){
		case XY_:
		case TXY_:
		case CHIPLET_ROUTING_MESH:
			getFromRouter_mesh(from,port);
			break;
		case CHIPLET_STAR_TOPO_ROUTING:
			getFromRouter_chipletStar(from,port);
			break;
		default:
			throw "Error: Wrong type of routing algorithm";
	} */
	(this->*curr_prevRouterFunc)(from,port);
}
void sim_router_template::getFromRouter_mesh(add_type&from,long port)
{
	//端口所对应的相邻路由器是唯一的
	getNextAddress_mesh(from,port);
}
void sim_router_template::getFromRouter_chipletStar(add_type&from,long port)
{
	//获得微片的上一个的路由器
	long centerXY=ary_size_-1;
	long t=port-1;
	if(isCentral(address_,centerXY)){
		//中心结点
		ldiv_t xy=ldiv(t,centerXY);
		from={xy.quot,xy.rem,0,0};
	}else{
		long p=t>>1;
		if(p==address_.size())from={centerXY,centerXY,0,0};
		else{
			from=address_;
			if(t&1){
				from[p]++;
				if(from[p]==ary_size_)from[p]=0;
			}else{
				from[p]--;
				if(from[p]==-1)from[p]=ary_size_-1;
			}
		}
	}
}
//获得该微片是从上一个路由器哪一个输出端口出来的
long sim_router_template::getFromPort(long port)
{
	/* switch(routing_alg_){
		case XY_:
		case TXY_:
		case CHIPLET_ROUTING_MESH:
			return getFromPort_mesh(port);
		case CHIPLET_STAR_TOPO_ROUTING:
			return getFromPort_chipletStar(port);
		default:
			throw "Error: Wrong type of routing algorithm";
	} */
	(this->*curr_prevPortFunc)(port);
}
long sim_router_template::getFromPort_mesh(long port)
{
	//端口所对应的相邻路由器的端口是唯一的
	return getWirePc_mesh(port);
}
long sim_router_template::getFromPort_chipletStar(long port)
{
	long centerXY=ary_size_-1;
	long centerPort=((long)address_.size()<<1)+1;
	if(isCentral(address_,centerXY)){
		return centerPort;
	}
	if(centerPort==port)return address_[0]*centerXY+address_[1]+1;
	else return port+(port&1?1:-1);
}
//commented at 2020-5-22
//将微片从输入缓存传给输出缓存
//***************************************************************************//
//flit out buffer to the output buffer
void sim_router_template::flit_outbuffer()
{
	for (long i = 0; i < physic_ports_; i++)
	{
		for (long j = 0; j < vc_number_; j++)
		{
			if (input_module_.state(i, j) == SW_TR_)
			{
				VC_type out_t = input_module_.crouting(i, j);
				output_module_.counter_dec(out_t.first, out_t.second);

				time_type event_time = mess_queue::m_pointer().current_time();
				if (i != 0)
				{
					//changed at 2020-5-22
					add_type cre_add_t/*  = address_ */;
					long cre_pc_t/*  = i */;
					/* if ((i % 2) == 0)
					{
						cre_pc_t = i - 1;
						cre_add_t[(i - 1) / 2]++;
						if (cre_add_t[(i - 1) / 2] == ary_size_)
						{
							cre_add_t[(i - 1) / 2] = 0;
						}
					}
					else
					{
						cre_pc_t = i + 1;
						cre_add_t[(i - 1) / 2]--;
						if (cre_add_t[(i - 1) / 2] == -1)
						{
							cre_add_t[(i - 1) / 2] = ary_size_ - 1;
						}
					} */
					getFromRouter(cre_add_t,i);
					cre_pc_t=getFromPort(i);
					//cre_pc_t=getWirePc(i);
					//commented at 2020-5-22
					//发送CREDIT事件，让原来的输出缓存空位数加一
					mess_queue::wm_pointer().add_message(
						mess_event(event_time + CREDIT_DELAY_,
								   CREDIT_, address_, cre_add_t, cre_pc_t, j));
				}

				long in_size_t = input_module_.input(i, j).size();
				Sassert(in_size_t >= 1);
				flit_template flit_t(input_module_.get_flit(i, j));
				input_module_.remove_flit(i, j);
				power_module_.power_buffer_read(i, flit_t.data());
				power_module_.power_crossbar_trav(i, out_t.first, flit_t.data());
				output_module_.add_flit(out_t.first, flit_t);
				if (i == 0)
				{
					if (input_module_.ibuff_full() == true)
					{
						if (input_module_.input(0, j).size() < BUFF_BOUND_)
						{
							input_module_.ibuff_not_full();
							receive_packet();
						}
					}
				}
				output_module_.add_add(out_t.first, out_t);
				//changed at 2020-5-9
				if (/* flit_t.type() == TAIL_ */ isTail(flit_t.type()))
				{
					output_module_.release(out_t.first, out_t.second);
				}
				if (in_size_t > 1)
				{
					//changed at 2020-5-9
					if (/* flit_t.type() == TAIL_ */ isTail(flit_t.type()))
					{
						if (configuration::ap().vc_share() == MONO_)
						{
							if (i != 0)
							{
								if (in_size_t != 1)
								{
									cout << i << ":" << in_size_t << endl;
								}
								Sassert(in_size_t == 1);
							}
						}
						input_module_.state_update(i, j, ROUTING_);
					}
					else
					{
						input_module_.state_update(i, j, SW_AB_);
					}
				}
				else
				{
					input_module_.state_update(i, j, INIT_);
				}
			}
		}
	}
}

/* const time_type SHORT_WIRE_DELAY = WIRE_DELAY_;
const time_type LONG_WIRE_DELAY_X = WIRE_DELAY_;
const time_type LONG_WIRE_DELAY_Y = WIRE_DELAY_; */
const time_type FREQUENCY = 1;
const time_type SHORT_WIRE_DELAY=9.57*FREQUENCY;
const time_type LONG_WIRE_DELAY_X=153.16*FREQUENCY;
const time_type LONG_WIRE_DELAY_Y=153.16*FREQUENCY;
const time_type STAR_TOPO_2_2 = 76.5769 * FREQUENCY;
const time_type STAR_TOPO_4_2_INNER=47.8721*FREQUENCY;
const time_type STAR_TOPO_4_2_OUTER=124.4789*FREQUENCY;
const time_type STAR_TOPO_4_4_INNER=19.1442*FREQUENCY;
const time_type STAR_TOPO_4_4_OUTER=95.7453*FREQUENCY;
const time_type STAR_TOPO_4_4_CORNER=172.2980*FREQUENCY;
time_type sim_router_template::getWireDelay_mesh(long port)
{
	return WIRE_DELAY_;
}
time_type sim_router_template::getWireDelay_chipletMesh(long port)
{
	return (port<=5?(port<3?LONG_WIRE_DELAY_X:LONG_WIRE_DELAY_Y):SHORT_WIRE_DELAY);
}
time_type sim_router_template::getWireDelay_chipletStar(long port)
{
	time_type r = 0;
	/* switch(routing_alg_){
		case XY_:
		case TXY_:
			return WIRE_DELAY_;
		case CHIPLET_ROUTING_MESH:
			return (port<=5?(port<3?LONG_WIRE_DELAY_X:LONG_WIRE_DELAY_Y):SHORT_WIRE_DELAY);
	} */
	long centerXY=ary_size_-1;
	if(isCentral(address_,centerXY)){
		long chipletX=(port-1)/centerXY;
		bool inner=(chipletX==1||chipletX==2);
		r=inner?STAR_TOPO_4_2_INNER:STAR_TOPO_4_2_OUTER;
		/* ldiv_t xy=ldiv(port-1,centerXY);
		int d=0;
		if(xy.quot==1||xy.quot==2)d++;
		if(xy.rem==1||xy.rem==2)d++;
		switch(d){
			case 2:
			r=STAR_TOPO_4_4_INNER;
			break;
			case 1:
			r=STAR_TOPO_4_4_OUTER;
			break;
			case 0:
			r=STAR_TOPO_4_4_CORNER;
			break;
		} */
	}
	else if(((long)address_.size()<<1)+1==port){
		r=(address_[0]==1||address_[0]==2)?STAR_TOPO_4_2_INNER:STAR_TOPO_4_2_OUTER;
		/* int d=0;
		if(address_[0]==1||address_[1]==2)d++;
		if(address_[1]==2||address_[1]==2)d++;
		switch(d){
			case 0:
			r=STAR_TOPO_4_4_CORNER;
			break;
			case 1:
			r=STAR_TOPO_4_4_OUTER;
			break;
			case 2:
			r=STAR_TOPO_4_4_INNER;
			break;
		} */
	}
	else r=SHORT_WIRE_DELAY;
	//return WIRE_DELAY_;
	return r;
}
//计算下一跳的延迟
time_type sim_router_template::getWireDelay(long port)
{
	return (this->*curr_wireDelayFunc)(port);
}
//根据输出端口计算下一跳路由器的地址
void sim_router_template::getNextAddress(add_type &nextAddress, long port)
{
	/* switch(routing_alg_){
		case XY_:
		case TXY_:
			getNextAddress_mesh(nextAddress,port);
			break;
		case CHIPLET_ROUTING_MESH:
			getNextAddress_chipletMesh(nextAddress,port);
			break;
		case CHIPLET_STAR_TOPO_ROUTING:
			getNextAddress_chipletStar(nextAddress,port);
			break;
		default:
			throw "Error: Wrong type of routing algorithm";
	} */
	(this->*curr_nextAddFunc)(nextAddress,port);
}
inline void remainderAdd(long&dividend,long divisor)
{
	dividend++;
	if(dividend==divisor)dividend=0;
}
inline void remainderReduce(long&dividend,long divisor)
{
	if(dividend==0)dividend=divisor-1;
	else dividend--;
}
void sim_router_template::getNextAddress_mesh(add_type&nextAdd,long port)
{
	long t=port-1;
	nextAdd=address_;
	if(t&1){
		remainderAdd(nextAdd[t>>1],ary_size_);
	}else remainderReduce(nextAdd[t>>1],ary_size_);
}
void sim_router_template::getNextAddress_chipletMesh(add_type&nextAdd,long port)
{
	getNextAddress_mesh(nextAdd,port);
}
void sim_router_template::getNextAddress_chipletStar(add_type &nextAddress, long port)
{
	long t = port - 1;
	long centerXY = ary_size_ - 1;
	if (address_[0] == centerXY && address_[1] == centerXY && address_[2] == 0 && address_[3] == 0)
	{
		//中心结点
		ldiv_t xy = ldiv(t, centerXY);
		nextAddress = {xy.quot, xy.rem, 0, 0};
	}
	else
	{
		long p = t >> 1;
		if (p == address_.size())
			nextAddress = {centerXY, centerXY, 0, 0};
		else
		{
			nextAddress = address_;
			if (t & 1)
			{
				nextAddress[p]++;
				if (nextAddress[p] == ary_size_)
					nextAddress[p] = 0;
			}
			else
			{
				nextAddress[p]--;
				if (nextAddress[p] == -1)
					nextAddress[p] = ary_size_ - 1;
			}
		}
	}
	/* cout<<"Router ";
	for(auto&x:address_)cout<<x<<' ';
	cout<<endl; */
}
//根据输出端口计算下一跳路由器的输入端口
long sim_router_template::getWirePc(long port)
{
	/* switch(routing_alg_){
		case XY_:
		case TXY_:
		case CHIPLET_ROUTING_MESH:
			return getWirePc_mesh(port);
		case CHIPLET_STAR_TOPO_ROUTING:
			return getWirePc_chipletStar(port);
		default:
			throw "Error: Wrong type of routing algorithm";
	} */
	(this->*curr_wirePcFunc)(port);
}
long sim_router_template::getWirePc_mesh(long port)
{
	//正方向为双数，负方向为单数
	return port-1+((port&1)<<1);
}
long sim_router_template::getWirePc_chipletStar(long port)
{
	long centerPort=((long)address_.size()<<1)+1;
	long centerXY=ary_size_-1;
	if(isCentral(address_,centerXY)){
		return centerPort;
	}
	if (port==centerPort)
		return address_[0] *centerXY+ address_[1] + 1;
	else
		return port + (port & 1 ? 1 : -1);
}
//commented at 2020-5-7
//将在输出缓存的微片（flit）传给下一跳的路由器
//这里有物理（输出）端口与相邻路由器的对应关系
//***************************************************************************//
//flit traversal through the link stage
void sim_router_template::flit_traversal(long i)
{
	time_type event_time = mess_queue::m_pointer().current_time();
	if (output_module_.outbuffers(i).size() > 0)
	{
		time_type delay = getWireDelay(i);
		time_type flit_delay_t = delay + event_time;
		//changed at 2020-5-20
		add_type wire_add_t /* = address_ */;
		long wire_pc_t = getWirePc(i);
		/* if((i % 2) == 0) {
			wire_pc_t = i - 1;
			wire_add_t[(i - 1) / 2] ++;
			if(wire_add_t[(i-1) / 2] == ary_size_) {
				wire_add_t[(i-1) / 2] = 0;
			}
		}else {
			wire_pc_t = i + 1;
			wire_add_t[(i - 1) / 2] --;
			if(wire_add_t[(i-1) / 2] == -1) {
				wire_add_t[(i-1) / 2] = ary_size_ - 1;
			}
		} */
		getNextAddress(wire_add_t, i);
		flit_template flit_t(output_module_.get_flit(i));
		VC_type outadd_t = output_module_.get_add(i);
		power_module_.power_link_traversal(i, flit_t.data());

		output_module_.remove_flit(i);
		output_module_.remove_add(i);
		mess_queue::wm_pointer().add_message(mess_event(flit_delay_t,
														WIRE_, address_, wire_add_t, wire_pc_t,
														outadd_t.second, flit_t));
		//changed at 2020-5-8
		//输出下一跳路由
		/* cout << "Router ";
		for (auto &x : address_)
			cout << x << ' ';
		cout << "give a ";
		cout << flit_t.type() << " flit to Router ";
		for (auto &x : wire_add_t)
			cout << x << ' ';
		cout<<"from ";
		for(auto&x:flit_t.sor_addr())cout<<x<<' ';
		cout<<" to ";
		for(auto&x:flit_t.des_addr())cout<<x<<' ';
		cout << endl; */
	}
}

//***************************************************************************//
//receive the flit at the destination router
void sim_router_template::accept_flit(time_type a, const flit_template &b)
{
	//changed at 2020-5-9
	static ofstream ofs("delayInfo.txt");
	if (/* b.type() == TAIL_ */ isTail(b.type()))
	{
		mess_queue::wm_pointer().TotFin_inc();
		time_type t = a - b.start_time();
		delay_update(t);
		//发送时间（周期），源地址，目的地址，延迟（周期）
		ofs << b.start_time() << ' '
			/* <<a<<' ' */;
		for (auto &x : b.sor_addr())
			ofs << x << ' ';
		for (auto &x : b.des_addr())
			ofs << x << ' ';
		ofs << t << endl;
	}
}

//***************************************************************************//
//flit traversal through link
void sim_router_template::flit_traversal()
{
	for (long i = 1; i < physic_ports_; i++)
	{
		flit_traversal(i);
	}
}
//***************************************************************************//
//routing pipeline stages
void sim_router_template::router_sim_pwr()
{
	//stage 5 flit traversal
	flit_traversal();
	//stage 4 flit output buffer
	flit_outbuffer();
	//stage 3 switch arbitration
	sw_arbitration();
	//stage 2, vc arbitration
	vc_arbitration();
	//stage 1, routing decision
	routing_decision();
}

//***************************************************************************//

void sim_router_template::empty_check() const
{
	for (long i = 0; i < physic_ports_; i++)
	{
		for (long j = 0; j < vc_number_; j++)
		{
			if (input_module_.input(i, j).size() > 0)
			{
				cout << "Input is not empty" << endl;
				//Sassert(0);
			}
			if (input_module_.state(i, j) != INIT_)
			{
				cout << "Input state is wrong" << endl;
				//Sassert(0);
			}
			cout << output_module_.counter(i, j) << ":";
			if (output_module_.counter(i, j) != buffer_size_)
			{
				cout << "Output vc counter is wrong" << endl;
				//Sassert(0);
			}
			if (output_module_.usage(i, j) != FREE_)
			{
				cout << "Output is not free" << endl;
				//Sassert(0);
			}
			if (output_module_.assign(i, j) != VC_NULL)
			{
				cout << "Output is not free" << endl;
				//Sassert(0);
			}
		}
		if (output_module_.outbuffers(i).size() > 0)
		{
			cout << "Output temp buffer is not empty" << endl;
			//Sassert(0);
		}
		if (output_module_.outadd(i).size() > 0)
		{
			cout << "Output temp buffer is not empty" << endl;
			//Sassert(0);
		}
		if (output_module_.localcounter(i) != outbuffer_size_)
		{
			cout << "Output local counter is not reset" << endl;
			//Sassert(0);
		}
	}
}
