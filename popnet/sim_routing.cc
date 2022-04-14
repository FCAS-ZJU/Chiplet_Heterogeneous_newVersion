#include "sim_router.h"
#include "sim_foundation.h"
#include "mess_queue.h"
#include "mess_event.h"
#include "SRGen.h"
#include "SStd.h"


//***************************************************************************/
void sim_router_template::TXY_algorithm(const add_type & des_t,
		const add_type & sor_t, long s_ph, long s_vc)
{
	long xoffset = des_t[0] - address_[0];
	long yoffset = des_t[1] - address_[1];
	bool xdirection = (abs(static_cast<int>(xoffset)) * 2 
					<= ary_size_)? true: false; 
	bool ydirection = (abs(static_cast<int>(yoffset)) * 2 
					<= ary_size_)? true: false; 

	if(xdirection) {
		if(xoffset < 0) {
			input_module_.add_routing(s_ph, s_vc, VC_type(1, 0));
		}else if(xoffset > 0) {
			input_module_.add_routing(s_ph, s_vc, VC_type(2, 1));
		}else {
			if(ydirection) {
				if(yoffset < 0) {
					input_module_.add_routing(s_ph, s_vc, VC_type(3, 0));
				}else if(yoffset > 0) {
					input_module_.add_routing(s_ph, s_vc, VC_type(4, 1));
				}
			}else {
				if(yoffset < 0) {
					input_module_.add_routing(s_ph, s_vc, VC_type(4, 0));
				}else if(yoffset > 0) {
					input_module_.add_routing(s_ph, s_vc, VC_type(3, 1)); 
				}
			}
		}
	}else  {
		if(xoffset < 0) {
			input_module_.add_routing(s_ph, s_vc, VC_type(2, 0));
		}else if(xoffset > 0) {
			input_module_.add_routing(s_ph, s_vc, VC_type(1, 1));
		}else {
			if(ydirection) {
				if(yoffset < 0) {
					input_module_.add_routing(s_ph, s_vc, VC_type(3, 0));
				}else if(yoffset > 0) {
					input_module_.add_routing(s_ph, s_vc, VC_type(4, 1));
				}
			}else {
				if(yoffset < 0) {
					input_module_.add_routing(s_ph, s_vc, VC_type(4, 0));
				}else if(yoffset> 0) {
					input_module_.add_routing(s_ph, s_vc, VC_type(3, 1)); 
				}
			}
		}
	}
}

//***************************************************************************//
void sim_router_template::XY_algorithm(const add_type & des_t,
		const add_type & sor_t, long s_ph, long s_vc)
{
	long xoffset = des_t[0] - address_[0];
	long yoffset = des_t[1] - address_[1];

	if(yoffset < 0) {
		input_module_.add_routing(s_ph, s_vc, VC_type(3,0));
		input_module_.add_routing(s_ph, s_vc, VC_type(3,1));
		input_module_.add_routing(s_ph, s_vc, VC_type(3,2));
		input_module_.add_routing(s_ph, s_vc, VC_type(3,3));
	}else if(yoffset > 0) {
		input_module_.add_routing(s_ph, s_vc, VC_type(4,0));
		input_module_.add_routing(s_ph, s_vc, VC_type(4,1));
		input_module_.add_routing(s_ph, s_vc, VC_type(4,2));
		input_module_.add_routing(s_ph, s_vc, VC_type(4,3));
	}else {
		if(xoffset < 0) {
			input_module_.add_routing(s_ph, s_vc, VC_type(1,0));
			input_module_.add_routing(s_ph, s_vc, VC_type(1,1));
			input_module_.add_routing(s_ph, s_vc, VC_type(1,2));
			input_module_.add_routing(s_ph, s_vc, VC_type(1,3));
		}else if (xoffset > 0) {
			input_module_.add_routing(s_ph, s_vc, VC_type(2,0));
			input_module_.add_routing(s_ph, s_vc, VC_type(2,1));
			input_module_.add_routing(s_ph, s_vc, VC_type(2,2));
			input_module_.add_routing(s_ph, s_vc, VC_type(2,3));
		}
	}
}

//changed at 2020-5-6
//芯粒路由算法
void sim_router_template::chiplet_routing_alg(const add_type & des_t,const add_type & src_t, long s_ph, long s_vc)
{
	//地址格式：芯粒横坐标，芯粒纵坐标，核心横坐标，核心纵坐标
	//芯粒间和芯粒内使用XY路由。
	//芯粒内(0,0)核心为网关。
	const long VIRTUAL_CHANNEL_COUNT=2;
	add_type delta;
	size_t addr_len=des_t.size();
	long i;
	delta.reserve(addr_len);
	for(i=0;i<addr_len;++i)delta.push_back(des_t[i]-address_[i]);
	if(delta[0]==0&&delta[1]==0){
		//芯粒内
		VC_type vc;
		if(delta[2]==0)vc.first=(delta[3]<0?7:8);
		else vc.first=(delta[2]<0?5:6);
		for(i=0;i<VIRTUAL_CHANNEL_COUNT;++i){
			vc.second=i;
			input_module_.add_routing(s_ph,s_vc,vc);
		}
	}else{
		//跨芯粒
		long vc_first;
		if(address_[2]==0&&address_[3]==0){
			//vc.first赋值为5-8不知行不行？
			//改了，现在可以
			if(delta[0]==0)vc_first=(delta[1]<0?3:4);
			else vc_first=(delta[0]<0?1:2);
			for(i=0;i<VIRTUAL_CHANNEL_COUNT;++i)input_module_.add_routing(s_ph,s_vc,VC_type(vc_first,i));
		}else{
			if(address_[2]>0)vc_first=5;
			else vc_first=7;
			for(i=0;i<VIRTUAL_CHANNEL_COUNT;++i)input_module_.add_routing(s_ph,s_vc,VC_type(vc_first,i));
		}
	}
}

//changed at 2020-5-19
void addRoutingForDifferentVC(input_template&inputModule,long s_ph,long s_vc,long port)
{
	const long VIRTUAL_CHANNEL_COUNT=2;
	for(long i=0;i<VIRTUAL_CHANNEL_COUNT;++i){
		inputModule.add_routing(s_ph,s_vc,VC_type(port,i));
	}
}
bool checkAddressIndex(const add_type&d,long idx,long&port)
{
	//根据本地地址与目的地址的差计算出端口号
	if(d[idx]==0)return true;
	port=(idx<<1)+(d[idx]>0?2:1);
	return false;
}
//changed at 2020-5-19
//增加了星型拓扑的芯粒路由算法
void sim_router_template::chiplet_star_topo_routing_alg(
	const add_type&des_t,
	const add_type&src_t,
	long s_ph, 
	long s_vc)
{
	//星型拓扑
	long addrLen=address_.size();
	add_type delta/* (addrLen) */;
	long i,port;
	for(i=0;i<addrLen;++i)delta.push_back(des_t[i]-address_[i]);
	long centerXY=ary_size_-1;
	if(address_[0]==centerXY&&address_[1]==centerXY&&address_[2]==0&&address_[3]==0){
		//中心节点坐标：(n-1,n-1,0,0)
		addRoutingForDifferentVC(input_module_,s_ph,s_vc,des_t[0]*centerXY+des_t[1]+1);
	}else if(delta[0]==0&&delta[1]==0){
		//芯粒内
		if(checkAddressIndex(delta,2,port))checkAddressIndex(delta,3,port);
		addRoutingForDifferentVC(input_module_,s_ph,s_vc,port);
	}else{
		//跨芯粒
		if(address_[2]==0&&address_[3]==0){
			addRoutingForDifferentVC(input_module_,s_ph,s_vc,(addrLen<<1)+1);
		}else{
			//if(checkAddressIndex(address_,2,port))checkAddressIndex(address_,3,port);
			port=(address_[2]>0?5:7);
			addRoutingForDifferentVC(input_module_,s_ph,s_vc,port);
		}
	}
}
			
//***************************************************************************//
//only two-dimension is supported
void sim_router_template::routing_decision()
{
	time_type event_time = mess_queue::m_pointer().current_time();

	//for injection physical port 0
	for(long j = 0; j < vc_number_; j++) {
		//for the HEADER_ flit
		flit_template flit_t;
		if(input_module_.state(0,j) == ROUTING_) {
			flit_t = input_module_.get_flit(0,j);
			add_type des_t = flit_t.des_addr();
			add_type sor_t = flit_t.sor_addr();
			if(address_ == des_t) {
				accept_flit(event_time, flit_t);
				input_module_.remove_flit(0, j);
				//changed at 2020-5-9
				//input_module_.state_update(0, j, HOME_);
				VC_state_type vcst;
				if(flit_t.type()==HEADER_)vcst=HOME_;
				else{
					vcst=(input_module_.input(0,j).size()>0?ROUTING_:INIT_);
				}
				input_module_.state_update(0, j, vcst);
			}else {
				input_module_.clear_routing(0,j);
				input_module_.clear_crouting(0,j);
				(this->*curr_algorithm)(des_t, sor_t, 0, j);
				input_module_.state_update(0, j, VC_AB_);
			}
		//the BODY_ or TAIL_ flits
		}else if(input_module_.state(0,j) == HOME_)  {
			if(input_module_.input(0, j).size() > 0) {
				flit_t = input_module_.get_flit(0, j);
				//changed at 2020-5-9
				Sassert(/* flit_t.type() != HEADER_ */!isHeader(flit_t.type()));
				accept_flit(event_time, flit_t);
				input_module_.remove_flit(0, j);
				if(flit_t.type() == TAIL_) {
					if(input_module_.input(0, j).size() > 0) {
						input_module_.state_update(0, j, ROUTING_);
					}else {
						input_module_.state_update(0, j, INIT_);
					}
				}
			}
		}
	}

	//for other physical ports
	for(long i = 1; i < physic_ports_; i++) {
		for(long j = 0; j < vc_number_; j++) {
			//send back CREDIT message
			flit_template flit_t;
			if(input_module_.input(i,j).size() > 0) {
				flit_t = input_module_.get_flit(i,j);
				add_type des_t = flit_t.des_addr();
				if(address_ == des_t) {
					//changed at 2020-5-23
					//发送CREDIT事件，让原来的输出缓存空位数加一
					add_type cre_add_t/*  = address_ */;
					long cre_pc_t/*  = i */;
					/* if((i % 2) == 0) {
						cre_pc_t = i - 1;
						cre_add_t[(i-1)/2] ++;
						if(cre_add_t[(i-1)/2] == ary_size_) {
							cre_add_t[(i-1)/2] = 0;
						}
					}else {
						cre_pc_t = i + 1;
						cre_add_t[(i-1)/2] --;
						if(cre_add_t[(i-1)/2] == -1) {
							cre_add_t[(i-1)/2] = ary_size_ - 1;
						}
					} */
					getFromRouter(cre_add_t,i);
					cre_pc_t=getFromPort(i);
					mess_queue::wm_pointer().add_message(
						mess_event(event_time + CREDIT_DELAY_, 
						CREDIT_, address_, cre_add_t, cre_pc_t, j));
				}
			}
			//for HEADER_ flit
			if(input_module_.state(i, j) == ROUTING_) {
				flit_t = input_module_.get_flit(i, j);
				//changed at 2020-5-9
				Sassert(/* flit_t.type() == HEADER_ */isHeader(flit_t.type()));
				add_type des_t = flit_t.des_addr();
				add_type sor_t = flit_t.sor_addr();
				//changed at 2020-5-8
				/* cout<<"Router ";
				for(auto&x:address_)cout<<x<<' ';
				cout<<"gets a flit from Router ";
				for(auto&x:sor_t)cout<<x<<' ';
				cout<<"to Router ";
				for(auto&x:des_t)cout<<x<<' ';
				cout<<endl; */
				if(address_ == des_t) {
					accept_flit(event_time, flit_t);
					input_module_.remove_flit(i, j);
					//changed at 2020-5-9
					//input_module_.state_update(i, j, HOME_);
					VC_state_type vcst;
					if(flit_t.type()==HEADER_)vcst=HOME_;
					else{
						vcst=(input_module_.input(i,j).size()>0?ROUTING_:INIT_);
					}
					input_module_.state_update(i, j, vcst);
				}else {
					input_module_.clear_routing(i, j);
					input_module_.clear_crouting(i, j);
					(this->*curr_algorithm)(des_t, sor_t, i, j);
					input_module_.state_update(i, j, VC_AB_);
				}
			//for BODY_ or TAIL_ flits
			}else if(input_module_.state(i, j) == HOME_) {
				if(input_module_.input(i, j).size() > 0) {
					flit_t = input_module_.get_flit(i, j);
					//changed at 2020-5-9
					Sassert(/* flit_t.type() != HEADER_ */!isHeader(flit_t.type()));
					accept_flit(event_time, flit_t);
					input_module_.remove_flit(i, j);
					if(flit_t.type() == TAIL_) {
						if(input_module_.input(i, j).size() > 0) {
							input_module_.state_update(i, j, ROUTING_);
						}else {
							input_module_.state_update(i, j, INIT_);
						}
					}
				}
			}
		}
	}
}

//***************************************************************************//
