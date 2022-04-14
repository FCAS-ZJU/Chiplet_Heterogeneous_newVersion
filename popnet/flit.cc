#include "flit.h"

ostream& operator<<(ostream& os, const flit_template & ft)
{
	os<<ft.flit_id_<<"::"<<static_cast<long> (ft.type_)<<":";
	long j = ft.sor_addr_.size();
	for(long i = 0 ; i <j; i++) {
		os << ft.sor_addr_[i]<<" ";
	}
	for(long i = 0 ; i <j; i++) {
		os << ft.des_addr_[i]<<" ";
	} os<<endl;
	return os;
}


flit_template::flit_template():
	flit_id_(),
	type_(HEADER_),
	sor_addr_(),
	des_addr_(),
	start_time_(),
	finish_time_(),
	data_()
{
}

flit_template::flit_template(long a, flit_type b, add_type & c, 
				 add_type & d, time_type e, const Data_type & f):
	flit_id_(a),
	type_(b),
	sor_addr_(c),
	des_addr_(d),
	start_time_(e),
	data_(f)
{
}

flit_template::flit_template(const flit_template & a):
	flit_id_(a.flit_id()),
	type_(a.type()),
	sor_addr_(a.sor_addr()),
	des_addr_(a.des_addr()),
	start_time_(a.start_time()),
	finish_time_(a.finish_time()),
	data_(a.data())
{
}

//changed at 2020-5-9
bool isHeader(flit_type ft)
{
	return ft==HEADER_||ft==SINGLE;
}
bool isTail(flit_type ft)
{
	return ft==TAIL_||ft==SINGLE;
}

//changed at 2020-5-23
ostream&operator<<(ostream&os,const add_type&address)
{
	bool b=true;
	for(auto&x:address){
		if(b)b=false;
		else os<<' ';
		os<<x;
	}
	return os;
}
