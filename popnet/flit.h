#ifndef SIM_FLIT_H_
#define SIM_FLIT_H_

#include "index.h"
#include "SStd.h"
#include <vector>
#include <iostream>
#include <vector>

//数据包，用于整个网络传递，从发送方到接收方
//最大尺寸是预设的，若超过要分片传输
class flit_template {
	friend ostream& operator<<(ostream& os, const flit_template & ft);
	private:
		long flit_id_;
		flit_type type_;
		add_type sor_addr_;
		add_type des_addr_;
		time_type start_time_;
		time_type finish_time_;
		Data_type data_;
	
	public:
		long flit_id() {return flit_id_;}
		long flit_id() const {return flit_id_;}
		flit_type type() {return type_;}
		flit_type type() const {return type_;}
		add_type & sor_addr() {return sor_addr_;}
		const add_type & sor_addr() const {return sor_addr_;}
		add_type & des_addr() {return des_addr_;}
		const add_type & des_addr() const {return des_addr_;}

		time_type start_time() {return start_time_;}
		time_type start_time() const {return start_time_;}

		time_type finish_time() {return finish_time_;}
		time_type finish_time() const {return finish_time_;}

		void set_start(time_type a) {start_time_ = a;}
		void set_finish(time_type a) {finish_time_ = a;}

		Data_type & data() {return data_;}
		const Data_type & data() const {return data_;}

		flit_template();
		flit_template(long a, flit_type b, add_type & c, add_type & d,
					time_type e, const Data_type & f);
		flit_template(const flit_template & a);
};

//changed at 2020-5-9
bool isHeader(flit_type ft);
bool isTail(flit_type ft);

//changed at 2020-5-23
ostream&operator<<(ostream&os,const add_type&address);

#endif
		

