#include "mess_event.h"

mess_event::mess_event(time_type t, mess_type mt):
    time_(t),
	mess_(mt),
	src_(),
	des_(),
	pc_(),
	vc_(),
	flit_()
{
}

mess_event::mess_event(time_type t, mess_type mt,
		const add_type & a, const add_type & b, long c, long d):
    time_(t),
	mess_(mt),
	src_(a),
	des_(b),
	pc_(c),
	vc_(d),
	flit_()
{
}

mess_event::mess_event(time_type t, mess_type mt,
	const add_type & a, const add_type & b, long c, long d, 
										const flit_template & f):
    time_(t),
	mess_(mt),
	src_(a),
	des_(b),
	pc_(c),
	vc_(d),
	flit_(f)
{
}

mess_event::mess_event(time_type t, mess_type mt, const flit_template & f):
    time_(t),
	mess_(mt),
	src_(),
	des_(),
	pc_(),
	vc_(),
	flit_(f)
{
}

mess_event::mess_event(mess_event& me):
	time_(me.event_start()),
	mess_(me.event_type()),
	src_(me.src()),
	des_(me.des()),
	pc_(me.pc()),
	vc_(me.vc()),
	flit_(me.get_flit())
{
}

mess_event::mess_event(const mess_event& me):
	time_(me.event_start()),
	mess_(me.event_type()),
	src_(me.src()),
	des_(me.des()),
	pc_(me.pc()),
	vc_(me.vc()),
	flit_(me.get_flit())
{
}

ostream& operator<<(ostream & os, const mess_event & sg) 
{
	os <<"time: "<<sg.time_<<"\tmess type: "<<sg.mess_<<endl;
	return os;
}
