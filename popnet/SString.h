#ifndef NETWORK_STRING_H_
#define NETWORK_STRING_H_

/*###########################################################################*/

#include <string>
#include <strstream>

using namespace std;

/*===========================================================================*/
// Automatically converts strings to many other types.

class Conv {
public:
	Conv(const string & str);

	operator bool();

	operator signed char();
	operator unsigned char();

	operator short();
	operator unsigned short();

	operator int();
	operator unsigned();

	operator long();
	operator unsigned long();

	operator float();
	operator double();

private:
	string str_;
};

/*###########################################################################*/
#endif
