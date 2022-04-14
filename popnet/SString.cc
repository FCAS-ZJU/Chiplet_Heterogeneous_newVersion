#include "SString.h"
#include "SStd.h"

#include <algorithm>
#include <functional>
#include <string>
#include <cstring>
#include <iostream>
/*###########################################################################*/
namespace {
    template <typename T>
    T convert(const string & s) {
        istrstream iss(s.c_str());
        double dbl;
        iss >> dbl;
        Sassert(! iss.bad());
        return static_cast<T>(dbl);
    }
}

Conv::Conv(const string & str) :
	str_(str)
{}

/*===========================================================================*/
Conv::operator bool()
	{ return convert<bool>(str_); }

Conv::operator signed char()
	{ return convert<char>(str_); }

Conv::operator unsigned char()
	{ return convert<unsigned char>(str_); }

Conv::operator short()
	{ return convert<short>(str_); }

Conv::operator unsigned short()
	{ return convert<unsigned short>(str_); }

Conv::operator int()
	{ return convert<int>(str_); }

Conv::operator unsigned()
	{ return convert<unsigned>(str_); }

Conv::operator long()
	{ return convert<long>(str_); }

Conv::operator unsigned long()
	{ return convert<unsigned long>(str_); }

Conv::operator float()
	{ return convert<float>(str_); }

Conv::operator double()
	{ return convert<double>(str_); }

/*===========================================================================*/
