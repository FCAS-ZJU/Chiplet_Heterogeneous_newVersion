#include "RGen.h"

/*##########################################################################*/
RGen * RGen::rg_ = 0;
/*===========================================================================*/
RGen::RGen(long a) :
	seed_(a)
{
	rg_ = this;
	srandom(seed_);
}

/*===========================================================================*/
double RGen::sflat01() {
	double a = random() * 1.0 / RAND_MAX;
	return a;
}

/*===========================================================================*/
double RGen::flat_d(double low, double high) {
	assert(low < high);
	return (high - low) * sflat01() + low;
}

/*===========================================================================*/
long RGen::flat_l(long low, long high) {
	assert(low < high);
	long val = long((high - low) * sflat01() + low);
	assert(val >= low && val < high);
	return val;
}
