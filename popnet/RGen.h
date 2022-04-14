#ifndef NETWORK_GEN_H_
#define NETWORK_GEN_H_

/*###########################################################################*/

#include <iosfwd>
#include <stdlib.h>
#include <cassert>

/*===========================================================================*/
// Random number generator.  True Gaussian pseudo-random variables.

class RGen
{
	private:
		long seed_;
		static RGen * rg_;
		double sflat01();
	public:
		static const RGen & rg() {return *rg_;}
		static RGen & wrg() {return *rg_;}
		double flat_d(double low, double high);
		long flat_l(long low, long high);
		RGen(long a);
		~RGen() {rg_ = 0;}

};

/*###########################################################################*/
#endif
