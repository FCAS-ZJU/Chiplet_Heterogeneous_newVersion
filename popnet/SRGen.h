#ifndef S_GEN_H_
#define S_GEN_H_

/*###########################################################################*/

#include <iosfwd>
#include <stdlib.h>
#include <cassert>

/*===========================================================================*/
// Random number generator.  True Gaussian pseudo-random variables.

class SRGen
{
	private:
		long seed_;
		static SRGen * rg_;
		double sflat01();
		double gauss01();
	public:
		static const SRGen & rg() {return *rg_;}
		static SRGen & wrg() {return *rg_;}
		double flat_d(double low, double high);
		long flat_l(long low, long high);
		unsigned long flat_ul(unsigned long low, unsigned long high);
		unsigned long long flat_ull(unsigned long long low, 
				unsigned long long high);
		double gauss_mean_d(double mean, double variance);
		long gauss_mean_l(long mean, double variance);
		unsigned long gauss_mean_ul(unsigned long mean, double variance);
		unsigned long long gauss_mean_ull( unsigned long long mean, 
				double variance);
		void set_seed(long a);
		SRGen(long a);
		SRGen();
		~SRGen() {rg_ = 0;}

};

/*###########################################################################*/
#endif
