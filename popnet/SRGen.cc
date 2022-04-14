#include "SRGen.h"
#include "SStd.h"
#include <cmath>

/*##########################################################################*/
SRGen * SRGen::rg_ = 0;
/*===========================================================================*/
SRGen::SRGen(long a) :
	seed_(a)
{
	rg_ = this;
	srandom(seed_);
}
/*===========================================================================*/

SRGen::SRGen() :
	seed_(1)
{
	rg_ = this;
	srandom(seed_);
}
/*===========================================================================*/
double SRGen::sflat01() {
	double a = random() * 1.0 / RAND_MAX;
	return a;
}

/*===========================================================================*/
void SRGen::set_seed(long a) {
	srandom(a);
}
/*===========================================================================*/
double SRGen::flat_d(double low, double high) {
	assert(low < high);
	return (high - low) * sflat01() + low;
}

/*===========================================================================*/
long SRGen::flat_l(long low, long high) {
	assert(low < high);
	long val = long((high - low) * sflat01() + low);
	assert(val >= low && val < high);
	return val;
}

unsigned long SRGen::flat_ul(unsigned long low, unsigned long high) {
	assert(low < high);
	unsigned long val = (unsigned long) ((high-low) * sflat01() + low);
	assert(val >= low && val < high);
	return val;
}

unsigned long long SRGen::flat_ull(unsigned long long low, 
		unsigned long long high) {
	assert(low < high);
	unsigned long long val = (unsigned long long) ((high-low) * sflat01() 
			+ low);
	assert(val >= low && val < high);
	return val;
}
/*===========================================================================*/
double SRGen::gauss01() {
	//mean 0, variance 1
		static int recalc=1;
		double in_a, in_b;
		double out_a;
		static double out_b;
		double modifier;
		double compile_b;

		if(recalc) {
			//Range from (0:1], not [0:1). Had to change this to prevent
			//log(0).
			in_a = 1.0 - sflat01();
			in_b = sflat01();

			modifier = sqrt(-2.0 * log(in_a));
			compile_b = 2.0 * PI * in_b;

			out_a = modifier * cos(compile_b);
			out_b = modifier * sin(compile_b);

			recalc = 0;
			return out_b;
		}

		recalc = 1;
		return out_b;
}
/*===========================================================================*/
double SRGen::gauss_mean_d(double mean, double variance) {
	double ret = gauss01() * variance + mean;
	return ret;
}

long SRGen::gauss_mean_l(long mean, double variance) {
	return ((long)(gauss01() * variance + mean));
}

unsigned long SRGen::gauss_mean_ul(unsigned long mean, double variance) {
	return((unsigned long)(gauss01() * variance + mean));
}

unsigned long long SRGen::gauss_mean_ull(unsigned long long mean, double
		variance) {
	return ((unsigned long long)(gauss01() * variance + mean));
}
