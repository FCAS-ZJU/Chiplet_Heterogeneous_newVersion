#include "SStd.h"

#include <iostream>
#include <cstdlib>

/*###########################################################################*/
ostream scerrhex(cerr.rdbuf());
ostream scouthex(cout.rdbuf());

namespace {
// Cause hex streams to be initialized.
	int cerr_hex_init() { scerrhex << hex; return 0; }
	int cout_hex_init() { scouthex << hex; return 0; }

	int L_cout_hex_init = cout_hex_init();
	int L_cerr_hex_init = cerr_hex_init();
}

/*===========================================================================*/
void spos(const char * file, int line) {
	cerr << "File: " << file << ", Line: " << line << endl;
}

/*===========================================================================*/
void sabort(const char * file, int line, const char *message) {
	spos(file, line);
	cerr << message;
	abort();
}

/*===========================================================================*/
void sexit(const char * message) {
	cout << message;
	exit(0);
}

/*===========================================================================*/
void sassert(bool x, const char * file, int line) {
    if (! x) {
        sabort(file, line, "Assertion failed.  Aborting.\n");
    }
}
/*###########################################################################*/


