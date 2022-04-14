#ifndef NETWORK_STD_H_
#define NETWORK_STD_H_

/*###########################################################################*/
#include <iostream>
using namespace std;
extern ostream scerrhex;
extern ostream scouthex;

// Use for a sane and fast vector of bools.
typedef int big_bool;

const double PI = 3.141592658979323846;

/*===========================================================================*/
// Debugging

// Macros need to be used to accomplish this.  Inlines won't work.

// Print position in code.
void spos(const char * file, int line);
#define Spos() spos(__FILE__, __LINE__)

// Print position in code and abort after printing message.
void sabort(const char * file, int line,
	const char * message = "Program aborted.\n");

void sexit(const char * message = "Exiting.\n");

#define Rabort() sabort(__FILE__, __LINE__)

// Abort, printing position in code, if (! x).
void sassert(bool x, const char * file, int line);
#define Sassert(x) sassert((x), __FILE__, __LINE__)

inline void sconfirm(bool x, const char * message) { if (! x) sexit(message); }

/*===========================================================================*/
/*###########################################################################*/
#endif
