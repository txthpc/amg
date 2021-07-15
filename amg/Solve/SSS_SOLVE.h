#ifndef _SSS_SOLVE_H_
#define _SSS_SOLVE_H_

#include "../SSS_main.h"
#include "../SSS_matvec.h"
#include "../SSS_utils.h"
#include "SSS_cycle.h"
 
SSS_RTN SSS_amg_solve(SSS_AMG *mg, SSS_VEC *x, SSS_VEC *b);


#endif