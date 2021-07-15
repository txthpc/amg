#ifndef _SSS_CYCLE_H_
#define _SSS_CYCLE_H_

#include "../SSS_main.h"
#include "../SSS_matvec.h"
#include "../SSS_utils.h"
#include "SSS_smooth.h"

#if defined (__cplusplus)
extern "C"
{
#endif    
static int SSS_solver_cg(SSS_KRYLOV *ks);

static int SSS_solver_gmres(SSS_KRYLOV *ks);

void SSS_amg_coarest_solve(SSS_MAT *A, SSS_VEC *b, SSS_VEC *x, const double ctol);

void SSS_amg_cycle(SSS_AMG *mg);
#if defined (__cplusplus)
}
#endif

#endif