#ifndef _SSS_INTER_H_
#define _SSS_INTER_H_

#include "../SSS_main.h"
#include "../SSS_utils.h"
#include "../SSS_matvec.h"
#include <cuda.h>
#include <omp.h>
#include "SSS_SETUP.h" 

#if defined (__cplusplus)
extern "C"
{
#endif    

void SSS_amg_interp_trunc(SSS_MAT *P, SSS_AMG_PARS *pars);

static void interp_STD(SSS_MAT * A, SSS_IVEC * vertices, SSS_MAT * P, SSS_IMAT * S, SSS_AMG_PARS * pars);

void SSS_amg_interp(SSS_MAT *A, SSS_IVEC *vertices, SSS_MAT *P, SSS_IMAT *S, SSS_AMG_PARS *pars);

void interp_DIR(SSS_MAT * A, SSS_IVEC * vertices, SSS_MAT * P, SSS_AMG_PARS * pars);

void interp_DIR_cuda(SSS_MAT *A, SSS_IVEC *vertices, SSS_MAT *P, SSS_AMG_PARS *pars);


#if defined (__cplusplus)
}
#endif

#endif