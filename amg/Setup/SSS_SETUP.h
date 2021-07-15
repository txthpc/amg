#ifndef _SSS_SETUP_H_
#define _SSS_SETUP_H_

#include "../SSS_main.h"
#include "../SSS_matvec.h"
#include "../SSS_utils.h"

#include "SSS_coarsen.h"
#include "SSS_inter.h"

#if defined (__cplusplus)
extern "C"
{
#endif    
void SSS_amg_complexity_print(SSS_AMG *mg);

void SSS_amg_setup(SSS_AMG *mg, SSS_MAT *A, SSS_AMG_PARS *pars);





#if defined (__cplusplus)
}
#endif

#endif 