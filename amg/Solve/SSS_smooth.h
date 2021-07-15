#ifndef _SSS_SMOOTH_H_
#define _SSS_SMOOTH_H_
#include "../SSS_main.h"
#include "../SSS_matvec.h"
#include "../SSS_utils.h"


#if defined (__cplusplus)
extern "C"
{
#endif    
static void SSS_amg_smoother_gs_cf(SSS_VEC * u, SSS_MAT * A, SSS_VEC * b, int L, int * mark, const int order);

static void SSS_amg_smoother_gs(SSS_VEC * u, const int i_1, const int i_n, const int s, SSS_MAT *A, SSS_VEC *b, int L);



void SSS_amg_smoother_pre(SSS_SMTR *s);

void SSS_amg_smoother_post(SSS_SMTR *s);
#if defined (__cplusplus)
}
#endif
#endif