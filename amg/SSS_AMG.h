#ifndef _SSS_AMG_H_
#define _SSS_AMG_H_

#include "SSS_main.h"

//计算向量b的L2范数
double SSS_blas_vec_norm2(const SSS_VEC *x);

//打印出迭代解算器的迭代信息
void SSS_print_itinfo( const int stop_type, const int iter, const double relres,const double absres, const double factor);

SSS_RTN SSS_solver_amg(SSS_MAT *A, SSS_VEC *x, SSS_VEC *b, SSS_AMG_PARS *pars);


#endif
