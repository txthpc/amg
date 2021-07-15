#ifndef _SSS_UTILS_H_
#define _SSS_UTILS_H_

#include "SSS_main.h"
#include <sys/time.h>

#if defined (__cplusplus)
extern "C"
{
#endif    
double SSS_get_time(void);

void SSS_free(void *mem);

//计算向量b的L2范数
double SSS_blas_vec_norm2(const SSS_VEC *x);

//打印出迭代解算器的迭代信息
void SSS_print_itinfo(const int stop_type, const int iter, const double relres,
        const double absres, const double factor);

//Standard and aggressive coarsening schemes
void SSS_exit_on_errcode(const int status, const char *fctname);



void SSS_blas_mv_amxpy(double alpha, const SSS_MAT *A, const SSS_VEC *x, SSS_VEC *y);


void SSS_blas_mv_mxy(const SSS_MAT *A, const SSS_VEC *x, SSS_VEC *y);

double SSS_blas_array_norm2(int n, const double *x);


double SSS_blas_array_dot(int n, const double *x, const double *y);

void SSS_blas_array_axpy(int n, double a, const double *x, double *y);

double SSS_blas_array_norminf(int n, const double *x);


//Set initial value for an array to be x=Ax
void SSS_blas_array_set( int n, double *x, double Ax);

void SSS_blas_array_axpby(int n, double a, const double *x, double b, double *y);

void SSS_blas_array_ax(int n, double a, double *x);
#if defined (__cplusplus)
}
#endif
#endif