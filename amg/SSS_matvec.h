#ifndef _SSS_MATVEC_H_
#define _SSS_MATVEC_H_
#include "SSS_main.h"
#include "SSS_utils.h"

#if defined (__cplusplus)
extern "C"
{
#endif    

//分配vec空间
SSS_VEC SSS_vec_create(int m);

//给vec赋值
void SSS_vec_set_value(SSS_VEC *x, double Ax);

//释放mat内存空间
void SSS_mat_destroy(SSS_MAT *A);

//释放vec内存空间
void SSS_vec_destroy(SSS_VEC * u);

//calloc
void * SSS_calloc(size_t size, int type);

//创建mg网格结构
SSS_AMG SSS_amg_data_create(SSS_AMG_PARS *pars);

SSS_IVEC SSS_ivec_create(int m);

//Create CSR sparse matrix data memory space
SSS_MAT SSS_mat_struct_create(int m, int n, int nnz);

void SSS_vec_cp(const SSS_VEC *x, SSS_VEC *y);

//Copy an array to the other y=x
void SSS_iarray_cp(const int n, int *x, int *y);

//Copy an array to the other y=x
void SSS_blas_array_cp(int n, const double *x, double *y);

//copy a SSS_MAT to a new one des=src
void SSS_mat_cp(SSS_MAT *src, SSS_MAT *des);

//Get first n diagonal entries of a CSR matrix A
SSS_VEC SSS_mat_get_diag(SSS_MAT *A, int n);


//Free vector data space of int type
void SSS_ivec_destroy(SSS_IVEC * u);

//Free SSS_AMG data memeory space
void SSS_amg_data_destroy(SSS_AMG *mg);

SSS_IMAT SSS_imat_trans(SSS_IMAT *A);

void SSS_iarray_set(const int n, int *x, const int Ax);


//Free CSR sparse matrix data memory space
void SSS_imat_destroy(SSS_IMAT *A);


// Find transpose of SSS_MAT matrix A
SSS_MAT SSS_mat_trans(SSS_MAT *A);

SSS_MAT SSS_blas_mat_rap(const SSS_MAT *R, const SSS_MAT *A, const SSS_MAT *P);

void * SSS_realloc(void *oldmem, size_t tsize);

#if defined (__cplusplus)
}
#endif

#endif