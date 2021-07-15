#ifndef __SSS_CUDA_H__
#define __SSS_CIDA_H__

#include <cuda.h> 

#include "../SSS_main.h"

#define get_tid() (blockDim.x * (blockIdx.x + blockIdx.y * gridDim.x) + threadIdx.x)
#define get_bid() (blockIdx.x + blockIdx.y * gridDim.x)



 
double get_time(void) ;

double dot_host(double *x,double *y, int n);

static __device__ void warpReduce(volatile double *sdata,int tid);

__global__ void dot_stg_1(const double *x,double *y, double *z ,int n);

__global__ void dot_stg_2(const double *x,double *y ,int n);

__global__ void dot_stg_3(double *x,int N);

void dot_device(double *dx,double *dy,double *dz,double *d,int N);

__global__ void dot_kernel(int N,double *a,double *b,double *c);

double dot_cuda(int N, double *hx,double *hy, double *dx, double *dy ,double *dz, double *recive);

__global__ void spmv_kernel(const int m,int *row_ptr,int *col_idx,double *A_val,double *x_val,double *y_val);

void spmv_cuda(SSS_MAT *A, SSS_VEC *x, SSS_VEC *y, int *d_row_ptr,int *d_col_idx,double *d_A_val,double *d_x_val,double *d_y_val);

__global__ void alpha_spmv_kernel(const int alpha,const int m,int *row_ptr,int *col_idx,double *A_val,double *x_val,double *y_val);

void alpha_spmv_cuda(const int alpha,SSS_MAT *A, SSS_VEC *x, SSS_VEC *y, int *d_row_ptr,int *d_col_idx,double *d_A_val,double *d_x_val,double *d_y_val);



#endif