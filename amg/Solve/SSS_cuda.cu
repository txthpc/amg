//#include "SSS_cuda.h"
//#include <cuda.h>
 #include "SSS_cuda.h"

 const int threadsPerBlock = 64;

 

 
double dot_host(double *x,double *y, int n)
{
    int i;
    double t=0;

    for (i=0;i<n;i++)
    {
        t +=x[i] * y[i];
    }

    return t;
}

__global__ void dot_kernel(int N,double *a,double *b,double *c)
{
   __shared__ double cache[threadsPerBlock];
    int tid=threadIdx.x+blockIdx.x*blockDim.x;

    
    int cacheIndex=threadIdx.x;
 
    double temp=0;
    while(tid<N)
    {
        temp += a[tid]*b[tid];
        tid += blockDim.x*gridDim.x;
    }
 
    cache[cacheIndex]=temp;
 
    __syncthreads();
 
    //对于归约运算来说，以下代码要求threadPerBLock必须`为2的指数
    int i=blockDim.x/2;
    while(i != 0)
    {
        if(cacheIndex<i)
            cache[cacheIndex] += cache[cacheIndex+i];
        __syncthreads();
        i /=2;
    }
 
    if(cacheIndex==0)
    {
        c[blockIdx.x]=cache[0];
    }
   // printf("c[%d] = %lf\n",blockIdx.x,c[blockIdx.x]);
}

double dot_cuda(int N, double *hx,double *hy, double *dx, double *dy ,double *dz, double *recive)
{
   double result = 0;

   cudaMemcpy(dx, hx, N * sizeof(double), cudaMemcpyHostToDevice);
   cudaMemcpy(dy, hy, N * sizeof(double), cudaMemcpyHostToDevice);

   dot_kernel<<<64,64>>>(N,dx,dy,dz);
    
   cudaMemcpy(recive,dz,N*sizeof(double),cudaMemcpyDeviceToHost);

   for(int i=0;i<N;i++)
   {
       result +=recive[i];
   }
   return result;
}

__global__ void spmv_kernel(const int m,int *row_ptr,int *col_idx,double *A_val,double *x_val,double *y_val)
{

    int tid = blockDim.x * blockIdx.x +threadIdx.x;

    if (tid < m)
    {
        double temp = 0;
        int begin_row = row_ptr[tid];
        int end_row = row_ptr[tid+1];

        for(int k = begin_row; k < end_row; k++) 
        {
            temp+= A_val[k] *x_val[col_idx[k]];
        }
        y_val[tid]+=temp;

    }

}


__global__ void alpha_spmv_kernel(const int alpha ,const int m,int *row_ptr,int *col_idx,double *A_val,double *x_val,double *y_val)
{

    int tid = blockDim.x * blockIdx.x +threadIdx.x;

    if (tid < m)
    {
        double temp = 0;
        int begin_row = row_ptr[tid];
        int end_row = row_ptr[tid+1];

        for(int k = begin_row; k < end_row; k++) 
        {
            temp+= A_val[k] *x_val[col_idx[k]];
        }
        y_val[tid]+=temp * alpha;

    }

}

void spmv_cuda(SSS_MAT *A, SSS_VEC *x, SSS_VEC *y, int *d_row_ptr,int *d_col_idx,double *d_A_val,double *d_x_val,double *d_y_val)
{
    //cuda spmv

    cudaMemcpy(d_row_ptr, A->row_ptr, (A->num_rows+1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_col_idx, A->col_idx, A->num_nnzs * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_A_val, A->val, A->num_nnzs * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x_val, x->d, x->n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y_val, y->d, y->n * sizeof(double), cudaMemcpyHostToDevice);
    
    double time1=get_time();
    spmv_kernel<<<64,64>>>(y->n,d_row_ptr,d_col_idx,d_A_val,d_x_val,d_y_val);

    cudaDeviceSynchronize();

    double time2=get_time();

    
    //printf("cuda_time = %lf\n",time3);
    cudaMemcpy(y->d,d_y_val, y->n * sizeof(double), cudaMemcpyDeviceToHost);
}

void alpha_spmv_cuda(const int alpha,SSS_MAT *A, SSS_VEC *x, SSS_VEC *y, int *d_row_ptr,int *d_col_idx,double *d_A_val,double *d_x_val,double *d_y_val)
{
    //cuda spmv
    //int num = 0 ;
    cudaMemcpy(d_row_ptr, A->row_ptr, (A->num_rows+1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_col_idx, A->col_idx, A->num_nnzs * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_A_val, A->val, A->num_nnzs * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x_val, x->d, x->n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y_val, y->d, y->n * sizeof(double), cudaMemcpyHostToDevice);

    alpha_spmv_kernel<<<64,64>>>(alpha,y->n,d_row_ptr,d_col_idx,d_A_val,d_x_val,d_y_val);
    //num+=1;
    cudaDeviceSynchronize();
    
    cudaMemcpy(y->d,d_y_val, y->n * sizeof(double), cudaMemcpyDeviceToHost);
    //cudaDeviceSynchronize();
    //printf("alpha_spmv\n");
  //  cudaFree(d_row_ptr);
    //cudaFree(d_col_idx);
    //cudaFree(d_A_val);
    //cudaFree(d_x_val);
    //cudaFree(d_y_val);

}