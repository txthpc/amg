#include "SSS_inter.h"
#include <cuda.h>

#define gridsize 2048
#define blocksize 64
 
double get_time(void){
    struct timeval tv;
    double t;

    gettimeofday(&tv,NULL);
    t = tv.tv_sec * 1000 + tv.tv_usec /1000;
    return t;
}

void SSS_amg_interp_trunc(SSS_MAT *P, SSS_AMG_PARS *pars)
{
    const int row = P->num_rows;
    const int nnzold = P->num_nnzs;
    const double eps_tr = pars->trunc_threshold;

    // local variables
    int num_nonzero = 0;        // number of non zeros after truncation
    double Min_neg, max_pos;     // min negative and max positive entries
    double Fac_neg, Fac_pos;     // factors for negative and positive entries
    double Sum_neg, TSum_neg;    // sum and truncated sum of negative entries
    double Sum_pos, TSum_pos;    // sum and truncated sum of positive entries

    int index1 = 0, index2 = 0, begin, end;
    int i, j;

    for (i = 0; i < row; ++i) {
        begin = P->row_ptr[i];
        end = P->row_ptr[i + 1];

        P->row_ptr[i] = num_nonzero;
        Min_neg = max_pos = 0;
        Sum_neg = Sum_pos = 0;
        TSum_neg = TSum_pos = 0; 

        // 1. Summations of positive and negative entries
        for (j = begin; j < end; ++j) {
            if (P->val[j] > 0) {
                Sum_pos += P->val[j];
                max_pos = SSS_max(max_pos, P->val[j]);
            }
            else if (P->val[j] < 0) {
                Sum_neg += P->val[j];
                Min_neg = SSS_MIN(Min_neg, P->val[j]);
            }
        }

        max_pos *= eps_tr;
        Min_neg *= eps_tr;

        // 2. Set JA of truncated P
        for (j = begin; j < end; ++j) {
            if (P->val[j] >= max_pos) {
                num_nonzero++;
                P->col_idx[index1++] = P->col_idx[j];
                TSum_pos += P->val[j];
            }
            else if (P->val[j] <= Min_neg) {
                num_nonzero++;
                P->col_idx[index1++] = P->col_idx[j];
                TSum_neg += P->val[j];
            }
        }

        // 3. Compute factors and set values of truncated P
        if (TSum_pos > SMALLFLOAT) {
            Fac_pos = Sum_pos / TSum_pos;       // factor for positive entries
        }
        else {
            Fac_pos = 1.0;
        }

        if (TSum_neg < -SMALLFLOAT) {
            Fac_neg = Sum_neg / TSum_neg;       // factor for negative entries
        }
        else {
            Fac_neg = 1.0;
        }

        for (j = begin; j < end; ++j) {
            if (P->val[j] >= max_pos)
                P->val[index2++] = P->val[j] * Fac_pos;
            else if (P->val[j] <= Min_neg)
                P->val[index2++] = P->val[j] * Fac_neg;
        }
    }

    // resize the truncated prolongation P
    P->num_nnzs = P->row_ptr[row] = num_nonzero;
    P->col_idx = (int *) SSS_realloc(P->col_idx, num_nonzero * sizeof(int));
    P->val = (double *) SSS_realloc(P->val, num_nonzero * sizeof(double));
    
    //Truncate prolongation
    //    printf("Truncate prolongation, nnz before: %10d, after: %10d\n",
         //      nnzold, num_nonzero);
    
}

__global__ void DIR_Step_1(int row,int *d_A_row_ptr,int *d_A_col_idx,double *d_A_val,int *d_vec,int *d_P_row_ptr,int *d_P_col_idx,double *d_P_val)
{
    int tid = blockDim.x * blockIdx.x +threadIdx.x;
    int begin_row,end_row;

    //--------------------参数---------------------
    double alpha, beta, aii = 0;
    // a_minus and a_plus for Neighbors and Prolongation support
    double amN, amP, apN, apP;
    int IS_STRONG;              // is the variable strong coupled to i?

    int num_pcouple;            // number of positive strong couplings
    
    int j, k, l, index  = 0, idiag;


    //-------------------cuda----------------------
    if(tid<row)
    {
        begin_row = d_A_row_ptr[tid];
        end_row = d_A_row_ptr[tid + 1];


        // find diagonal entry first!!!
        for (idiag = begin_row; idiag < end_row; idiag++)
        {
            if (d_A_col_idx[idiag] == tid)
            {
                aii = d_A_val[idiag];
                break;
            }
        }

        if (d_vec[tid] == FGPT) 
        {  
        // fine grid nodes
            amN = amP = apN = apP = 0.0;
            num_pcouple = 0;

            for (j = begin_row; j < end_row; ++j) 
            {
                if (j == idiag) continue;   // skip diagonal

        // check a point strong-coupled to i or not
                IS_STRONG = FALSE;

                for (k = d_P_row_ptr[tid]; k < d_P_row_ptr[tid + 1]; ++k) 
                {
                    if (d_P_col_idx[k] == d_A_col_idx[j]) 
                    {
                        IS_STRONG = TRUE;
                        break;
                    }
                }

                if (d_A_val[j] > 0) 
                {
                    apN += d_A_val[j];   // sum up positive entries
                    if (IS_STRONG)
                    {
                        apP += d_A_val[j];
                        num_pcouple++;
                    }
                }
                else 
                {
                    amN += d_A_val[j];   // sum up negative entries
                    if (IS_STRONG) {
                        amP += d_A_val[j];
                    }
                }
            }

            // set weight factors
            alpha = amN / amP;
            if (num_pcouple > 0) {
                beta = apN / apP;
            }
            else {
                beta = 0.0;
                aii += apN;
            }

            // keep aii inside the loop to avoid floating pt error
            for (j = d_P_row_ptr[tid]; j < d_P_row_ptr[tid + 1]; ++j) 
            {
                k = d_P_col_idx[j];

                for (l = d_A_row_ptr[tid]; l < d_A_row_ptr[tid + 1]; l++) 
                {
                    if (d_A_col_idx[l] == k) break;
                }
                if (d_A_val[l] > 0) 
                {
                    d_P_val[j] = -beta * d_A_val[l] / aii;
                }
                else 
                {
                    d_P_val[j] = -alpha * d_A_val[l] / aii;
                }
            }
        }                       // end if vec
        else if (d_vec[tid] == CGPT) {      // coarse grid nodes
            d_P_val[d_P_row_ptr[tid]] = 1.0;
        }
    }
}


__global__ void DIR_Step_2(int row,int *d_vec,int *d_cindex,int index)
{
    int tid = blockDim.x * blockIdx.x +threadIdx.x;

    index =tid;

    if(tid<row)
    {
        if (d_vec[tid] == CGPT)
            d_cindex[tid] = index++;
    }
}


__global__ void DIR_Step_3(int p_nnz,int *d_P_col_idx,int *d_cindex)
{
    int tid = blockDim.x * blockIdx.x +threadIdx.x;

    int j=0;
    if (tid <p_nnz)
    {
        j=d_P_col_idx[tid];
        d_P_col_idx[tid] =  d_cindex[j];
    }
}

void interp_DIR_cuda(SSS_MAT *A, SSS_IVEC *vertices, SSS_MAT *P, SSS_AMG_PARS *pars)
{
    int row = A->num_rows;
    int  index  = 0;

    int *vec = vertices->d;
    // indices of C-nodes
    int *cindex = (int *) SSS_calloc(row, sizeof(int));

    int    *d_cindex =    NULL;
    int    *d_vec =       NULL;

    int    *d_A_row_ptr = NULL;
    int    *d_A_col_idx = NULL;
    double *d_A_val     = NULL;

    int    *d_P_row_ptr = NULL;
    int    *d_P_col_idx = NULL;
    double *d_P_val     = NULL;

    struct timeval ww, rr;
    gettimeofday(&ww,NULL);

    cudaFree(0);
    gettimeofday(&rr,NULL);
    double ee = (rr.tv_sec - ww.tv_sec) * 1000.0 + (rr.tv_usec - ww.tv_usec) / 1000.0;

    //printf("-------------cuda_warmup_time = %f ms -------------------\n",ee);

    //---------------- cuda Malloc ----------

    struct timeval cudamalloc_1, cudamalloc_2;
    gettimeofday(&cudamalloc_1,NULL);
    //vec cindex
    cudaMalloc((void **)&d_cindex,row * sizeof(int));
    cudaMalloc((void **)&d_vec,row * sizeof(int));
    //A
    cudaMalloc((void **)&d_A_row_ptr,(A->num_rows+1) * sizeof(int));
    cudaMalloc((void **)&d_A_col_idx,A->num_nnzs * sizeof(int));
    cudaMalloc((void **)&d_A_val,A->num_nnzs * sizeof(double));
    //P
    cudaMalloc( (void **)&d_P_row_ptr,(P->num_rows+1) * sizeof(int));
    cudaMalloc( (void **)&d_P_col_idx,P->num_nnzs * sizeof(int));
    cudaMalloc( (void **)&d_P_val,P->num_nnzs * sizeof(double));

    gettimeofday(&cudamalloc_2,NULL);

    double cudamalloc_3 = (cudamalloc_2.tv_sec - cudamalloc_1.tv_sec) * 1000.0 + (cudamalloc_2.tv_usec - cudamalloc_1.tv_usec) / 1000.0;

    //printf("-------------cuda_malloc_time = %f ms -------------------\n",cudamalloc_3);

    if (d_cindex == NULL || d_vec ==NULL || d_A_row_ptr == NULL|| d_A_col_idx == NULL|| d_A_val == NULL|| d_P_row_ptr == NULL|| d_P_col_idx == NULL || d_P_val == NULL) 
    {
        printf("could't allocate GPU mem \n");
    }

    //-----------cuda Memcpy host_to_device----------

    struct timeval hosttodevice_1, hosttodevice_2;
    gettimeofday(&hosttodevice_1,NULL);
    //vec cindex
    cudaMemcpy(d_cindex, cindex,  row * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_vec, vec, row * sizeof(int), cudaMemcpyHostToDevice);
    //A
    cudaMemcpy(d_A_row_ptr, A->row_ptr, (A->num_rows+1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_A_col_idx, A->col_idx, A->num_nnzs * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_A_val, A->val, A->num_nnzs * sizeof(double), cudaMemcpyHostToDevice);
    //P
    cudaMemcpy(d_P_row_ptr, P->row_ptr, (P->num_rows+1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_P_col_idx, P->col_idx, P->num_nnzs * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_P_val, P->val, P->num_nnzs * sizeof(double), cudaMemcpyHostToDevice);

    gettimeofday(&hosttodevice_2,NULL);
    
    double hosttodevice_3 = (hosttodevice_2.tv_sec - hosttodevice_1.tv_sec) * 1000.0 + (hosttodevice_2.tv_usec - hosttodevice_1.tv_usec) / 1000.0;

    //printf("-------------cuda_host_to_device_time = %f ms -------------------\n",hosttodevice_3);

    //--------------------cuda step1-----------------------

    struct timeval cuda_step1_1, cuda_step1_2;
    gettimeofday(&cuda_step1_1,NULL);

    DIR_Step_1<<<gridsize,blocksize>>>(row, d_A_row_ptr, d_A_col_idx, d_A_val, d_vec, d_P_row_ptr, d_P_col_idx, d_P_val);

    cudaDeviceSynchronize();
    gettimeofday(&cuda_step1_2,NULL);

    double cuda_step1_3 = (cuda_step1_2.tv_sec - cuda_step1_1.tv_sec) * 1000.0 + (cuda_step1_2.tv_usec - cuda_step1_1.tv_usec) / 1000.0;
    
    printf("-------------cuda_step_1_time = %f ms -------------------\n",cuda_step1_3);

    //-----------cuda Memcpy device_to_host----------
    struct timeval devicetohost1, devicetohost2;
    gettimeofday(&devicetohost1,NULL);
    //vec cindex
    cudaMemcpy(vec,d_vec,row * sizeof(int),cudaMemcpyDeviceToHost);
    cudaMemcpy(cindex, d_cindex,  row * sizeof(int), cudaMemcpyDeviceToDevice);
    //A
    cudaMemcpy(A->row_ptr,d_A_row_ptr,(A->num_rows+1)*sizeof(int),cudaMemcpyDeviceToHost);
    cudaMemcpy(A->col_idx,d_A_col_idx,A->num_nnzs*sizeof(int),cudaMemcpyDeviceToHost);
    cudaMemcpy(A->val,d_A_val,A->num_nnzs*sizeof(double),cudaMemcpyDeviceToHost);
    //P
    cudaMemcpy(P->row_ptr,d_P_row_ptr,(P->num_rows+1)*sizeof(int),cudaMemcpyDeviceToHost);
    cudaMemcpy(P->col_idx,d_P_col_idx,P->num_nnzs*sizeof(int),cudaMemcpyDeviceToHost);
    cudaMemcpy(P->val,d_P_val,P->num_nnzs*sizeof(double),cudaMemcpyDeviceToHost);

    gettimeofday(&devicetohost2,NULL);

    double devicetohost3 = (devicetohost2.tv_sec - devicetohost1.tv_sec) * 1000.0 + (devicetohost2.tv_usec - devicetohost1.tv_usec) / 1000.0;
    
    //printf("-------------cuda_device_to_host_time = %f ms -------------------\n",devicetohost3);


    //-------------------------------cudaFree------------------------
    struct timeval cudafree_1, cudafree_2;
    gettimeofday(&cudafree_1,NULL);
    cudaFree(d_cindex);
    cudaFree(d_vec);

    cudaFree(d_A_row_ptr);
    cudaFree(d_A_col_idx);
    cudaFree(d_A_val);
    
    cudaFree(d_P_row_ptr);
    cudaFree(d_P_col_idx);
    cudaFree(d_P_val);

    gettimeofday(&cudafree_2,NULL);

    double cudafree_3 = (cudafree_2.tv_sec - cudafree_1.tv_sec) * 1000.0 + (cudafree_2.tv_usec - cudafree_1.tv_usec) / 1000.0;
    
    //printf("-------------cuda_free_time = %f ms -------------------\n",cudafree_3);

    // Step 2. Generate coarse level indices and set values of P.Aj
    int i,j;
    for (index = i = 0; i < row; ++i) 
    {
        if (vec[i] == CGPT)
        cindex[i] = index++;
    }

    P->num_cols = index;

    for (i = 0; i < P->num_nnzs; ++i) 
    {
        j = P->col_idx[i];
        P->col_idx[i] = cindex[j];
    }
    
    // clean up
    SSS_free(cindex);

    // Step 3. Truncate the prolongation operator to reduce cost

    SSS_amg_interp_trunc(P, pars);

}



void interp_DIR(SSS_MAT * A, SSS_IVEC * vertices, SSS_MAT * P, SSS_AMG_PARS * pars)
{ 
    int row = A->num_rows;
    int *vec = vertices->d;
    
    // local variables
    int IS_STRONG;              // is the variable strong coupled to i?
    int num_pcouple;            // number of positive strong couplings
    int begin_row, end_row;
    int i, j, k, l, index  = 0, idiag;

    // a_minus and a_plus for Neighbors and Prolongation support
    double amN, amP, apN, apP;
    double alpha, beta, aii = 0;

    // indices of C-nodes
    int *cindex = (int *) SSS_calloc(row, sizeof(int));

    struct timeval cpu_step_1, cpu_step_2;
    gettimeofday(&cpu_step_1,NULL);
    // Step 1. Fill in values for interpolation operator P
    for (i = 0; i < row; ++i) 
    {

        begin_row = A->row_ptr[i];
        end_row = A->row_ptr[i + 1];

        // find diagonal entry first!!!
        for (idiag = begin_row; idiag < end_row; idiag++)
        {
            if (A->col_idx[idiag] == i)
            {
                aii = A->val[idiag];
                break;
            }
        }

        if (vec[i] == FGPT) 
        {   // fine grid nodes
            amN = amP = apN = apP = 0.0;
            num_pcouple = 0;

            for (j = begin_row; j < end_row; ++j) 
            {
                if (j == idiag) continue;   // skip diagonal

                // check a point strong-coupled to i or not
                IS_STRONG = FALSE;

                for (k = P->row_ptr[i]; k < P->row_ptr[i + 1]; ++k) 
                {
                    if (P->col_idx[k] == A->col_idx[j]) 
                    {
                        IS_STRONG = TRUE;
                        break;
                    }
                }

                if (A->val[j] > 0) 
                {
                    apN += A->val[j];   // sum up positive entries
                    if (IS_STRONG)
                    {
                        apP += A->val[j];
                        num_pcouple++;
                    }
                }
                else 
                {
                    amN += A->val[j];   // sum up negative entries
                    if (IS_STRONG) {
                        amP += A->val[j];
                    }
                }
            }

            // set weight factors
            alpha = amN / amP;
            if (num_pcouple > 0) {
                beta = apN / apP;
            }
            else {
                beta = 0.0;
                aii += apN;
            }

            // keep aii inside the loop to avoid floating pt error
            for (j = P->row_ptr[i]; j < P->row_ptr[i + 1]; ++j) 
            {
                k = P->col_idx[j];

                for (l = A->row_ptr[i]; l < A->row_ptr[i + 1]; l++) 
                {
                    if (A->col_idx[l] == k) break;
                }
                if (A->val[l] > 0) 
                {
                    P->val[j] = -beta * A->val[l] / aii;
                }
                else 
                {
                    P->val[j] = -alpha * A->val[l] / aii;
                }
            }
        }                       // end if vec
        else if (vec[i] == CGPT) {      // coarse grid nodes
            P->val[P->row_ptr[i]] = 1.0;
        }
    }
    gettimeofday(&cpu_step_2,NULL);

    double cpu_step_3 = (cpu_step_2.tv_sec - cpu_step_1.tv_sec) * 1000.0 + (cpu_step_2.tv_usec - cpu_step_1.tv_usec) / 1000.0;
    printf("-------------cpu_step1_time = %f ms -------------------\n",cpu_step_3);



    // Step 2. Generate coarse level indices and set values of P.Aj
    double time4=get_time();
    for (index = i = 0; i < row; ++i) {
        if (vec[i] == CGPT)
            cindex[i] = index++;
    }

    P->num_cols = index;

    for (i = 0; i < P->num_nnzs; ++i) {
        j = P->col_idx[i];
        P->col_idx[i] = cindex[j];
    }
    double time5=get_time();
    double time6=time5-time4;

  //  printf("step2 time = %lf\n",time6);


    // clean up
    SSS_free(cindex);

    // Step 3. Truncate the prolongation operator to reduce cost
    double time7=get_time();

    SSS_amg_interp_trunc(P, pars);
    double time8=get_time();
    double time9=time8-time7;

//    printf("step3 time = %lf\n",time9);

}


static void interp_STD(SSS_MAT * A, SSS_IVEC * vertices, SSS_MAT * P, SSS_IMAT * S, SSS_AMG_PARS * pars)
{
    //8 faster
    omp_set_num_threads(8);

    const int row = A->num_rows;
    int *vec = vertices->d;

    // local variables
    int i, j, k, l, m, index;
    double alpha, factor, alN, alP;
    double akk, akl, aik, aki;

    // indices for coarse neighbor node for every node
    int *cindex = (int *) SSS_calloc(row, sizeof(int));

    // indices from column number to index in nonzeros in i-th row
    int *rindi = (int *) SSS_calloc(2 * row, sizeof(int));

    // indices from column number to index in nonzeros in k-th row
    int *rindk = (int *) SSS_calloc(2 * row, sizeof(int));

    // sums of strongly connected C neighbors
    double *csum = (double *) SSS_calloc(row, sizeof(double));

    // sums of all neighbors except ISPT
    double *psum = (double *) SSS_calloc(row, sizeof(double));

    // sums of all neighbors
    double *nsum = (double *) SSS_calloc(row, sizeof(double));

    // diagonal entries
    double *diag = (double *) SSS_calloc(row, sizeof(double));

    // coefficients hat a_ij for relevant CGPT of the i-th node
    double *Ahat = (double *) SSS_calloc(row, sizeof(double));

    // Step 0. Prepare diagonal, Cs-sum, and N-sum
    SSS_iarray_set(row, cindex, -1);
    SSS_blas_array_set(row, csum, 0.0);
    SSS_blas_array_set(row, nsum, 0.0);
    for (i = 0; i < row; i++) {
        // set flags for strong-connected C nodes
 //num = 8 
 //#pragma omp parallel for
        for (j = S->row_ptr[i]; j < S->row_ptr[i + 1]; j++) {
            k = S->col_idx[j];
            if (vec[k] == CGPT) cindex[k] = i;
        }
    //#pragma omp parallel for
        for (j = A->row_ptr[i]; j < A->row_ptr[i + 1]; j++) {
            k = A->col_idx[j];

            if (cindex[k] == i) csum[i] += A->val[j];   // strong C-couplings

            if (k == i)
                diag[i] = A->val[j];
            else {
                nsum[i] += A->val[j];
                if (vec[k] != ISPT) {
                    psum[i] += A->val[j];
                }
            }
        }
    }

    // Step 1. Fill in values for interpolation operator P
//#pragma omp parallel for
    for (i = 0; i < row; i++) {
        if (vec[i] == FGPT) {
            alN = psum[i];
            alP = csum[i];

// form the reverse indices for i-th row
//#pragma omp parallel for
            for (j = A->row_ptr[i]; j < A->row_ptr[i + 1]; j++) rindi[A->col_idx[j]] = j;
//#pragma omp parallel for
            // clean up Ahat for relevant nodes only
//#pragma omp parallel for
            for (j = P->row_ptr[i]; j < P->row_ptr[i + 1]; j++) Ahat[P->col_idx[j]] = 0.0;

            // set values of Ahat
            Ahat[i] = diag[i];
//#pragma omp parallel for
            for (j = S->row_ptr[i]; j < S->row_ptr[i + 1]; j++) {
                k = S->col_idx[j];
                aik = A->val[rindi[k]];

                if (vec[k] == CGPT) {
                    Ahat[k] += aik;
                }
                else if (vec[k] == FGPT) {
                    akk = diag[k];


                    // form the reverse indices for k-th row
//#pragma omp parallel for
                    for (m = A->row_ptr[k]; m < A->row_ptr[k + 1]; m++) rindk[A->col_idx[m]] = m;

                    factor = aik / akk;

                    // visit the strong-connected C neighbors of k, compute
                    // Ahat in the i-th row, set aki if found
                    aki = 0.0;
//#pragma omp parallel for
                    for (m = A->row_ptr[k]; m < A->row_ptr[k + 1]; m++) {
                        if (A->col_idx[m] == i) {
                            aki = A->val[m];
                            Ahat[i] -= factor * aki;
                        }
                    }
//#pragma omp parallel for
                    for (m = S->row_ptr[k]; m < S->row_ptr[k + 1]; m++) {
                        l = S->col_idx[m];
                        akl = A->val[rindk[l]];
                        if (vec[l] == CGPT)
                            Ahat[l] -= factor * akl;
                    }           // end for m

                    // compute Cs-sum and N-sum for Ahat
                    alN -= factor * (nsum[k] - aki + akk);
                    alP -= factor * csum[k];

                }               // end if vec[k]
            }                   // end for j

            // How about positive entries
            if (P->row_ptr[i + 1] > P->row_ptr[i]) alpha = alN / alP;
//#pragma omp parallel for
            for (j = P->row_ptr[i]; j < P->row_ptr[i + 1]; j++) {
                k = P->col_idx[j];
                P->val[j] = -alpha * Ahat[k] / Ahat[i];
            }
        }
        else if (vec[i] == CGPT) {
            P->val[P->row_ptr[i]] = 1.0;
        }
    }                           // end for i

    // Step 2. Generate coarse level indices and set values of P.col_idx
//#pragma omp parallel for
    for (index = i = 0; i < row; ++i) {
        if (vec[i] == CGPT) cindex[i] = index++;
    }

    P->num_cols = index;
//#pragma omp parallel for
    for (i = 0; i < P->row_ptr[P->num_rows]; ++i) {
        j = P->col_idx[i];
        P->col_idx[i] = cindex[j];
    }

    // clean up
    SSS_free(cindex);
    SSS_free(rindi);
    SSS_free(rindk);
    SSS_free(nsum);

    SSS_free(psum);
    SSS_free(csum);
    SSS_free(diag);
    SSS_free(Ahat);

    // Step 3. Truncate the prolongation operator to reduce cost
    SSS_amg_interp_trunc(P, pars);
}

void SSS_amg_interp(SSS_MAT *A, SSS_IVEC *vertices, SSS_MAT *P, SSS_IMAT *S, SSS_AMG_PARS *pars)
{
    interp_type interp_type = pars->interp_type;

    switch (interp_type) {
        case intERP_DIR:       // Direct interpolation
            //interp_DIR(A, vertices, P, pars);
            interp_DIR_cuda(A, vertices, P, pars);

             break;

        case intERP_STD:       // Standard interpolation
            interp_STD(A, vertices, P, S, pars);
            break;

        default:
            SSS_exit_on_errcode(ERROR_AMG_interp_type, __FUNCTION__);
    }
}
