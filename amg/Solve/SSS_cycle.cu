#include "SSS_cycle.h"
#include "SSS_cuda.h"
//#include <__clang_cuda_math_forward_declares.h>
//#include <__clang_cuda_builtin_vars.h>
//#include <__clang_cuda_builtin_vars.h>
#include <cstdio>
//#include <string.h>

#define min(a,b) (a<b ? a:b)
#define sum_squares(x) (x*(x+1)*(2*x+1)/6)




int SSS_solver_cg(SSS_KRYLOV *ks)
{
    SSS_MAT *A = ks->A;
    SSS_VEC *b = ks->b;
    SSS_VEC *u = ks->u;
    double tol = ks->tol;
    int matrix = ks->matrix;
    int stop_type = ks->stop_type;

    int maxStag = max_STAG, maxRestartStep = max_RESTART;
    int m = b->n;
    int nnz =A->num_nnzs;
    double maxdiff = tol * 1e-4;       // staganation tolerance
    double sol_inf_tol = SMALLFLOAT;   // infinity norm tolerance
    int iter = 0, stag = 1, more_step = 1, restart_step = 1;
    double absres0 = BIGFLOAT, absres = BIGFLOAT;
    double relres = BIGFLOAT, normu = BIGFLOAT, normr0 = BIGFLOAT;
    double reldiff, factor, infnormu;
    double alpha, beta, temp1, temp2;
    double absres_cuda;
    double temp1_cuda,temp2_cuda;

    int iter_best = 0;              // initial best known iteration
    double absres_best = BIGFLOAT;   // initial best known residual

    // allocate temp memory (need 5*m double numbers)
    double *work = (double *) SSS_calloc(5 * m, sizeof(double));
    double *p = work, *z = work + m, *r = z + m, *t = r + m, *u_best = t + m;
    SSS_VEC vr;

    vr.n = b->n;
    vr.d = r;

   // int blockPerGrid=(m+threadsPerBlock-1)/threadsPerBlock;

   //cuda dot
    double *dx = NULL;
    double *dy = NULL;
    double *dz = NULL;
    
    cudaMalloc((void **)&dx,5 * m * sizeof(double));
    cudaMalloc((void **)&dy,5 * m * sizeof(double));
    cudaMalloc((void **)&dz,5 * m * sizeof(double));

    double *recive =NULL;
    recive =(double *)malloc(5* m *sizeof(double));

    if (dx == NULL || dy ==NULL || dz == NULL) 
    {
        printf("could't allocate GPU mem \n");
        return -1;
    }


    //cuda spmv
    int *d_row_ptr = NULL;
    int *d_col_idx = NULL;
    double *d_A_val = NULL;
    double *d_x_val = NULL;
    double *d_y_val = NULL;

    cudaMalloc((void **)&d_row_ptr,(m+1) * sizeof(int));
    cudaMalloc((void **)&d_col_idx,nnz * sizeof(int));
    cudaMalloc((void **)&d_A_val,  nnz * sizeof(double));
    cudaMalloc((void **)&d_x_val,  m * sizeof(double));
    cudaMalloc((void **)&d_y_val,  m * sizeof(double));

    if (d_row_ptr == NULL || d_col_idx ==NULL || d_A_val == NULL ||d_x_val ==NULL ||d_y_val ==NULL) 
    {
        printf("could't allocate GPU mem \n");
        return -1;
    }

    
    // r = b-A*u
    SSS_blas_array_cp(m, b->d, r);
    //SSS_blas_mv_amxpy(-1.0, A, u, &vr);
    alpha_spmv_cuda(-1.0,A,u,&vr,d_row_ptr,d_col_idx,d_A_val,d_x_val,d_y_val);

    SSS_blas_array_cp(m, r, z);
    //printf("m = %d\n",m);
    double dot_device_result = 0;
    //double dot_device_result_tmp2 = 0;
    //double dot_device_result_absres =0;

    // compute initial residuals
    switch (stop_type) {
        case STOP_REL_RES:
            absres0 = SSS_blas_array_norm2(m, r);
            normr0 = SSS_max(SMALLFLOAT, absres0);
            relres = absres0 / normr0;
            break;

        case STOP_REL_PRECRES:
            //CPU dot
            absres0 = sqrt(SSS_blas_array_dot(m, r, z));
            //GPU dot
          //  dot_device_result = sqrt(dot_cuda(m,r,z,dx,dy,dz,recive));
            if (absres0 ==dot_device_result)
            {
                printf(" dot CPU = GPU \n");
            }
            normr0 = SSS_max(SMALLFLOAT, absres0);
            relres = absres0 / normr0;
            break;

        case STOP_MOD_REL_RES:
            absres0 = SSS_blas_array_norm2(m, r);
            normu = SSS_max(SMALLFLOAT, SSS_blas_array_norm2(m, u->d));
            relres = absres0 / normu;
            break;

        default:
            printf("### ERROR: Unrecognized stopping type for %s!\n", __FUNCTION__);
            goto eofc;
    }

    // if initial residual is small, no need to iterate!
    if (relres < tol) goto eofc;

    // output iteration information if needed
    //SSS_print_itinfo( stop_type, iter, relres, absres0, 0.0);
    SSS_blas_array_cp(m, z, p);
  
    temp1 = SSS_blas_array_dot(m, z, r);
    
    //GPU dot
    //temp1_cuda = dot_cuda(m,z,r,dx,dy,dz,recive);
   // printf("2222222222222222222\n");
  //  printf("temp1 = %lf\n",temp1);
/*
    if (temp1 ==temp1_cuda)
    {
        printf("dot temp1 CPU = GPU \n");
    }
*/
    //dot_device_result = dot_cuda(m,r,z);
         //   if (temp1 ==dot_device_result)
        //    {
        //        printf("2222222222222222222222 dot CPU = GPU \n");
       //     }


    // main CG loop
    while (iter++ < matrix) {
        SSS_VEC vp, vt;

        // t=A*p
        vp.n = b->n;
        vp.d = p;
        vt.n = b->n;
        vt.d = t;

        //CPU_SpMV
        //SSS_blas_mv_mxy(A, &vp, &vt);
        
        //GPU_SpMV
        spmv_cuda(A,&vp,&vt,d_row_ptr,d_col_idx,d_A_val,d_x_val,d_y_val);
      

        // alpha_k=(z_{k-1},r_{k-1})/(A*p_{k-1},p_{k-1})
        temp2 = SSS_blas_array_dot(m, t, p);
        //temp2_cuda = dot_cuda(m,t,p,dx,dy,dz,recive);
        if (temp2 ==temp2_cuda)
        {
            //printf("dot temp2 CPU = GPU \n");
        }
      //  dot_device_result_tmp2 = dot_cuda(m,t,p);
       // if (temp2 ==dot_device_result_tmp2)
       // {
      //      printf("temp2 dot CPU = GPU \n");
      //  }

        if (SSS_ABS(temp2) > SMALLFLOAT2) {
            alpha = temp1 / temp2;
        }
        else {                  // Possible breakdown
            goto RESTORE_BESTSOL;
        }

        // u_k=u_{k-1} + alpha_k*p_{k-1}
        SSS_blas_array_axpy(m, alpha, p, u->d);

        // r_k=r_{k-1} - alpha_k*A*p_{k-1}
        SSS_blas_array_axpy(m, -alpha, t, r);

        // compute residuals
        switch (stop_type) {
            case STOP_REL_RES:
                absres = SSS_blas_array_norm2(m, r);
                relres = absres / normr0;
                break;
            case STOP_REL_PRECRES:
                // z = B(r)
                SSS_blas_array_cp(m, r, z); /* No preconditioner */
                absres = sqrt(SSS_ABS(SSS_blas_array_dot(m, z, r)));
                printf("1111111111111111\n");
                //absres_cuda = sqrt(dot_cuda(m,z,r,dx,dy,dz,recive));
                if (absres_cuda ==absres)
                {
                    //printf("dot absres CPU = GPU \n");
                }
             //  dot_device_result_absres = sqrt(dot_cuda(m,z,r));
             //   if (temp2 ==dot_device_result_absres)
             //   {
            //        printf("absres dot CPU = GPU \n");
            //    }
                relres = absres / normr0;
                break;
            case STOP_MOD_REL_RES:
                absres = SSS_blas_array_norm2(m, r);
                relres = absres / normu;
                break;
        }

        // compute reducation factor of residual ||r||

        factor = absres / absres0;

        // output iteration information if needed
        //SSS_print_itinfo( stop_type, iter, relres, absres, factor);

        // safety net check: save the best-so-far solution
        if (absres < absres_best - maxdiff) {
            absres_best = absres;
            iter_best = iter;
            SSS_blas_array_cp(m, u->d, u_best);
        }

        // Check I: if soultion is close to zero, return ERROR_SOLVER_SOLSTAG
        infnormu = SSS_blas_array_norminf(m, u->d);
        if (infnormu <= sol_inf_tol) {
            iter = ERROR_SOLVER_SOLSTAG;
            break;
        }

        //  Check II: if staggenated, try to restart
        normu = SSS_blas_vec_norm2(u);

        // compute relative difference
        reldiff = SSS_ABS(alpha) * SSS_blas_array_norm2(m, p) / normu;
        if ((stag <= maxStag) & (reldiff < maxdiff)) {

            SSS_blas_array_cp(m, b->d, r);
            //SSS_blas_mv_amxpy(-1.0, A, u, &vr);
            alpha_spmv_cuda(-1.0,A,u,&vr,d_row_ptr,d_col_idx,d_A_val,d_x_val,d_y_val);


            // compute residuals
            switch (stop_type) {
                case STOP_REL_RES:
                    absres = SSS_blas_array_norm2(m, r);
                    relres = absres / normr0;
                    break;

                case STOP_REL_PRECRES:
                    // z = B(r)
                    SSS_blas_array_cp(m, r, z);     /* No preconditioner */
                    absres = sqrt(SSS_ABS(SSS_blas_array_dot(m, z, r)));
                    printf("22222222222222\n");
                //absres_cuda = sqrt(dot_cuda(m,z,r,dx,dy,dz,recive));
                if (absres_cuda ==absres)
                {
                    //printf("dot absres22 CPU = GPU \n");
                }

              //      dot_device_result_absres = sqrt(dot_cuda(m,z,r));
              //  if (temp2 ==dot_device_result_absres)
              //  {
               //     printf("// z = B(r) dot CPU = GPU \n");
              //  }

                    relres = absres / normr0;
                    break;

                case STOP_MOD_REL_RES:
                    absres = SSS_blas_array_norm2(m, r);
                    relres = absres / normu;
                    break;
            }

            if (relres < tol) {
                break;
            }
            else {
                if (stag >= maxStag) {
                    iter = ERROR_SOLVER_STAG;
                    break;
                }
                SSS_blas_array_set(m, p, 0.0);
                ++stag;
                ++restart_step;
            }
        }                       // end of staggnation check!

        // Check III: prevent false convergence
        if (relres < tol) {
            // compute residual r = b - Ax again
            SSS_blas_array_cp(m, b->d, r);
            //SSS_blas_mv_amxpy(-1.0, A, u, &vr);
            alpha_spmv_cuda(-1.0,A,u,&vr,d_row_ptr,d_col_idx,d_A_val,d_x_val,d_y_val);

            // compute residuals
            switch (stop_type) {
                case STOP_REL_RES:
                    absres = SSS_blas_array_norm2(m, r);
                    relres = absres / normr0;
                    break;

                case STOP_REL_PRECRES:
                    // z = B(r)
                    SSS_blas_array_cp(m, r, z);     /* No preconditioner */
                    absres = sqrt(SSS_ABS(SSS_blas_array_dot(m, z, r)));
                    printf("333333333333333333\n");
                //absres_cuda = sqrt(dot_cuda(m,z,r,dx,dy,dz,recive));
                if (absres_cuda ==absres)
                {
                    //printf("dot absres3333 CPU = GPU \n");
                }
                    relres = absres / normr0;
                    break;

                case STOP_MOD_REL_RES:
                    absres = SSS_blas_array_norm2(m, r);
                    relres = absres / normu;
                    break;
            }

            // check convergence
            if (relres < tol) break;

            if (more_step >= maxRestartStep) {
                iter = ERROR_SOLVER_TOLSMALL;
                break;
            }
            
            // prepare for restarting the method
            SSS_blas_array_set(m, p, 0.0);
            ++more_step;
            ++restart_step;
        }

        // save residual for next iteration
        absres0 = absres;

        // compute z_k = B(r_k)
        if (stop_type != STOP_REL_PRECRES) {
            SSS_blas_array_cp(m, r, z);     /* No preconditioner, B=I */
        }

        // compute beta_k = (z_k, r_k)/(z_{k-1}, r_{k-1})
        temp2 = SSS_blas_array_dot(m, z, r);
        ///printf("4444444444444444\n");
        //temp2_cuda = dot_cuda(m,z,r,dx,dy,dz,recive);
                if (temp2_cuda ==temp2)
                {
                    //printf("dot222 temp2_cuda CPU = GPU \n");
                }
        beta = temp2_cuda / temp1;
        temp1 = temp2_cuda;

        // compute p_k = z_k + beta_k*p_{k-1}
       SSS_blas_array_axpby(m, 1.0, z, beta, p);

    }

    RESTORE_BESTSOL:
    if (iter != iter_best) {
        SSS_VEC vb;

        vb.n = b->n;
        vb.d = u_best;

        // compute best residual
        SSS_blas_array_cp(m, b->d, r);
        //SSS_blas_mv_amxpy(-1.0, A, &vb, &vr);
        alpha_spmv_cuda(-1.0,A,&vb,&vr,d_row_ptr,d_col_idx,d_A_val,d_x_val,d_y_val);


        switch (stop_type) {
            case STOP_REL_RES:
                absres_best = SSS_blas_array_norm2(m, r);
                break;

            case STOP_REL_PRECRES:
                // z = B(r)
                SSS_blas_array_cp(m, r, z); /* No preconditioner */
                absres_best = sqrt(SSS_ABS(SSS_blas_array_dot(m, z, r)));
               // absres_cuda = sqrt(dot_cuda(m,z,r,dx,dy,dz,recive));
                if (absres_cuda ==absres_best)
                {
                    printf(" absres_cuda CPU = GPU \n");
                }
                break;

            case STOP_MOD_REL_RES:
                absres_best = SSS_blas_array_norm2(m, r);
                break;
        }

        if (absres > absres_best + maxdiff) {
            SSS_blas_array_cp(m, u_best, u->d);
            relres = absres_best / normr0;
        }
    }

     eofc:

    // clean up temp memory
    SSS_free(work);
    cudaFree(dx);
    cudaFree(dy);
    cudaFree(dz);
    free(recive);




    if (iter > matrix)
        return ERROR_SOLVER_matrix;
    else
        return iter;
}


static int SSS_solver_gmres(SSS_KRYLOV *ks)
{

   // printf("SSS_solver_gmres\n");
    SSS_MAT *A = ks->A;
    SSS_VEC *b = ks->b;
    SSS_VEC *x = ks->u;
    double tol = ks->tol;
    int matrix = ks->matrix;
    int restart = ks->restart;
    int stop_type = ks->stop_type;

    const int n = b->n;
    const int MIN_ITER = 0;
    const double maxdiff = tol * 1e-4;     // staganation tolerance
    const double epsmac = SMALLFLOAT;
    int iter = 0;
    int restart1 = restart + 1;
    int i, j, k;

    double r_norm, r_normb, gamma, t;
    double r_normb_cuda;
    double normr0 = BIGFLOAT, absres = BIGFLOAT;
    double relres = BIGFLOAT, normu = BIGFLOAT;

    int iter_best = 0;          // initial best known iteration
    double absres_best = BIGFLOAT;       // initial best known residual

    // allocate temp memory (need about (restart+4)*n FLOAT numbers)
    double *c = NULL, *s = NULL, *rs = NULL;
    double *norms = NULL, *r = NULL, *w = NULL;
    double *work = NULL, *x_best = NULL;
    double **p = NULL, **hh = NULL;
    SSS_VEC vs, vr;


    /* allocate memory and setup temp work space */
    work = (double *) SSS_calloc((restart + 4) * (restart + n) + 1, sizeof(double));

    /* check whether memory is enough for GMRES */
    while ((work == NULL) && (restart > 5)) {
        restart = restart - 5;
        work = (double *) SSS_calloc((restart + 4) * (restart + n) + 1, sizeof(double));
        printf("### WARNING: GMRES restart number set to %d!\n", restart);
        restart1 = restart + 1;
    }

    if (work == NULL) {
        printf("### ERROR: No enough memory for GMRES %s : %s: %d !\n", __FILE__,
                __FUNCTION__, __LINE__);

        exit(ERROR_ALLOC_MEM);
    }


    double *dx = NULL;
    double *dy = NULL;
    double *dz = NULL;

    double *dhh = NULL;

    cudaMalloc((void **)&dx,((restart + 4) * (restart + n) + 1) * sizeof(double));
    cudaMalloc((void **)&dy,((restart + 4) * (restart + n) + 1) * sizeof(double));
    cudaMalloc((void **)&dz,((restart + 4) * (restart + n) + 1) * sizeof(double));


    double *recive =NULL;
    recive =(double *)malloc(((restart + 4) * (restart + n) + 1)*sizeof(double));

    if (dx == NULL || dy ==NULL || dz == NULL) 
    {
        printf("could't allocate GPU mem \n");
        return -1;
    }

    
    int *d_row_ptr = NULL;
    int *d_col_idx = NULL;
    double *d_A_val = NULL;
    double *d_x_val = NULL;
    double *d_y_val = NULL;

    int m = A->num_rows;
    int nnz = A->num_nnzs;

    cudaMalloc((void **)&d_row_ptr,(m+1) * sizeof(int));
    cudaMalloc((void **)&d_col_idx,nnz * sizeof(int));
    cudaMalloc((void **)&d_A_val,  nnz * sizeof(double));
    cudaMalloc((void **)&d_x_val,  m * sizeof(double));
    cudaMalloc((void **)&d_y_val,  m * sizeof(double));

    if (d_row_ptr == NULL || d_col_idx ==NULL || d_A_val == NULL ||d_x_val ==NULL ||d_y_val ==NULL) 
    {
        printf("could't allocate GPU mem \n");
        return -1;
    }















    p = (double **) SSS_calloc(restart1, sizeof(double *));

    hh = (double **) SSS_calloc(restart1, sizeof(double *));
    cudaMalloc((void **)&dhh,restart1 * sizeof(double));

    norms = (double *) SSS_calloc(matrix + 1, sizeof(double));

    r = work;
    w = r + n;
    rs = w + n;
    c = rs + restart1;
    x_best = c + restart;
    s = x_best + n;
    vr.n = vs.n = b->n;

    for (i = 0; i < restart1; i++) p[i] = s + restart + i * n;

    for (i = 0; i < restart1; i++) hh[i] = p[restart] + n + i * restart;

    // r = b-A*x
    SSS_blas_array_cp(n, b->d, p[0]);

    vs.d = p[0];
    //SSS_blas_mv_amxpy(-1.0, A, x, &vs);
    alpha_spmv_cuda(-1.0,A,x,&vs,d_row_ptr,d_col_idx,d_A_val,d_x_val,d_y_val);

    r_norm = SSS_blas_array_norm2(n, p[0]);

    // compute initial residuals
    switch (stop_type) {
        case STOP_REL_RES:
            normr0 = SSS_max(SMALLFLOAT, r_norm);
            relres = r_norm / normr0;
            break;

        case STOP_REL_PRECRES:
            SSS_blas_array_cp(n, p[0], r);
            r_normb = sqrt(SSS_blas_array_dot(n, p[0], r));
            printf("00000000000000000000000\n");
            //r_normb_cuda =sqrt(dot_cuda(n,p[0],r,dx,dy,dz,recive));
            if (r_normb_cuda ==r_normb)
            {
                printf(" dot _ r_normb CPU = GPU \n");
            }

            normr0 = SSS_max(SMALLFLOAT, r_normb);
            relres = r_normb / normr0;
            break;

        case STOP_MOD_REL_RES:
            normu = SSS_max(SMALLFLOAT, SSS_blas_array_norm2(n, x->d));
            normr0 = r_norm;
            relres = normr0 / normu;
            break;

        default:
            printf("### ERROR: Unrecognized stopping type for %s!\n", __FUNCTION__);
            goto eofc;
    }

    // if initial residual is small, no need to iterate!
    if (relres < tol) goto eofc;

    // output iteration information if needed
   // SSS_print_itinfo(stop_type, 0, relres, normr0, 0.0);

    // store initial residual
    norms[0] = relres;

    /* outer iteration cycle */
    while (iter < matrix) {
        rs[0] = r_norm;
        t = 1.0 / r_norm;
        SSS_blas_array_ax(n, t, p[0]);

        /* RESTART CYCLE (right-preconditioning) */
        i = 0;
        while (i < restart && iter < matrix) {
            SSS_VEC vp;

            i++;
            iter++;

            vr.n = b->n;
            vp.n = b->n;
            vr.d = r;
            vp.d = p[i];

            SSS_blas_array_cp(n, p[i - 1], r);
            //SSS_blas_mv_mxy(A, &vr, &vp);
            spmv_cuda(A,&vr,&vp,d_row_ptr,d_col_idx,d_A_val,d_x_val,d_y_val);

            /* modified Gram_Schmidt */
            for (j = 0; j < i; j++) {
                hh[j][i - 1] = SSS_blas_array_dot(n, p[j], p[i]);
                //dot_cuda(n,p[0],r,dx,dy,hh[j][i-1]));

                SSS_blas_array_axpy(n, -hh[j][i - 1], p[j], p[i]);
            }
            t = SSS_blas_array_norm2(n, p[i]);
            hh[i][i - 1] = t;
            if (t != 0.0) {
                t = 1.0 / t;
                SSS_blas_array_ax(n, t, p[i]);
            }

            for (j = 1; j < i; ++j) {
                t = hh[j - 1][i - 1];
                hh[j - 1][i - 1] = s[j - 1] * hh[j][i - 1] + c[j - 1] * t;
                hh[j][i - 1] = -s[j - 1] * t + c[j - 1] * hh[j][i - 1];
            }

            t = hh[i][i - 1] * hh[i][i - 1];
            t += hh[i - 1][i - 1] * hh[i - 1][i - 1];

            gamma = sqrt(t);
            if (gamma == 0.0) gamma = epsmac;

            c[i - 1] = hh[i - 1][i - 1] / gamma;
            s[i - 1] = hh[i][i - 1] / gamma;
            rs[i] = -s[i - 1] * rs[i - 1];
            rs[i - 1] = c[i - 1] * rs[i - 1];
            hh[i - 1][i - 1] = s[i - 1] * hh[i][i - 1] + c[i - 1] * hh[i - 1][i - 1];

            absres = r_norm = fabs(rs[i]);

            relres = absres / normr0;

            norms[iter] = relres;

            // output iteration information if needed
          //  SSS_print_itinfo(stop_type, iter, relres, absres, norms[iter] / norms[iter - 1]);

            // should we exit the restart cycle
            if (relres <= tol && iter >= MIN_ITER) break;

        }

        /* compute solution, first solve upper triangular system */
        rs[i - 1] = rs[i - 1] / hh[i - 1][i - 1];
        for (k = i - 2; k >= 0; k--) {
            t = 0.0;
            for (j = k + 1; j < i; j++) t -= hh[k][j] * rs[j];

            t += rs[k];
            rs[k] = t / hh[k][k];
        }

        SSS_blas_array_cp(n, p[i - 1], w);
        SSS_blas_array_ax(n, rs[i - 1], w);

        for (j = i - 2; j >= 0; j--) SSS_blas_array_axpy(n, rs[j], p[j], w);

        /* apply the preconditioner */
        SSS_blas_array_cp(n, w, r);
        SSS_blas_array_axpy(n, 1.0, r, x->d);

        if (absres < absres_best - maxdiff) {
            absres_best = absres;
            iter_best = iter;
            SSS_blas_array_cp(n, x->d, x_best);
        }

        // Check: prevent false convergence
        if (relres <= tol && iter >= MIN_ITER) {

            SSS_blas_array_cp(n, b->d, r);

            vs.d = r;
            //SSS_blas_mv_amxpy(-1.0, A, x, &vs);
            alpha_spmv_cuda(-1.0,A,x,&vs,d_row_ptr,d_col_idx,d_A_val,d_x_val,d_y_val);


            r_norm = SSS_blas_array_norm2(n, r);

            switch (stop_type) {
                case STOP_REL_RES:
                    absres = r_norm;
                    relres = absres / normr0;
                    break;

                case STOP_REL_PRECRES:
                    SSS_blas_array_cp(n, r, w);
                    absres = sqrt(SSS_blas_array_dot(n, w, r));
                    relres = absres / normr0;
                    break;

                case STOP_MOD_REL_RES:
                    absres = r_norm;
                    normu = SSS_max(SMALLFLOAT, SSS_blas_array_norm2(n, x->d));
                    relres = absres / normu;
                    break;
            }

            norms[iter] = relres;

            if (relres <= tol) {
                break;
            }
            else {
                // Need to restart
                SSS_blas_array_cp(n, r, p[0]);
                i = 0;
            }

        }

        /* compute residual vector and continue loop */
        for (j = i; j > 0; j--) {
            rs[j - 1] = -s[j - 1] * rs[j];
            rs[j] = c[j - 1] * rs[j];
        }

        if (i) SSS_blas_array_axpy(n, rs[i] - 1.0, p[i], p[i]);

        for (j = i - 1; j > 0; j--)
            SSS_blas_array_axpy(n, rs[j], p[j], p[i]);

        if (i) {
            SSS_blas_array_axpy(n, rs[0] - 1.0, p[0], p[0]);
            SSS_blas_array_axpy(n, 1.0, p[i], p[0]);
        }
    }                           /* end of main while loop */

    if (iter != iter_best) {
        // compute best residual
        SSS_blas_array_cp(n, b->d, r);

        vs.d = x_best;
        vr.d = r;
        //SSS_blas_mv_amxpy(-1.0, A, &vs, &vr);
        alpha_spmv_cuda(-1.0,A,&vs,&vr,d_row_ptr,d_col_idx,d_A_val,d_x_val,d_y_val);


        switch (stop_type) {
            case STOP_REL_RES:
                absres_best = SSS_blas_array_norm2(n, r);
                break;

            case STOP_REL_PRECRES:
                // z = B(r)
                SSS_blas_array_cp(n, r, w); /* No preconditioner */
                absres_best = sqrt(SSS_ABS(SSS_blas_array_dot(n, w, r)));
                break;

            case STOP_MOD_REL_RES:
                absres_best = SSS_blas_array_norm2(n, r);
                break;
        }

        if (absres > absres_best + maxdiff) {
            SSS_blas_array_cp(n, x_best, x->d);
            relres = absres_best / normr0;
        }
    }

 eofc:
    SSS_free(work);
    SSS_free(p);
    SSS_free(hh);
    SSS_free(norms);

    if (iter >= matrix)
        return ERROR_SOLVER_matrix;
    else
        return iter;
}

void SSS_amg_coarest_solve(SSS_MAT *A, SSS_VEC *b, SSS_VEC *x, const double ctol)
{
    const int n = A->num_rows;
    const int matrix = SSS_max(250, SSS_MIN(n * n, 1000));

    int status;
    SSS_KRYLOV ks;

    /* try cg first */
    ks.A = A;
    ks.b = b;
    ks.u = x;
    ks.tol = ctol;
    ks.matrix = matrix;
    ks.stop_type = 1;
    status = SSS_solver_cg(&ks);

    /* try GMRES if cg fails */
    if (status < 0) {
        
        ks.restart = max_RESTART;
        status = SSS_solver_gmres(&ks);
    }

    if (status < 0) {
        printf("### WARNING: Coarse level solver failed to converge!\n");
    }
}

void SSS_amg_cycle(SSS_AMG *mg)
{
    double spmv_time = 0;
    int cycle_type = mg->pars.cycle_type;
    int nl = mg->num_levels;
    double tol = mg->pars.ctol;

    double alpha = 1.0;
    int num_lvl[max_AMG_LVL] = {0}, l = 0;

    if (tol > mg->pars.tol)  tol = mg->pars.tol * 0.1;
    if (cycle_type <= 0) cycle_type = 1;// V-cycle

    ForwardSweep:
    while (l < nl - 1) {
        SSS_SMTR s;

        num_lvl[l]++;

        // pre-smoothing
        s.smoother = mg->pars.smoother;
        s.A = &mg->cg[l].A;
        s.b = &mg->cg[l].b;
        s.x = &mg->cg[l].x;
        //s. = &mg->cg[l].R;
        s.nsweeps = mg->pars.pre_iter;
        s.istart = 0;
        s.iend = mg->cg[l].A.num_rows - 1;
        s.istep = 1;
        s.relax = mg->pars.relax;
        s.ndeg = mg->pars.poly_deg;
        s.cf_order = mg->pars.cf_order;
        s.ordering = mg->cg[l].cfmark.d;

        //smoothing could't use cuda!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!



        /*
        int m = mg->cg[l].R.num_rows;
       // printf("m = %d\n",m);
        int nnz = mg->cg[l].R.num_nnzs;
       // printf("nnz = %d\n",nnz);

        int *d_row_ptr = NULL;
        int *d_col_idx = NULL;
        double *d_A_val = NULL;
        double *d_x_val = NULL;
        double *d_y_val = NULL;


        cudaMalloc((void **)&d_row_ptr,(m+1) * sizeof(int));
        cudaMalloc((void **)&d_col_idx,nnz * sizeof(int));
        cudaMalloc((void **)&d_A_val,  nnz * sizeof(double));

        cudaMalloc((void **)&d_x_val,  m * sizeof(double));
        cudaMalloc((void **)&d_y_val,  m * sizeof(double));

        if (d_row_ptr == NULL || d_col_idx ==NULL || d_A_val == NULL ||d_x_val ==NULL ||d_y_val ==NULL) 
        {
            printf("could't allocate GPU mem \n");
        }

        */

        SSS_amg_smoother_pre(&s);

        // form residual r = b - A x
        SSS_blas_array_cp(mg->cg[l].A.num_rows, mg->cg[l].b.d, mg->cg[l].wp.d);
        SSS_blas_mv_amxpy(-1.0, &mg->cg[l].A, &mg->cg[l].x, &mg->cg[l].wp);
        

        // restriction r1 = R*r0
         SSS_blas_mv_mxy(&mg->cg[l].R, &mg->cg[l].wp, &mg->cg[l + 1].b);
        //spmv_cudaspmv_cuda(&mg->cg[l].R,&mg->cg[l].wp, &mg->cg[l + 1].b,d_row_ptr,d_col_idx,d_A_val,d_x_val,d_y_val);

        


        // prepare for the next level
        l++;
        SSS_vec_set_value(&mg->cg[l].x, 0.0);
    }

    // call the coarse space solver:
    SSS_amg_coarest_solve(&mg->cg[nl - 1].A, &mg->cg[nl - 1].b, &mg->cg[nl - 1].x, tol);

    //BackwardSweep:
    while (l > 0) {
        SSS_SMTR s;

        l--;

        // prolongation u = u + alpha*P*e1
        SSS_blas_mv_amxpy(alpha, &mg->cg[l].P, &mg->cg[l + 1].x, &mg->cg[l].x);

        // post-smoothing
        s.smoother = mg->pars.smoother;
        s.A = &mg->cg[l].A;
        s.b = &mg->cg[l].b;
        s.x = &mg->cg[l].x;
        s.nsweeps = mg->pars.post_iter;
        s.istart = 0;
        s.iend = mg->cg[l].A.num_rows - 1;
        s.istep = -1;
        s.relax = mg->pars.relax;
        s.ndeg = mg->pars.poly_deg;
        s.cf_order = mg->pars.cf_order;
        s.ordering = mg->cg[l].cfmark.d;

        SSS_amg_smoother_post(&s);

        if (num_lvl[l] < cycle_type)
            break;
        else
            num_lvl[l] = 0;
    }

    if (l > 0) goto ForwardSweep;
}
