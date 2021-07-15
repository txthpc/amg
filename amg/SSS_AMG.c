#include "SSS_matvec.h"
#include "SSS_utils.h"
#include "Setup/SSS_SETUP.h"
#include "Solve/SSS_SOLVE.h"

//#include <math.h>


SSS_RTN SSS_solver_amg(SSS_MAT *A, SSS_VEC *x, SSS_VEC *b, SSS_AMG_PARS *pars)
{
    int nnz, m, n;
    SSS_RTN rtn;
    SSS_AMG_PARS npars;

    SSS_AMG mg;
    double AMG_start, AMG_end;
    double sumb;

    /* 计算向量b的L2范数 */
    //将其做为绝对误差？？
    sumb = SSS_blas_vec_norm2(b);

    if (fabs(sumb) == 0.) {
        SSS_vec_set_value(x, 0);
        rtn.ares = 0;
        rtn.rres = 0;
        rtn.nits = 0;
        SSS_print_itinfo(STOP_REL_RES, 0, 0., sumb, 0.0);
        return rtn;
    }

    nnz = A->num_nnzs;
    m = A->num_rows;
    n = A->num_cols;

    AMG_start = SSS_get_time();

    // check matrix data
    if (m != n) {
        printf("### ERROR: A is not a square matrix!\n");
    }

    if (nnz <= 0) {
        printf("### ERROR: A has no nonzero entries!\n");
    }

    // Step 1: AMG setup phase
    SSS_amg_setup(&mg, A, pars);

    // Step 2: AMG solve phase
    rtn = SSS_amg_solve(&mg, x, b);

    // clean-up memory
    SSS_amg_data_destroy(&mg);

    // print out CPU time if needed
        AMG_end = SSS_get_time();
        printf("AMG totally time: %g s\n", AMG_end - AMG_start);
    
    return rtn;
}