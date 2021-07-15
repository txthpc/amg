
#include "SSS_SOLVE.h"

SSS_RTN SSS_amg_solve(SSS_AMG *mg, SSS_VEC *x, SSS_VEC *b)
{
    SSS_MAT *ptrA;
    SSS_VEC *r;

    int max_it;
    double tol;
    double sumb; // L2norm(b) 

    double solve_start, solve_end;
    double relres1 = 1., absres0, absres, factor;
    int iter = 0;
    SSS_RTN rtn;

    /* check pars */
    assert(mg != NULL);
    assert(x != NULL);
    assert(b != NULL);

    ptrA = &mg->cg[0].A;
    r = &mg->cg[0].wp;

    max_it = mg->pars.max_it;
    tol = mg->pars.tol;
    sumb = SSS_blas_vec_norm2(b); // L2norm(b)
    absres0 = sumb;

    solve_start = SSS_get_time(); 

    // Print iteration information if needed
    SSS_print_itinfo( STOP_REL_RES, iter, 1.0, sumb, 0.0);

    /* init, to make compiler hAppy */
    rtn.ares = 0;
    rtn.rres = 0;
    rtn.nits = 0;

    if (fabs(sumb) == 0.) {
        SSS_vec_set_value(x, 0);
        mg->rtn = rtn;

        return rtn;
    }

    /* set x and b */
    mg->cg[0].x = *x;
    mg->cg[0].b = *b;

    // MG solver here
    while ((++iter <= max_it)) {

        SSS_amg_cycle(mg);


        // Form residual r = b - A*x
        SSS_vec_cp(b, r);
        SSS_blas_mv_amxpy(-1.0, ptrA, x, r);
        

        // Compute norms of r and convergence factor
        absres = SSS_blas_vec_norm2(r);  // residual ||r||
        relres1 = absres / sumb;        // relative residual ||r||/||b||
        factor = absres / absres0;      // contraction factor
        absres0 = absres;               // prepare for next iteration

        // Print iteration information if needed
        SSS_print_itinfo( STOP_REL_RES, iter, relres1, absres, factor);

        /* save convergence info */
        rtn.ares = absres;
        rtn.rres = relres1;
        rtn.nits = iter;
        mg->rtn = rtn;

        // Check convergence
        if (relres1 < tol) break;
    }

        solve_end = SSS_get_time();
        printf("AMG solve time: %g s\n", solve_end - solve_start);
    

    return rtn;
}