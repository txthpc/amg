#include "SSS_smooth.h"


static void SSS_amg_smoother_gs_cf(SSS_VEC * u, SSS_MAT * A, SSS_VEC * b, int L, int * mark,
        const int order)
{
    const int nrow = b->n;    // number of rows
    const int *ia = A->row_ptr, *ja = A->col_idx;
    const double *Aj = A->val, *bAx = b->d;
    double *uAx = u->d;

    int i, j, k, begin_row, end_row;
    double t, d = 0.0;

    // F-point first, C-point second
    if (order) {
        while (L--) {
            for (i = 0; i < nrow; i++) {
                if (mark[i] != 1) {
                    t = bAx[i];
                    begin_row = ia[i], end_row = ia[i + 1];
                    for (k = begin_row; k < end_row; k++) {
                        j = ja[k];
                        if (i != j)
                            t -= Aj[k] * uAx[j];
                        else
                            d = Aj[k];
                    }           // end for k

                    if (SSS_ABS(d) > SMALLFLOAT) uAx[i] = t / d;
                }
            }

            for (i = 0; i < nrow; i++) {
                if (mark[i] == 1) {
                    t = bAx[i];
                    begin_row = ia[i], end_row = ia[i + 1];
                    for (k = begin_row; k < end_row; k++) {
                        j = ja[k];
                        if (i != j)
                            t -= Aj[k] * uAx[j];
                        else
                            d = Aj[k];
                    }           // end for k

                    if (SSS_ABS(d) > SMALLFLOAT) uAx[i] = t / d;
                }
            }                   // end for i
        }                       // end while
    }
    else {                      // C-point first, F-point second
        while (L--) {
            for (i = 0; i < nrow; i++) {
                if (mark[i] == 1) {
                    t = bAx[i];
                    begin_row = ia[i], end_row = ia[i + 1];
                    for (k = begin_row; k < end_row; k++) {
                        j = ja[k];
                        if (i != j)
                            t -= Aj[k] * uAx[j];
                        else
                            d = Aj[k];
                    }           // end for k

                    if (SSS_ABS(d) > SMALLFLOAT) uAx[i] = t / d;
                }
            }

            for (i = 0; i < nrow; i++) {
                if (mark[i] != 1) {
                    t = bAx[i];
                    begin_row = ia[i], end_row = ia[i + 1];
                    for (k = begin_row; k < end_row; k++) {
                        j = ja[k];
                        if (i != j)
                            t -= Aj[k] * uAx[j];
                        else
                            d = Aj[k];
                    }           // end for k

                    if (SSS_ABS(d) > SMALLFLOAT)
                        uAx[i] = t / d;
                }
            }                   // end for i
        }                       // end while
    }                           // end if order
}


static void SSS_amg_smoother_gs(SSS_VEC * u, const int i_1, const int i_n, const int s,
        SSS_MAT *A, SSS_VEC *b, int L)
{
    const int *ia = A->row_ptr, *ja = A->col_idx;
    const double *Aj = A->val, *bAx = b->d;
    double *uAx = u->d;

    // local variables
    int i, j, k, begin_row, end_row;
    double t, d = 0.0;

    if (s > 0) {
        while (L--) {
            for (i = i_1; i <= i_n; i += s) {
                t = bAx[i];
                begin_row = ia[i], end_row = ia[i + 1];

                for (k = begin_row; k < end_row; ++k) {
                    j = ja[k];
                    if (i != j)
                        t -= Aj[k] * uAx[j];
                    else if (SSS_ABS(Aj[k]) > SMALLFLOAT)
                        d = 1.e+0 / Aj[k];
                }

                uAx[i] = t * d;
            }                   // end for i
        }                       // end while
    }                           // if s
    else {

        while (L--) {
            for (i = i_1; i >= i_n; i += s) {
                t = bAx[i];
                begin_row = ia[i], end_row = ia[i + 1];
                for (k = begin_row; k < end_row; ++k) {
                    j = ja[k];
                    if (i != j)
                        t -= Aj[k] * uAx[j];
                    else if (SSS_ABS(Aj[k]) > SMALLFLOAT)
                        d = 1.0 / Aj[k];
                }

                uAx[i] = t * d;
            }                   // end for i
        }                       // end while
    }                           // end if
}
void SSS_amg_smoother_pre(SSS_SMTR *s)
{
    SSS_SM_TYPE smoother;
    SSS_MAT * A;
    SSS_VEC * b;
    SSS_VEC * x;
    int nsweeps;
    int istart;
    int iend;
    int istep;
    double relax;
    int ndeg;
    int order;
    int *ordering;

    assert(s != NULL);

    smoother = s->smoother;
    A = s->A;
    b = s->b;
    x = s->x;
    nsweeps = s->nsweeps;
    istart = s->istart;
    iend = s->iend;
    istep = s->istep;
    relax = s->relax;
    ndeg = s->ndeg;
    order = s->cf_order;
    ordering = s->ordering;

    switch (smoother) {
        case SSS_SM_GS:
            if (order && ordering != NULL) {
                SSS_amg_smoother_gs_cf(x, A, b, nsweeps, ordering, 1);
            }
            else {

                SSS_amg_smoother_gs(x, istart, iend, istep, A, b, nsweeps);
            }
            break;
/*
        case SSS_SM_SGS:
            SSS_amg_smoother_sgs(x, A, b, nsweeps);
            break;

        case SSS_SM_JACOBI:
            SSS_amg_smoother_jacobi(x, istart, iend, istep, A, b, nsweeps);
            break;

        case SSS_SM_L1DIAG:
            SSS_amg_smoother_L1diag(x, istart, iend, istep, A, b, nsweeps);
            break;

        case SSS_SM_POLY:
            SSS_amg_smoother_poly(A, b, x, iend + 1, ndeg, nsweeps);
            break;

        case SSS_SM_SOR:
            SSS_amg_smoother_sor(x, istart, iend, istep, A, b, nsweeps, relax);
            break;

        case SSS_SM_SSOR:
            SSS_amg_smoother_sor(x, istart, iend, istep, A, b, nsweeps, relax);
            SSS_amg_smoother_sor(x, iend, istart, -istep, A, b, nsweeps, relax);
            break;

        case SSS_SM_GSOR:
            SSS_amg_smoother_gs(x, istart, iend, istep, A, b, nsweeps);
            SSS_amg_smoother_sor(x, iend, istart, -istep, A, b, nsweeps, relax);
            break;

        case SSS_SM_SGSOR:
            SSS_amg_smoother_gs(x, istart, iend, istep, A, b, nsweeps);
            SSS_amg_smoother_gs(x, iend, istart, -istep, A, b, nsweeps);
            SSS_amg_smoother_sor(x, istart, iend, istep, A, b, nsweeps, relax);
            SSS_amg_smoother_sor(x, iend, istart, -istep, A, b, nsweeps, relax);
            break;
*/
        default:
            printf("### ERROR: Wrong smoother type %d!\n", smoother);
            SSS_exit_on_errcode(ERROR_INPUT_PAR, __FUNCTION__);
    }
}


void SSS_amg_smoother_post(SSS_SMTR *s)
{
    SSS_SM_TYPE smoother;
    SSS_MAT * A;
    SSS_VEC * b;
    SSS_VEC * x;
    int nsweeps;
    int istart;
    int iend;
    int istep;
    double relax;
    int ndeg;
    int order;
    int *ordering;

    assert(s != NULL);

    smoother = s->smoother;
    A = s->A;
    b = s->b;
    x = s->x;
    nsweeps = s->nsweeps;
    istart = s->istart;
    iend = s->iend;
    istep = s->istep;
    relax = s->relax;
    ndeg = s->ndeg;
    order = s->cf_order;
    ordering = s->ordering;

    switch (smoother) {
        case SSS_SM_GS:
            if (order && ordering != NULL) {
                SSS_amg_smoother_gs_cf(x, A, b, nsweeps, ordering, -1);
            }
            else {
                SSS_amg_smoother_gs(x, iend, istart, istep, A, b, nsweeps);
            }
            break;
/*
        case SSS_SM_SGS:
            SSS_amg_smoother_sgs(x, A, b, nsweeps);
            break;

        case SSS_SM_JACOBI:
            SSS_amg_smoother_jacobi(x, iend, istart, istep, A, b, nsweeps);
            break;

        case SSS_SM_L1DIAG:
            SSS_amg_smoother_L1diag(x, iend, istart, istep, A, b, nsweeps);
            break;

        case SSS_SM_POLY:
            SSS_amg_smoother_poly(A, b, x, iend + 1, ndeg, nsweeps);
            break;

        case SSS_SM_SOR:
            SSS_amg_smoother_sor(x, iend, istart, istep, A, b, nsweeps, relax);
            break;

        case SSS_SM_SSOR:
            SSS_amg_smoother_sor(x, istart, iend, -istep, A, b, nsweeps, relax);
            SSS_amg_smoother_sor(x, iend, istart, istep, A, b, nsweeps, relax);
            break;

        case SSS_SM_GSOR:
            SSS_amg_smoother_sor(x, istart, iend, -istep, A, b, nsweeps, relax);
            SSS_amg_smoother_gs(x, iend, istart, istep, A, b, nsweeps);
            break;

        case SSS_SM_SGSOR:
            SSS_amg_smoother_sor(x, istart, iend, -istep, A, b, nsweeps, relax);
            SSS_amg_smoother_sor(x, iend, istart, istep, A, b, nsweeps, relax);
            SSS_amg_smoother_gs(x, istart, iend, -istep, A, b, nsweeps);
            SSS_amg_smoother_gs(x, iend, istart, istep, A, b, nsweeps);
            break;
*/
        default:
            printf("### ERROR: Wrong smoother type %d!\n", smoother);
            SSS_exit_on_errcode(ERROR_INPUT_PAR, __FUNCTION__);
    }
}