#include "SSS_matvec.h"
//calloc
void * SSS_calloc(size_t size, int type)
{
    const size_t tsize = size * type;

    void *mem = NULL;

    if (tsize > 0) {
        mem = calloc(size, type);
    }

    if (mem == NULL) {
        printf("### WARNING: Cannot allocate %.3lf MB RAM!\n",
               (double) tsize / 1048576);
    }

    return mem;
}

//分配vec空间
SSS_VEC SSS_vec_create(int m)
{
    SSS_VEC u;

    assert(m >= 0);

    u.n = m;
    u.d = (double *) SSS_calloc(m, sizeof(double));

    return u;
}

//给vec赋值
void SSS_vec_set_value(SSS_VEC *x, double val)
{
    int i;
    double *xpt = x->d;

    for (i = 0; i < x->n; ++i) xpt[i] = val;
}

//释放mat内存空间
void SSS_mat_destroy(SSS_MAT *A)
{

    if (A == NULL) return;

    SSS_free(A->row_ptr);
    A->row_ptr = NULL;

    SSS_free(A->col_idx);
    A->col_idx = NULL;

    SSS_free(A->val);
    A->val = NULL;
}

//释放vec内存空间
void SSS_vec_destroy(SSS_VEC * u)
{
    if (u == NULL) return;

    SSS_free(u->d);
    u->n = 0;
    u->d = NULL;
}

//创建mg网格结构
SSS_AMG SSS_amg_data_create(SSS_AMG_PARS *pars)
{
    SSS_AMG mg;

    assert(pars != NULL);
    assert(pars->max_levels > 0);

    bzero(&mg, sizeof(mg));
    mg.cg = SSS_calloc(pars->max_levels, sizeof(*mg.cg));

    /* record pars */
    mg.pars = *pars;

    return mg;
}


SSS_IVEC SSS_ivec_create(int m)
{
    SSS_IVEC u;

    assert(m >= 0);

    u.n = m;
    u.d = (int *) SSS_calloc(m, sizeof(int));

    return u;
}


//Create CSR sparse matrix data memory space
SSS_MAT SSS_mat_struct_create(int m, int n, int nnz)
{
    SSS_MAT A;

    assert(m > 0);
    assert(n > 0);
    assert(nnz >= 0);

    A.row_ptr = (int *) SSS_calloc(m + 1, sizeof(int));

    if (nnz > 0) {
        A.col_idx = (int *) SSS_calloc(nnz, sizeof(int));
    }
    else {
        A.col_idx = NULL;
    }

    if (nnz > 0) {
        A.val = (double *) SSS_calloc(nnz, sizeof(double));
    }
    else {
        A.val = NULL;
    }

    A.num_rows = m;
    A.num_cols = n;
    A.num_nnzs = nnz;

    return A;
}

//Copy an array to the other y=x
void SSS_iarray_cp(const int n, int *x, int *y)
{
    assert(x != NULL);
    assert(y != NULL);
    assert(n > 0);

    memcpy(y, x, n * sizeof(int));
}

//Copy an array to the other y=x
void SSS_blas_array_cp(int n, const double *x, double *y)
{
    memcpy(y, x, n * sizeof(double));
}

//copy a SSS_MAT to a new one des=src
void SSS_mat_cp(SSS_MAT *src, SSS_MAT *des)
{

    des->num_rows = src->num_rows;
    des->num_cols = src->num_cols;
    des->num_nnzs = src->num_nnzs;

    SSS_iarray_cp(src->num_rows + 1, src->row_ptr, des->row_ptr);
    SSS_iarray_cp(src->num_nnzs, src->col_idx, des->col_idx);
    SSS_blas_array_cp(src->num_nnzs, src->val, des->val);
}

//Get first n diagonal entries of a CSR matrix A
SSS_VEC SSS_mat_get_diag(SSS_MAT *A, int n)
{

    int i, k, j, ibegin, iend;
    SSS_VEC diag;

    if (n == 0 || n > A->num_rows || n > A->num_cols)
        n = SSS_MIN(A->num_rows, A->num_cols);

    diag.n = n;
    diag.d = (double *) SSS_calloc(n, sizeof(double));

    for (i = 0; i < n; ++i) {
        ibegin = A->row_ptr[i];
        iend = A->row_ptr[i + 1];
        for (k = ibegin; k < iend; ++k) {
            j = A->col_idx[k];
            if ((j - i) == 0) {
                diag.d[i] = A->val[k];
                break;
            }
        }
    }

    return diag;
}


//Free vector data space of int type
void SSS_ivec_destroy(SSS_IVEC * u)
{
    if (u == NULL) return;

    SSS_free(u->d);
    u->n = 0;
    u->d = NULL;
}


//Free SSS_AMG data memeory space
void SSS_amg_data_destroy(SSS_AMG *mg)
{

    int max_levels;
    int i;

    if (mg == NULL) return;

    max_levels = SSS_max(1, mg->num_levels);

    for (i = 0; i < max_levels; ++i) {
        SSS_mat_destroy(&mg->cg[i].A);
        SSS_mat_destroy(&mg->cg[i].P);
        SSS_mat_destroy(&mg->cg[i].R);

        if (i > 0) {
            SSS_vec_destroy(&mg->cg[i].b);
            SSS_vec_destroy(&mg->cg[i].x);
        }

        SSS_vec_destroy(&mg->cg[i].wp);
        SSS_ivec_destroy(&mg->cg[i].cfmark);
    }

    SSS_free(mg->cg);
    bzero(mg, sizeof(*mg));
}

//Set initial value for an array to be x=Ax
void SSS_iarray_set(const int n, int *x, const int Ax)
{
    int i;

    assert(x != NULL);
    assert(n > 0);

    if (Ax == 0) {
        memset(x, 0, sizeof(int) * n);
    }
    else {
        for (i = 0; i < n; ++i) x[i] = Ax;
    }
}

//Find transpose of SSS_IMAT matrix A
SSS_IMAT SSS_imat_trans(SSS_IMAT *A)
{

    const int n = A->num_rows, m = A->num_cols, nnz = A->num_nnzs, m1 = m - 1;

    // Local variables
    int i, j, k, p;
    int ibegin, iend;
    SSS_IMAT AT;

    AT.num_rows = m;
    AT.num_cols = n;
    AT.num_nnzs = nnz;

    AT.row_ptr = (int *) SSS_calloc(m + 1, sizeof(int));

    AT.col_idx = (int *) SSS_calloc(nnz, sizeof(int));

    if (A->val) {
        AT.val = (int *) SSS_calloc(nnz, sizeof(int));
    }
    else {
        AT.val = NULL;
    }

    // first pass: find the Number of nonzeros in the first m-1 columns of A
    // Note: these Numbers are stored in the array AT.row_ptr from 1 to m-1
    SSS_iarray_set(m + 1, AT.row_ptr, 0);

    for (j = 0; j < nnz; ++j) {
        i = A->col_idx[j];           // column Number of A = row Number of A'
        if (i < m1) AT.row_ptr[i + 2]++;
    }

    for (i = 2; i <= m; ++i) AT.row_ptr[i] += AT.row_ptr[i - 1];

    // second pass: form A'
    if (A->val != NULL) {
        for (i = 0; i < n; ++i) {
            ibegin = A->row_ptr[i], iend = A->row_ptr[i + 1];

            for (p = ibegin; p < iend; p++) {
                j = A->col_idx[p] + 1;
                k = AT.row_ptr[j];
                AT.col_idx[k] = i;
                AT.val[k] = A->val[p];
                AT.row_ptr[j] = k + 1;
            }
        }
    }
    else {
        for (i = 0; i < n; ++i) {
            ibegin = A->row_ptr[i], iend = A->row_ptr[i + 1];
            for (p = ibegin; p < iend; p++) {
                j = A->col_idx[p] + 1;
                k = AT.row_ptr[j];
                AT.col_idx[k] = i;
                AT.row_ptr[j] = k + 1;
            }
        }
    }

    return AT;
}


//Free CSR sparse matrix data memory space
void SSS_imat_destroy(SSS_IMAT *A)
{

    if (A == NULL) return;

    SSS_free(A->row_ptr);
    A->row_ptr = NULL;

    SSS_free(A->col_idx);
    A->col_idx = NULL;

    SSS_free(A->val);
    A->val = NULL;
}

// Find transpose of SSS_MAT matrix A
SSS_MAT SSS_mat_trans(SSS_MAT *A)
{

    const int n = A->num_rows, m = A->num_cols, nnz = A->num_nnzs;
    int i, j, k, p;
    SSS_MAT AT;

    AT.num_rows = m;
    AT.num_cols = n;
    AT.num_nnzs = nnz;

    AT.row_ptr = (int *) SSS_calloc(m + 1, sizeof(int));
    AT.col_idx = (int *) SSS_calloc(nnz, sizeof(int));

    if (A->val) {
        AT.val = (double *) SSS_calloc(nnz, sizeof(double));
    }
    else {
        AT.val = NULL;
    }

    // SSS_iarray_set(m+1, AT.row_ptr, 0);
    memset(AT.row_ptr, 0, sizeof(int) * (m + 1));

    for (j = 0; j < nnz; ++j) {
        i = A->col_idx[j];
        if (i < m - 1) AT.row_ptr[i + 2]++;
    }

    for (i = 2; i <= m; ++i) AT.row_ptr[i] += AT.row_ptr[i - 1];

    // second pass: form A'
    if (A->val) {
        for (i = 0; i < n; ++i) {
            int ibegin = A->row_ptr[i], iend = A->row_ptr[i + 1];
            for (p = ibegin; p < iend; p++) {
                j = A->col_idx[p] + 1;
                k = AT.row_ptr[j];
                AT.col_idx[k] = i;
                AT.val[k] = A->val[p];
                AT.row_ptr[j] = k + 1;
            }
        }
    }
    else {
        for (i = 0; i < n; ++i) {
            int ibegin = A->row_ptr[i], iend1 = A->row_ptr[i + 1];
            for (p = ibegin; p < iend1; p++) {
                j = A->col_idx[p] + 1;
                k = AT.row_ptr[j];
                AT.col_idx[k] = i;
                AT.row_ptr[j] = k + 1;
            }
        }
    }

    return AT;
}

void SSS_vec_cp(const SSS_VEC *x, SSS_VEC *y)
{
    assert(x->n > 0);

    y->n = x->n;
    memcpy(y->d, x->d, x->n * sizeof(double));
}


SSS_MAT SSS_blas_mat_rap(const SSS_MAT *R, const SSS_MAT *A, const SSS_MAT *P)
{
    int n_coarse = R->num_rows;
    int *R_i = R->row_ptr;
    int *R_j = R->col_idx;
    double *R_data = R->val;

    int n_fine = A->num_rows;
    int *A_i = A->row_ptr;
    int *A_j = A->col_idx;
    double *A_data = A->val;

    int *P_i = P->row_ptr;
    int *P_j = P->col_idx;
    double *P_data = P->val;

    int RAp_size;
    int *RAp_p = NULL;
    int *RAp_j = NULL;
    double *RAp_x = NULL;

    int *Ps_marker = NULL;
    int *As_marker = NULL;

    int ic, i1, i2, i3, jj1, jj2, jj3;
    int jj_cnter, jj_row_begining;
    double r_entry, r_a_product, r_a_p_product;

    int coarse_mul = n_coarse;
    int fine_mul = n_fine;
    int coarse_add = n_coarse + 1;
    int minus_one_length = coarse_mul + fine_mul;
    int total_calloc = minus_one_length + coarse_add + 1;
    SSS_MAT RAp;

    Ps_marker = (int *) SSS_calloc(total_calloc, sizeof(int));
    As_marker = Ps_marker + coarse_mul;

    /*------------------------------------------------------*
     *  First Pass: Determine size of RAp and set up RAp_p  *
     *------------------------------------------------------*/
    RAp_p = (int *) SSS_calloc(n_coarse + 1, sizeof(int));

    SSS_iarray_set(minus_one_length, Ps_marker, -1);

    jj_cnter = 0;
    for (ic = 0; ic < n_coarse; ic++) {
        Ps_marker[ic] = jj_cnter;
        jj_row_begining = jj_cnter;
        jj_cnter++;

        for (jj1 = R_i[ic]; jj1 < R_i[ic + 1]; jj1++) {
            i1 = R_j[jj1];

            for (jj2 = A_i[i1]; jj2 < A_i[i1 + 1]; jj2++) {
                i2 = A_j[jj2];
                if (As_marker[i2] != ic) {
                    As_marker[i2] = ic;
                    for (jj3 = P_i[i2]; jj3 < P_i[i2 + 1]; jj3++) {
                        i3 = P_j[jj3];
                        if (Ps_marker[i3] < jj_row_begining) {
                            Ps_marker[i3] = jj_cnter;
                            jj_cnter++;
                        }
                    }
                }
            }
        }

        RAp_p[ic] = jj_row_begining;
    }

    RAp_p[n_coarse] = jj_cnter;
    RAp_size = jj_cnter;

    RAp_j = (int *) SSS_calloc(RAp_size, sizeof(int));
    RAp_x = (double *) SSS_calloc(RAp_size, sizeof(double));

    SSS_iarray_set(minus_one_length, Ps_marker, -1);

    jj_cnter = 0;
    for (ic = 0; ic < n_coarse; ic++) {
        Ps_marker[ic] = jj_cnter;
        jj_row_begining = jj_cnter;
        RAp_j[jj_cnter] = ic;
        RAp_x[jj_cnter] = 0.0;
        jj_cnter++;

        for (jj1 = R_i[ic]; jj1 < R_i[ic + 1]; jj1++) {
            r_entry = R_data[jj1];

            i1 = R_j[jj1];
            for (jj2 = A_i[i1]; jj2 < A_i[i1 + 1]; jj2++) {
                r_a_product = r_entry * A_data[jj2];

                i2 = A_j[jj2];
                if (As_marker[i2] != ic) {
                    As_marker[i2] = ic;

                    for (jj3 = P_i[i2]; jj3 < P_i[i2 + 1]; jj3++) {
                        r_a_p_product = r_a_product * P_data[jj3];

                        i3 = P_j[jj3];
                        if (Ps_marker[i3] < jj_row_begining) {
                            Ps_marker[i3] = jj_cnter;
                            RAp_x[jj_cnter] = r_a_p_product;
                            RAp_j[jj_cnter] = i3;
                            jj_cnter++;
                        }
                        else {
                            RAp_x[Ps_marker[i3]] += r_a_p_product;
                        }
                    }
                }
                else {
                    for (jj3 = P_i[i2]; jj3 < P_i[i2 + 1]; jj3++) {
                        i3 = P_j[jj3];

                        r_a_p_product = r_a_product * P_data[jj3];
                        RAp_x[Ps_marker[i3]] += r_a_p_product;
                    }
                }
            }
        }
    }

    RAp.num_rows = n_coarse;
    RAp.num_cols = n_coarse;
    RAp.num_nnzs = RAp_size;
    RAp.row_ptr = RAp_p;
    RAp.col_idx = RAp_j;
    RAp.val = RAp_x;

    SSS_free(Ps_marker);

    return RAp;
}


void * SSS_realloc(void *oldmem, size_t tsize)
{
    void *mem = NULL;

    if (tsize > 0) {
        mem = realloc(oldmem, tsize);
    }

    if (mem == NULL) {
        printf("### WARNING: Cannot allocate %.3lfMB RAM!\n",
               (double) tsize / 1048576);
    }

    return mem;
}