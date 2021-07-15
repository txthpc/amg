#include "SSS_utils.h"

double SSS_get_time(void)
{
    struct timeval tv;
    double t;

    gettimeofday(&tv, (struct timezone *)0);
    t = tv.tv_sec + (double)tv.tv_usec * 1e-6;
 
    return t;
}


//Standard and aggressive coarsening schemes
void SSS_exit_on_errcode(const int status, const char *fctname)
{
    if (status >= 0) return;

    switch (status) {
        case ERROR_OPEN_FILE:
            printf("### ERROR: %s -- Cannot open file!\n", fctname);
            break;

        case ERROR_WRONG_FILE:
            printf("### ERROR: %s -- Wrong file format!\n", fctname);
            break;

        case ERROR_INPUT_PAR:
            printf("### ERROR: %s -- Wrong input arguments!\n", fctname);
            break;

        case ERROR_ALLOC_MEM:
            printf("### ERROR: %s -- Cannot allocate memory!\n", fctname);
            break;

        case ERROR_DATA_STRUCTURE:
            printf("### ERROR: %s -- Data structure mismatch!\n", fctname);
            break;

        case ERROR_DATA_ZERODIAG:
            printf("### ERROR: %s -- Matrix has zero diagonal entries!\n", fctname);
            break;

        case ERROR_DUMMY_VAR:
            printf("### ERROR: %s -- Unexpected input argument!\n", fctname);
            break;

        case ERROR_AMG_interp_type:
            printf("### ERROR: %s -- Unknown AMG interPolation type!\n", fctname);
            break;

        case ERROR_AMG_COARSE_TYPE:
            printf("### ERROR: %s -- Unknown AMG coarsening type!\n", fctname);
            break;

        case ERROR_AMG_SMOOTH_TYPE:
            printf("### ERROR: %s -- Unknown AMG smoother type!\n", fctname);
            break;

        case ERROR_SOLVER_STAG:
            printf("### ERROR: %s -- Solver stagnation error!\n", fctname);
            break;

        case ERROR_SOLVER_SOLSTAG:
            printf("### ERROR: %s -- Solution is close to zero!\n", fctname);
            break;

        case ERROR_SOLVER_TOLSMALL:
            printf("### ERROR: %s -- Tol is too small for the solver!\n", fctname);
            break;

        case ERROR_SOLVER_matrix:
            printf("### ERROR: %s -- max iteration number reached!\n", fctname);
            break;

        case ERROR_SOLVER_EXIT:
            printf("### ERROR: %s -- Solver exited unexpected!\n", fctname);
            break;

        case ERROR_MISC:
            printf("### ERROR: %s -- Unknown error occurred!\n", fctname);
            break;

        case ERROR_UNKNOWN:
            printf("### ERROR: %s -- Function does not exit successfully!\n", fctname);
            break;

        default:
            break;
    }

    exit(status);
}


void SSS_free(void *mem)
{
    if (mem) free(mem);
}

 
//打印出迭代解算器的迭代信息
void SSS_print_itinfo(const int stop_type, const int iter, const double relres,
        const double absres, const double factor)
{

        if (iter > 0) {
            printf("%6d | %13.6e   | %13.6e  | %10.4lf\n", iter, relres, absres,
                    factor);
        }
        else {                  // iter = 0: initial guess
            printf("-----------------------------------------------------------\n");

            switch (stop_type) {
                case STOP_REL_RES:
                    printf("It Num |   ||r||/||b||   |     ||r||      |  Conv. Factor\n");
                    break;

                case STOP_REL_PRECRES:
                    printf("It Num | ||r||_B/||b||_B |    ||r||_B     |  Conv. Factor\n");
                    break;

                case STOP_MOD_REL_RES:
                    printf("It Num |   ||r||/||x||   |     ||r||      |  Conv. Factor\n");
                    break;
            }

            printf("-----------------------------------------------------------\n");
            printf("%6d | %13.6e   | %13.6e  |     -.-- \n", iter, relres, absres);
        }
    
}



//计算向量b的L2范数  
double SSS_blas_vec_norm2(const SSS_VEC *x)
{
    double twonorm = 0;
    int i;
    int length = x->n;
    double *xpt = x->d;

    for (i = 0; i < length; ++i) twonorm += xpt[i] * xpt[i];

    return sqrt(twonorm);
}


double SSS_blas_array_norm2(int n, const double *x)
{
    int i;
    double twonorm = 0.;

    for (i = 0; i < n; ++i) twonorm += x[i] * x[i];

    return sqrt(twonorm);
}

void SSS_blas_mv_amxpy(double alpha, const SSS_MAT *A, const SSS_VEC *x, SSS_VEC *y)
{
    int m = A->num_rows;
    int *ia = A->row_ptr, *ja = A->col_idx;
    double *Aj = A->val;
    int i, k, begin_row, end_row;
    double temp;

    for (i = 0; i < m; ++i) {
        temp = 0.0;
        begin_row = ia[i];
        end_row = ia[i + 1];

        for (k = begin_row; k < end_row; ++k) temp += Aj[k] * x->d[ja[k]];

        y->d[i] += temp * alpha;
    }
}



void SSS_blas_mv_mxy(const SSS_MAT *A, const SSS_VEC *x, SSS_VEC *y)
{
    int m = A->num_rows;
    int *ia = A->row_ptr, *ja = A->col_idx;
    double *Aj = A->val;
    int i, k, begin_row, end_row;
    double temp;
    //printf("");
    for (i = 0; i < m; ++i) {
        temp = 0.0;
        begin_row = ia[i];
        end_row = ia[i + 1];

        for (k = begin_row; k < end_row; ++k) {
            temp += Aj[k] * x->d[ja[k]];
        }

        y->d[i] = temp;
    }
}



// dot
double SSS_blas_array_dot(int n, const double *x, const double *y)
{
    int i;
    double value = 0.0;

    for (i = 0; i < n; ++i) value += x[i] * y[i];
   // if(value > 100) printf("dot  = %lf\n",value);
    return value;
}

//稠密 y =a * x
void SSS_blas_array_axpy(int n, double a, const double *x, double *y)
{
    int i;

    for (i = 0; i < n; ++i) y[i] += a * x[i];
}


double SSS_blas_array_norminf(int n, const double *x)
{
    int i;
    double infnorm = 0.0;

    for (i = 0; i < n; ++i) infnorm = SSS_max(infnorm, SSS_ABS(x[i]));

    return infnorm;
}




//Set initial value for an array to be x=Ax
void SSS_blas_array_set( int n, double *x, double Ax)
{

    int i;

    for (i = 0; i < n; ++i) x[i] = Ax;
}

//稠密 y = ax+by
void SSS_blas_array_axpby(int n, double a, const double *x, double b, double *y)
{
    int i;

    for (i = 0; i < n; ++i) y[i] = a * x[i] + b * y[i];
}

void SSS_blas_array_ax(int n, double a, double *x)
{
    int i;

    for (i = 0; i < n; ++i) x[i] *= a;
}