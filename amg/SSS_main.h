#ifndef _SSS_MAIN_H_
#define _SSS_MAIN_H_
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <sys/time.h>
#include <assert.h>
#include <omp.h>
//#include <cuda.h>

#define TRUE       1
#define FALSE      0

#define max_AMG_LVL            30  /**< maximal AMG coarsening level */
#define max_STAG               20  /**< maximal number of stagnation times */
#define max_RESTART            30  /**< maximal restarting number */
#define BIGFLOAT             1e+20 /**< A large real number */
#define SMALLFLOAT2          1e-40 /**< An extremely small real number */

#define LIST_HEAD -1 /**< head of the linked list */
#define LIST_TAIL -2 /**< tail of the linked list */
#define FGPT                    0  /**< Fine grid points  */
#define CGPT                    1  /**< Coarse grid points */

#define MIN_CDOF               10  /**< Minimal number of coarsest variables */
#define SSS_max(a, b) (((a)>(b))?(a):(b))   /**< bigger one in a and b */
#define SSS_MIN(a, b) (((a)<(b))?(a):(b))   /**< smaller one in a and b */
#define SSS_ABS(a)    (((a)>=0.0)?(a):-(a)) /**< absolute value of a */
#define ISPT                    2  /**< Isolated points */
#define UNPT                   -1  /**< Undetermined points */

#define SMALLFLOAT           1e-20 /**< A small real number */


typedef enum
{
    ERROR_OPEN_FILE          = -10,  /**< fail to open a file */
    ERROR_WRONG_FILE         = -11,  /**< input contains wrong format */
    ERROR_INPUT_PAR          = -12,  /**< wrong input argument */
    ERROR_MAT_SIZE           = -13,  /**< wrong problem size */
    ERROR_MISC               = -14,  /**< other error */

    ERROR_ALLOC_MEM          = -20,  /**< fail to allocate memory */
    ERROR_DATA_STRUCTURE     = -21,  /**< problem with data structures */
    ERROR_DATA_ZERODIAG      = -22,  /**< matrix has zero diagonal entries */
    ERROR_DUMMY_VAR          = -23,  /**< unexpected input data */

    ERROR_AMG_interp_type    = -30,  /**< unknown interpolation type */
    ERROR_AMG_SMOOTH_TYPE    = -31,  /**< unknown smoother type */
    ERROR_AMG_COARSE_TYPE    = -32,  /**< unknown coarsening type */
    ERROR_AMG_COARSEING      = -33,  /**< coarsening step failed to complete */

    ERROR_SOLVER_STAG        = -42,  /**< solver stagnates */
    ERROR_SOLVER_SOLSTAG     = -43,  /**< solver's solution is too small */
    ERROR_SOLVER_TOLSMALL    = -44,  /**< solver's tolerance is too small */
    ERROR_SOLVER_matrix       = -48,  /**< maximal iteration number exceeded */
    ERROR_SOLVER_EXIT        = -49,  /**< solver does not quit successfully */

    ERROR_UNKNOWN            = -99,  /**< an unknown error type */

} SSS_ERROR_CODE;

typedef struct linked_list
{
    //! data
    int data;

    //! starting of the list
    int head;

    //! ending of the list
    int tail;

    //! next node
    struct linked_list *next_node;

    //! previous node
    struct linked_list *prev_node;

} ListElement;


typedef ListElement *LinkList; /**< linked list */

typedef enum
{
    STOP_REL_RES        = 1,   /**< relative residual: ||r||/||r_0|| */
    STOP_REL_PRECRES    = 2,   /**< relative B-residual: ||r||_B/||b||_B */
    STOP_MOD_REL_RES    = 3,   /**< modified relative residual ||r||/||x|| */

} SSS_STOP_TYPE;

typedef struct SSS_MAT_
{
    int num_rows;
    int num_cols;
    int num_nnzs;

    int *row_ptr;
    int *col_idx;
    double *val;

} SSS_MAT;

typedef struct SSS_IMAT_
{
    int num_rows;
    int num_cols;
    int num_nnzs;

    int *row_ptr;
    int *col_idx;
    int *val;

} SSS_IMAT;

typedef struct SSS_VEC_
{
    int n;
    double *d;

} SSS_VEC;

typedef struct SSS_IVEC_
{
    int n;
    int *d;

} SSS_IVEC;

typedef enum SSS_SM_TYPE_
{
    SSS_SM_JACOBI    = 1,  /**< Jacobi smoother */
    SSS_SM_GS        = 2,  /**< Gauss-Seidel smoother */
    SSS_SM_SGS       = 3,  /**< Symmetric Gauss-Seidel smoother */
    SSS_SM_SOR       = 4,  /**< SOR smoother */
    SSS_SM_SSOR      = 5,  /**< SSOR smoother */
    SSS_SM_GSOR      = 6,  /**< GS + SOR smoother */
    SSS_SM_SGSOR     = 7,  /**< SGS + SSOR smoother */
    SSS_SM_POLY      = 8,  /**< Polynomial smoother */
    SSS_SM_L1DIAG    = 9,  /**< L1 norm diagonal scaling smoother */

} SSS_SM_TYPE;

typedef enum interp_type_
{
    intERP_DIR     = 1,  /**< Direct interpolation */
    intERP_STD     = 2,  /**< Standard interpolation */

} interp_type;

typedef struct SSS_RTN_
{
    double ares;     /* absolute residual */
    double rres;     /* relative residual */
    int nits;     /* number of iterations */

} SSS_RTN;


typedef enum SSS_COARSEN_TYPE_
{
    SSS_COARSE_RS      = 1,  /**< Classical */
    SSS_COARSE_RSP     = 2,  /**< Classical, with positive offdiags */

} SSS_COARSEN_TYPE;

typedef struct SSS_AMG_PARS_
{

    int cycle_type;            /** type of AMG cycle, 0, 1 is for V, others for W */
    double tol;                  /** stopping tolerance for AMG solver */
    double ctol;                 /** stopping tolerance for coarsest solver */
    int max_it;                /** max number of iterations of AMG */

    SSS_COARSEN_TYPE cs_type;     /** coarsening type */
    int max_levels;           /** max number of levels of AMG */
    int coarse_dof;           /** max number of coarsest level DOF */

    SSS_SM_TYPE smoother;         /** smoother type */
    double relax;                /** relax parseter for SOR smoother */
    int cf_order;             /** False: nature order True: C/F order */
    int pre_iter;             /** number of presmoothers */
    int post_iter;            /** number of postsmoothers */
    int poly_deg;             /** degree of the polynomial smoother */

    interp_type interp_type;  /** interpolation type */
    double strong_threshold;     /** strong connection threshold for coarsening */
    double max_row_sum;          /** maximal row sum parseter */
    double trunc_threshold;      /** truncation threshold */

} SSS_AMG_PARS;

typedef struct SSS_AMG_COMP_
{
    SSS_MAT A;                    /** pointer to the matrix at level level_num */
    SSS_MAT R;                    /** restriction operator at level level_num */
    SSS_MAT P;                    /** prolongation operator at level level_num */
    SSS_VEC b;                    /** pointer to the right-hand side at level level_num */
    SSS_VEC x;                    /** pointer to the iterative solution at level level_num */
    SSS_IVEC cfmark;              /** pointer to the CF marker at level level_num */

    SSS_VEC wp;                   /** cache work space */

} SSS_AMG_COMP;

typedef struct SSS_AMG
{
    int num_levels;           /** number of levels */   //当前是底基层
    SSS_AMG_COMP *cg;             /* coarser grids */   //当前层的粗网格  

    SSS_AMG_PARS pars;            /* AMG parameters */

    SSS_RTN rtn;                  /* return values */   //当前层的残差等参数

} SSS_AMG;


typedef struct SSS_SMTR_
{
    SSS_SM_TYPE smoother;

    SSS_MAT *A;
    SSS_VEC *b;
    SSS_VEC *x;

    double relax;
    int nsweeps;
    int istart;
    int iend;
    int istep;
    int ndeg;
    int cf_order;
    int *ordering;

} SSS_SMTR;


typedef struct SSS_KRYLOV_
{
    double tol;
    SSS_MAT *A;
    SSS_VEC *b;
    SSS_VEC *u;
    int restart;
    int matrix;
    int stop_type;

} SSS_KRYLOV;

#endif