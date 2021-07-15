#include "SSS_main.h"
#include "SSS_matvec.h"
#include "SSS_AMG.h"
#include "mmio.h"
#include "mmio_highlevel.h"
#include "mmio_utils.h"

//#include <cuda.h>


//读矩阵
void SSS_mat_read(char *filemat, SSS_MAT *A)
{
    int isSymmetric;
    printf("filename: %s\n",filemat);
   	mmio_info(&A->num_rows, &A->num_cols, &A->num_nnzs, &isSymmetric, filemat);
    A->row_ptr  = (int *)malloc((A->num_rows+1) * sizeof(int));
	A->col_idx  = (int *)malloc(A->num_nnzs * sizeof(int));
	A->val  = (double *)malloc(A->num_nnzs * sizeof(double));
    mmio_data(A->row_ptr, A->col_idx, A->val, filemat);   
    printf("A: m = %d, n = %d, nnz = %d\n", A->num_rows,A->num_cols, A->num_nnzs);
}

//初始化参数
void SSS_amg_pars_init(SSS_AMG_PARS *pars)
{

    //选择的光滑器
    pars->smoother = SSS_SM_GS;
    //AMG最大迭代次数
    pars->max_it = 100;
    //AMG收敛精度
    pars->tol = 1e-6;
    //最粗层收敛精度
    pars->ctol = 1e-7;
    //AMG最大层数
    pars->max_levels = 30;
    //最粗层的自由度
    pars->coarse_dof = MIN_CDOF;

    //循环类型 V-cycle
    pars->cycle_type = 1;
    //cf_order
    //0: nature order      1: C/F order
    pars->cf_order = 1;
    //前光滑次数
    pars->pre_iter = 2;
    //后光滑次数
    pars->post_iter = 2;
    //SOR光滑器的松弛解析器
    pars->relax = 1.0;
    //多项式平滑度 (Polynomial smoother)
    pars->poly_deg = 3;
    //粗化类型
    pars->cs_type = SSS_COARSE_RS;
    //插值类型
    pars->interp_type = intERP_DIR;
    //最大行和分析器
    pars->max_row_sum = 0.9;
    //粗化的强连接阈值
    pars->strong_threshold = 0.3;
    //截断阈值
    pars->trunc_threshold = 0.2;
}
 
//打印参数
void SSS_amg_pars_print(SSS_AMG_PARS *pars)
{
    assert(pars != NULL);
    
    printf("\n               AMG Parameters \n");
    printf("-----------------------------------------------------------\n");

    printf("AMG max num of iter:               %d\n", pars->max_it);
    printf("AMG tol:                           %g\n", pars->tol);
    printf("AMG ctol:                          %g\n", pars->ctol);
    printf("AMG max levels:                    %d\n", pars->max_levels);
    printf("AMG cycle type:                    %d\n", pars->cycle_type);
    printf("AMG smoother type:                 %d\n", pars->smoother);
    printf("AMG smoother order:                %d\n", pars->cf_order);
    printf("AMG num of presmoothing:           %d\n", pars->pre_iter);
    printf("AMG num of postsmoothing:          %d\n", pars->post_iter);

    switch(pars->smoother) {
        case SSS_SM_SOR:
        case SSS_SM_SSOR:
        case SSS_SM_GSOR:
        case SSS_SM_SGSOR:
            printf("AMG relax factor:                  %.4lf\n", pars->relax);
            break;

        case SSS_SM_POLY:
            printf("AMG polynomial smoother degree:    %d\n", pars->poly_deg);
            break;

        default:
            break;
    }

    printf("AMG coarsening type:               %d\n", pars->cs_type);
    //printf("AMG interPolation type:            %d\n", pars->interp_type);
    switch (pars->interp_type)
    {
    case 1:
        printf("AMG interPolation type:            Dir\n");
        break;
    case 2:
        printf("AMG interPolation type:            STD\n");
        break;
    default:
        break;
    }
    printf("AMG dof on coarsest grid:          %d\n", pars->coarse_dof);
    printf("AMG strong threshold:              %.4lf\n", pars->strong_threshold);
    printf("AMG truncation threshold:          %.4lf\n", pars->trunc_threshold);
    printf("AMG max row sum:                   %.4lf\n", pars->max_row_sum);

    printf("-----------------------------------------------------------\n");
}

int main(int argc,char *argv[])
{
    SSS_AMG_PARS pars;
    SSS_MAT A;
    SSS_VEC b, x;
    SSS_RTN rtn;

    /* 1.读矩阵 处理矩阵 */
    char *mat_file = argv[1];
    SSS_mat_read(mat_file,&A);
    
    /* pars */
    /* 2.初始化参数 */
    SSS_amg_pars_init(&pars);

    /* print info */
    /* 3.打印AMG参数 */
    SSS_amg_pars_print(&pars);

    /* 4.初始化向量b和x */
    b = SSS_vec_create(A.num_rows);
    SSS_vec_set_value(&b, 1.0);

    x = SSS_vec_create(A.num_rows);
    SSS_vec_set_value(&x, 1.0);
    
    /* solve the system */
    /* 5.正式进行AMG */

    rtn = SSS_solver_amg(&A, &x, &b, &pars);
    
    printf("AMG residual: %g\n", rtn.ares);
    printf("AMG relative residual: %g\n", rtn.rres);
    printf("AMG iterations: %d\n", rtn.nits);

    SSS_mat_destroy(&A);
    SSS_vec_destroy(&x);
    SSS_vec_destroy(&b);
    return 0;
}