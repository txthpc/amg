
#include "SSS_SETUP.h"
#include "SSS_inter.h"

void SSS_amg_complexity_print(SSS_AMG *mg)
{

    const int max_levels = mg->num_levels;
    int level;
    double gridcom = 0.0, opcom = 0.0;

    printf("-----------------------------------------------------------\n");
    printf("  Level   Num of rows   Num of nonzeros   Avg. NNZ / row   \n");
    printf("-----------------------------------------------------------\n");

    for (level = 0; level < max_levels; ++level) {
        double AvgNNZ = (double) mg->cg[level].A.num_nnzs / mg->cg[level].A.num_rows;

        printf("%5d %13d %17d %14.2lf\n", level, mg->cg[level].A.num_rows,
                mg->cg[level].A.num_nnzs, AvgNNZ);

        gridcom += mg->cg[level].A.num_rows;
        opcom += mg->cg[level].A.num_nnzs;
    }

    printf("-----------------------------------------------------------\n");

    gridcom /= mg->cg[0].A.num_rows;
    opcom /= mg->cg[0].A.num_nnzs;
    printf("  Grid complexity = %.3lf  |", gridcom);
    printf("  Operator complexity = %.3lf\n", opcom);

    printf("-----------------------------------------------------------\n");
}

void SSS_amg_setup(SSS_AMG *mg, SSS_MAT *A, SSS_AMG_PARS *pars)
{
    int min_cdof; //最小粗变量数

    int m;
    int status = 0;
    int lvl = 0, max_lvls;
    double setup_start, setup_end;
    SSS_IMAT S_MAT;   //强连接矩阵
    int size;
    SSS_IVEC vertices;

    min_cdof = SSS_max(pars->coarse_dof, MIN_CDOF);  //最小粗变量数
    max_lvls = pars->max_levels;

    /* timer */
    setup_start = SSS_get_time();

    /* create mg */
    assert(mg != NULL);
    *mg = SSS_amg_data_create(pars);


    /* init */
    m = A->num_rows;
    vertices = SSS_ivec_create(m);

    /* init matrix */
    mg->cg[0].A = SSS_mat_struct_create(m, m, A->num_nnzs);
    SSS_mat_cp(A, &mg->cg[0].A);

    // Main AMG setup loop
    //当前层粗网格的行数 > 最小粗节点数  或  当前层 < 最大层数 -1
    while ((mg->cg[lvl].A.num_rows > min_cdof) && (lvl < max_lvls - 1)) {
        /* init */
        bzero(&S_MAT, sizeof(S_MAT));

        /*-- Coarsening and form the structure of interpolation --*/                                //粗化，形成插值结构
        status = SSS_amg_coarsen(&mg->cg[lvl].A, &vertices, &mg->cg[lvl].P, &S_MAT, pars);     //status为0代表粗化成功


         // Check 1: Did coarsening step succeeded?
        if (status < 0) {
            /*-- Clean up S_MAT generated in coarsening --*/
            SSS_free(S_MAT.row_ptr);
            SSS_free(S_MAT.col_idx);

            // When error happens, stop at the current multigrid level!
                printf("### WARNING: Could not find any C-variables!\n");
                printf("### WARNING: RS coarsening on level-%d failed!\n", lvl);
            
            status = 0;
            break;
        }

        // Check 2: Is coarse sparse too small?
        if (mg->cg[lvl].P.num_cols < min_cdof) {
            /*-- Clean up S_MAT generated in coarsening --*/
            SSS_free(S_MAT.row_ptr);
            SSS_free(S_MAT.col_idx);

            break;
        }

        // Check 3: Does this coarsening step too aggressive?
        if (mg->cg[lvl].P.num_rows > mg->cg[lvl].P.num_cols * 10) {
                printf("### WARNING: Coarsening might be too aggressive!\n");
                printf("### WARNING: Lvl = %d ,Fine level = %d, coarse level = %d. Discard!\n",
                     lvl,mg->cg[lvl].P.num_rows, mg->cg[lvl].P.num_cols);
            
        }

    
        /*-- Perform aggressive coarsening only up to the specified level --*/
        if (mg->cg[lvl].P.num_cols * 1.5 > mg->cg[lvl].A.num_rows) pars->cs_type = SSS_COARSE_RS;

        /*-- Store the C/F marker --*/
        size = mg->cg[lvl].A.num_rows;

        mg->cg[lvl].cfmark = SSS_ivec_create(size);                              //cfmark是一个A.num_rows的列向量
        memcpy(mg->cg[lvl].cfmark.d, vertices.d, size * sizeof(int));      //vertices是（0,1）的列向量（细，粗）


        /*-- Form interpolation --*/
        SSS_amg_interp(&mg->cg[lvl].A, &vertices, &mg->cg[lvl].P, &S_MAT, pars);
     //   printf("------------------end inter\n");

        /*-- Form coarse level matrix: two RAP routines available! --*/
        mg->cg[lvl].R = SSS_mat_trans(&mg->cg[lvl].P);
      //  printf("------------------end trans\n");


        struct timeval rap1, rap2;
        gettimeofday(&rap1,NULL);
        
        mg->cg[lvl + 1].A = SSS_blas_mat_rap(&mg->cg[lvl].R, &mg->cg[lvl].A, &mg->cg[lvl].P);
        
        gettimeofday(&rap2,NULL);
        double SSS_blas_mat_rap = (rap2.tv_sec - rap1.tv_sec) * 1000.0 + (rap2.tv_usec - rap1.tv_usec) / 1000.0;
     //   printf("-------------SSS_blas_mat_rap %f ms -------------------\n",SSS_blas_mat_rap);
        
        /*-- Clean up S_MAT generated in coarsening --*/
        SSS_free(S_MAT.row_ptr);
        SSS_free(S_MAT.col_idx);

        // Check 4: Is the coarse matrix too dense?
        if (mg->cg[lvl].A.num_nnzs / mg->cg[lvl].A.num_rows > mg->cg[lvl].A.num_cols * 0.2) {
                printf("### WARNING: Coarse matrix is too dense!\n");
                printf("### WARNING: m = n = %d, nnz = %d!\n", mg->cg[lvl].A.num_cols,
                        mg->cg[lvl].A.num_nnzs);
            

            /* free A */
            SSS_mat_destroy(&mg->cg[lvl + 1].A);

            break;
        }

        lvl++;
    }

    // setup total level number and current level
    mg->num_levels = max_lvls = lvl + 1;
    mg->cg[0].wp = SSS_vec_create(m);

    for (lvl = 1; lvl < max_lvls; ++lvl) {
        int mm = mg->cg[lvl].A.num_rows;

        mg->cg[lvl].b = SSS_vec_create(mm);
        mg->cg[lvl].x = SSS_vec_create(mm);

        // allocate work arrays for the solve phase
        mg->cg[lvl].wp = SSS_vec_create(2 * mm);
    }

    SSS_ivec_destroy(&vertices);

        setup_end = SSS_get_time();

        SSS_amg_complexity_print(mg);
        printf("AMG setup time: %g s\n", setup_end - setup_start);
    
}