#ifndef _MMIO_HIGHLEVEL_
#define _MMIO_HIGHLEVEL_

#include "mmio.h"
#include "mmio_utils.h"
#define MAT_PTR_TYPE int
#define MAT_Ax_TYPE double

// read matrix infomation from mtx file
int mmio_info(int *m, int *n, MAT_PTR_TYPE *nnz, int *isSymmetric, char *filename)
{
    int m_tmp, n_tmp;
    MAT_PTR_TYPE nnz_tmp;

    int ret_code;
    MM_typecode matcode;
    FILE *f;

    MAT_PTR_TYPE nnz_mtx_report;
    int isInteger = 0, isReal = 0, isPattern = 0, isSymmetric_tmp = 0, isComplex = 0;

    // load matrix
    if ((f = fopen(filename, "r")) == NULL)
        return -1;

    if (mm_read_banner(f, &matcode) != 0)
    {
        printf("Could not process Matrix Market banner.\n");
        return -2;
    }

    if ( mm_is_pattern( matcode ) )  { isPattern = 1; /*printf("type = Pattern\n");*/ }
    if ( mm_is_real ( matcode) )     { isReal = 1; /*printf("type = real\n");*/ }
    if ( mm_is_complex( matcode ) )  { isComplex = 1; /*printf("type = real\n");*/ }
    if ( mm_is_integer ( matcode ) ) { isInteger = 1; /*printf("type = integer\n");*/ }

    /* find out size of sparse matrix .... */
    ret_code = mm_read_mtx_crd_size(f, &m_tmp, &n_tmp, &nnz_mtx_report);
    if (ret_code != 0)
        return -4;

    if ( mm_is_symmetric( matcode ) || mm_is_hermitian( matcode ) )
    {
        isSymmetric_tmp = 1;
        //printf("input matrix is symmetric = true\n");
    }
    else
    {
        //printf("input matrix is symmetric = false\n");
    }

    MAT_PTR_TYPE *csrRowPtr_counter = (MAT_PTR_TYPE *)malloc((m_tmp+1) * sizeof(MAT_PTR_TYPE));
    memset(csrRowPtr_counter, 0, (m_tmp+1) * sizeof(MAT_PTR_TYPE));

    int *csrRowIdx_tmp = (int *)malloc(nnz_mtx_report * sizeof(int));
    int *csrColIdx_tmp = (int *)malloc(nnz_mtx_report * sizeof(int));
    MAT_Ax_TYPE *csrAx_tmp    = (MAT_Ax_TYPE *)malloc(nnz_mtx_report * sizeof(MAT_Ax_TYPE));
    MAT_Ax_TYPE *csrAx_im_tmp    = (MAT_Ax_TYPE *)malloc(nnz_mtx_report * sizeof(MAT_Ax_TYPE));


    /* NOTE: when reading in doubles, ANSI C requires the use of the "l"  */
    /*   specifier as in "%lg", "%lf", "%le", otherwise errors will occur */
    /*  (ANSI C X3.159-1989, Sec. 4.9.6.2, p. 136 lines 13-15)            */

    for (MAT_PTR_TYPE i = 0; i < nnz_mtx_report; i++)
    {
        int idxi, idxj;
        double fAx, fAx_im;
        int iAx;
        int returnvalue;

        if (isReal)
        {
            returnvalue = fscanf(f, "%d %d %lg\n", &idxi, &idxj, &fAx);
        }
        else if (isComplex)
        {
            returnvalue = fscanf(f, "%d %d %lg %lg\n", &idxi, &idxj, &fAx, &fAx_im);
        }
        else if (isInteger)
        {
            returnvalue = fscanf(f, "%d %d %d\n", &idxi, &idxj, &iAx);
            fAx = iAx;
        }
        else if (isPattern)
        {
            returnvalue = fscanf(f, "%d %d\n", &idxi, &idxj);
            fAx = 1.0;
        }

        // adjust from 1-based to 0-based
        idxi--;
        idxj--;

        csrRowPtr_counter[idxi]++;
        csrRowIdx_tmp[i] = idxi;
        csrColIdx_tmp[i] = idxj;
        csrAx_tmp[i] = fAx;
        csrAx_im_tmp[i] = fAx_im;
    }

    if (f != stdin)
        fclose(f);

    if (isSymmetric_tmp)
    {
        for (MAT_PTR_TYPE i = 0; i < nnz_mtx_report; i++)
        {
            if (csrRowIdx_tmp[i] != csrColIdx_tmp[i])
                csrRowPtr_counter[csrColIdx_tmp[i]]++;
        }
    }

    // exclusive scan for csrRowPtr_counter
    MAT_PTR_TYPE old_Ax, new_Ax;

    old_Ax = csrRowPtr_counter[0];
    csrRowPtr_counter[0] = 0;
    for (int i = 1; i <= m_tmp; i++)
    {
        new_Ax = csrRowPtr_counter[i];
        csrRowPtr_counter[i] = old_Ax + csrRowPtr_counter[i-1];
        old_Ax = new_Ax;
    }

    nnz_tmp = csrRowPtr_counter[m_tmp];

    *m = m_tmp;
    *n = n_tmp;
    *nnz = nnz_tmp;
    *isSymmetric = isSymmetric_tmp;

    // free tmp space
    free(csrColIdx_tmp);
    free(csrAx_tmp);
    free(csrAx_im_tmp);
    free(csrRowIdx_tmp);
    free(csrRowPtr_counter);

    return 0;
}

// read matrix infomation from mtx file
int mmio_data(MAT_PTR_TYPE *csrRowPtr, int *csrColIdx, MAT_Ax_TYPE *csrAx, char *filename)
{
    int m_tmp, n_tmp;
    MAT_PTR_TYPE nnz_tmp;

    int ret_code;
    MM_typecode matcode;
    FILE *f;

    MAT_PTR_TYPE nnz_mtx_report;
    int isInteger = 0, isReal = 0, isPattern = 0, isSymmetric_tmp = 0, isComplex = 0;

    // load matrix
    if ((f = fopen(filename, "r")) == NULL)
        return -1;

    if (mm_read_banner(f, &matcode) != 0)
    {
        printf("Could not process Matrix Market banner.\n");
        return -2;
    }

    if ( mm_is_pattern( matcode ) )  { isPattern = 1; /*printf("type = Pattern\n");*/ }
    if ( mm_is_real ( matcode) )     { isReal = 1; /*printf("type = real\n");*/ }
    if ( mm_is_complex( matcode ) )  { isComplex = 1; /*printf("type = real\n");*/ }
    if ( mm_is_integer ( matcode ) ) { isInteger = 1; /*printf("type = integer\n");*/ }

    /* find out size of sparse matrix .... */
    ret_code = mm_read_mtx_crd_size(f, &m_tmp, &n_tmp, &nnz_mtx_report);
    if (ret_code != 0)
        return -4;

    if ( mm_is_symmetric( matcode ) || mm_is_hermitian( matcode ) )
    {
        isSymmetric_tmp = 1;
        //printf("input matrix is symmetric = true\n");
    }
    else
    {
        //printf("input matrix is symmetric = false\n");
    }

    MAT_PTR_TYPE *csrRowPtr_counter = (MAT_PTR_TYPE *)malloc((m_tmp+1) * sizeof(MAT_PTR_TYPE));
    memset(csrRowPtr_counter, 0, (m_tmp+1) * sizeof(MAT_PTR_TYPE));

    int *csrRowIdx_tmp = (int *)malloc(nnz_mtx_report * sizeof(int));
    int *csrColIdx_tmp = (int *)malloc(nnz_mtx_report * sizeof(int));
    MAT_Ax_TYPE *csrAx_tmp    = (MAT_Ax_TYPE *)malloc(nnz_mtx_report * sizeof(MAT_Ax_TYPE));

    /* NOTE: when reading in doubles, ANSI C requires the use of the "l"  */
    /*   specifier as in "%lg", "%lf", "%le", otherwise errors will occur */
    /*  (ANSI C X3.159-1989, Sec. 4.9.6.2, p. 136 lines 13-15)            */

    for (MAT_PTR_TYPE i = 0; i < nnz_mtx_report; i++)
    {
        int idxi, idxj;
        double fAx, fAx_im;
        int iAx;
        int returnvalue;

        if (isReal)
        {
            returnvalue = fscanf(f, "%d %d %lg\n", &idxi, &idxj, &fAx);
        }
        else if (isComplex)
        {
            returnvalue = fscanf(f, "%d %d %lg %lg\n", &idxi, &idxj, &fAx, &fAx_im);
        }
        else if (isInteger)
        {
            returnvalue = fscanf(f, "%d %d %d\n", &idxi, &idxj, &iAx);
            fAx = iAx;
        }
        else if (isPattern)
        {
            returnvalue = fscanf(f, "%d %d\n", &idxi, &idxj);
            fAx = 1.0;
        }

        // adjust from 1-based to 0-based
        idxi--;
        idxj--;
        
        csrRowPtr_counter[idxi]++;
        csrRowIdx_tmp[i] = idxi;
        csrColIdx_tmp[i] = idxj;
        csrAx_tmp[i] = fAx;
    }

    if (f != stdin)
        fclose(f);

    if (isSymmetric_tmp)
    {
        for (MAT_PTR_TYPE i = 0; i < nnz_mtx_report; i++)
        {
            if (csrRowIdx_tmp[i] != csrColIdx_tmp[i])
                csrRowPtr_counter[csrColIdx_tmp[i]]++;
        }
    }

    // exclusive scan for csrRowPtr_counter
    /*MAT_PTR_TYPE old_Ax, new_Ax;

    old_Ax = csrRowPtr_counter[0];
    csrRowPtr_counter[0] = 0;
    for (int i = 1; i <= m_tmp; i++)
    {
        new_Ax = csrRowPtr_counter[i];
        csrRowPtr_counter[i] = old_Ax + csrRowPtr_counter[i-1];
        old_Ax = new_Ax;
    }*/
    exclusive_scan(csrRowPtr_counter, m_tmp+1);

    nnz_tmp = csrRowPtr_counter[m_tmp];
    memcpy(csrRowPtr, csrRowPtr_counter, (m_tmp+1) * sizeof(MAT_PTR_TYPE));
    memset(csrRowPtr_counter, 0, (m_tmp+1) * sizeof(MAT_PTR_TYPE));

    if (isSymmetric_tmp)
    {
        for (MAT_PTR_TYPE i = 0; i < nnz_mtx_report; i++)
        {
            if (csrRowIdx_tmp[i] != csrColIdx_tmp[i])
            {
                MAT_PTR_TYPE offset = csrRowPtr[csrRowIdx_tmp[i]] + csrRowPtr_counter[csrRowIdx_tmp[i]];
                csrColIdx[offset] = csrColIdx_tmp[i];
                csrAx[offset] = csrAx_tmp[i];
                csrRowPtr_counter[csrRowIdx_tmp[i]]++;

                offset = csrRowPtr[csrColIdx_tmp[i]] + csrRowPtr_counter[csrColIdx_tmp[i]];
                csrColIdx[offset] = csrRowIdx_tmp[i];
                csrAx[offset] = csrAx_tmp[i];
                csrRowPtr_counter[csrColIdx_tmp[i]]++;
            }
            else
            {
                MAT_PTR_TYPE offset = csrRowPtr[csrRowIdx_tmp[i]] + csrRowPtr_counter[csrRowIdx_tmp[i]];
                csrColIdx[offset] = csrColIdx_tmp[i];
                csrAx[offset] = csrAx_tmp[i];
                csrRowPtr_counter[csrRowIdx_tmp[i]]++;
            }
        }
    }
    else
    {
        for (MAT_PTR_TYPE i = 0; i < nnz_mtx_report; i++)
        {
            MAT_PTR_TYPE offset = csrRowPtr[csrRowIdx_tmp[i]] + csrRowPtr_counter[csrRowIdx_tmp[i]];
            csrColIdx[offset] = csrColIdx_tmp[i];
            csrAx[offset] = csrAx_tmp[i];
            csrRowPtr_counter[csrRowIdx_tmp[i]]++;
        }
    }

    // free tmp space
    free(csrColIdx_tmp);
    free(csrAx_tmp);
    free(csrRowIdx_tmp);
    free(csrRowPtr_counter);

    return 0;
}

// read matrix infomation from mtx file
int mmio_allinone(int *m, int *n, MAT_PTR_TYPE *nnz, int *isSymmetric, 
                  MAT_PTR_TYPE **csrRowPtr, int **csrColIdx, MAT_Ax_TYPE **csrAx, 
                  char *filename)
{
    int m_tmp, n_tmp;
    MAT_PTR_TYPE nnz_tmp;

    int ret_code;
    MM_typecode matcode;
    FILE *f;

    MAT_PTR_TYPE nnz_mtx_report;
    int isInteger = 0, isReal = 0, isPattern = 0, isSymmetric_tmp = 0, isComplex = 0;

    // load matrix
    if ((f = fopen(filename, "r")) == NULL)
        return -1;

    if (mm_read_banner(f, &matcode) != 0)
    {
        printf("Could not process Matrix Market banner.\n");
        return -2;
    }

    if ( mm_is_pattern( matcode ) )  { isPattern = 1; /*printf("type = Pattern\n");*/ }
    if ( mm_is_real ( matcode) )     { isReal = 1; /*printf("type = real\n");*/ }
    if ( mm_is_complex( matcode ) )  { isComplex = 1; /*printf("type = real\n");*/ }
    if ( mm_is_integer ( matcode ) ) { isInteger = 1; /*printf("type = integer\n");*/ }

    /* find out size of sparse matrix .... */
    ret_code = mm_read_mtx_crd_size(f, &m_tmp, &n_tmp, &nnz_mtx_report);
    if (ret_code != 0)
        return -4;

    if ( mm_is_symmetric( matcode ) || mm_is_hermitian( matcode ) )
    {
        isSymmetric_tmp = 1;
        //printf("input matrix is symmetric = true\n");
    }
    else
    {
        //printf("input matrix is symmetric = false\n");
    }

    MAT_PTR_TYPE *csrRowPtr_counter = (MAT_PTR_TYPE *)malloc((m_tmp+1) * sizeof(MAT_PTR_TYPE));
    memset(csrRowPtr_counter, 0, (m_tmp+1) * sizeof(MAT_PTR_TYPE));

    int *csrRowIdx_tmp = (int *)malloc(nnz_mtx_report * sizeof(int));
    int *csrColIdx_tmp = (int *)malloc(nnz_mtx_report * sizeof(int));
    MAT_Ax_TYPE *csrAx_tmp    = (MAT_Ax_TYPE *)malloc(nnz_mtx_report * sizeof(MAT_Ax_TYPE));

    /* NOTE: when reading in doubles, ANSI C requires the use of the "l"  */
    /*   specifier as in "%lg", "%lf", "%le", otherwise errors will occur */
    /*  (ANSI C X3.159-1989, Sec. 4.9.6.2, p. 136 lines 13-15)            */

    for (MAT_PTR_TYPE i = 0; i < nnz_mtx_report; i++)
    {
        int idxi, idxj;
        double fAx, fAx_im;
        int iAx;
        int returnvalue;

        if (isReal)
        {
            returnvalue = fscanf(f, "%d %d %lg\n", &idxi, &idxj, &fAx);
        }
        else if (isComplex)
        {
            returnvalue = fscanf(f, "%d %d %lg %lg\n", &idxi, &idxj, &fAx, &fAx_im);
        }
        else if (isInteger)
        {
            returnvalue = fscanf(f, "%d %d %d\n", &idxi, &idxj, &iAx);
            fAx = iAx;
        }
        else if (isPattern)
        {
            returnvalue = fscanf(f, "%d %d\n", &idxi, &idxj);
            fAx = 1.0;
        }

        // adjust from 1-based to 0-based
        idxi--;
        idxj--;
        
        csrRowPtr_counter[idxi]++;
        csrRowIdx_tmp[i] = idxi;
        csrColIdx_tmp[i] = idxj;
        csrAx_tmp[i] = fAx;
    }

    if (f != stdin)
        fclose(f);

    if (isSymmetric_tmp)
    {
        for (MAT_PTR_TYPE i = 0; i < nnz_mtx_report; i++)
        {
            if (csrRowIdx_tmp[i] != csrColIdx_tmp[i])
                csrRowPtr_counter[csrColIdx_tmp[i]]++;
        }
    }

    // exclusive scan for csrRowPtr_counter
    exclusive_scan(csrRowPtr_counter, m_tmp+1);

    MAT_PTR_TYPE *csrRowPtr_alias = (MAT_PTR_TYPE *)malloc((m_tmp+1) * sizeof(MAT_PTR_TYPE));
    nnz_tmp = csrRowPtr_counter[m_tmp];
    int *csrColIdx_alias = (int *)malloc(nnz_tmp * sizeof(int));
    MAT_Ax_TYPE *csrAx_alias    = (MAT_Ax_TYPE *)malloc(nnz_tmp * sizeof(MAT_Ax_TYPE));

    memcpy(csrRowPtr_alias, csrRowPtr_counter, (m_tmp+1) * sizeof(MAT_PTR_TYPE));
    memset(csrRowPtr_counter, 0, (m_tmp+1) * sizeof(MAT_PTR_TYPE));

    if (isSymmetric_tmp)
    {
        for (MAT_PTR_TYPE i = 0; i < nnz_mtx_report; i++)
        {
            if (csrRowIdx_tmp[i] != csrColIdx_tmp[i])
            {
                MAT_PTR_TYPE offset = csrRowPtr_alias[csrRowIdx_tmp[i]] + csrRowPtr_counter[csrRowIdx_tmp[i]];
                csrColIdx_alias[offset] = csrColIdx_tmp[i];
                csrAx_alias[offset] = csrAx_tmp[i];
                csrRowPtr_counter[csrRowIdx_tmp[i]]++;

                offset = csrRowPtr_alias[csrColIdx_tmp[i]] + csrRowPtr_counter[csrColIdx_tmp[i]];
                csrColIdx_alias[offset] = csrRowIdx_tmp[i];
                csrAx_alias[offset] = csrAx_tmp[i];
                csrRowPtr_counter[csrColIdx_tmp[i]]++;
            }
            else
            {
                MAT_PTR_TYPE offset = csrRowPtr_alias[csrRowIdx_tmp[i]] + csrRowPtr_counter[csrRowIdx_tmp[i]];
                csrColIdx_alias[offset] = csrColIdx_tmp[i];
                csrAx_alias[offset] = csrAx_tmp[i];
                csrRowPtr_counter[csrRowIdx_tmp[i]]++;
            }
        }
    }
    else
    {
        for (MAT_PTR_TYPE i = 0; i < nnz_mtx_report; i++)
        {            
            MAT_PTR_TYPE offset = csrRowPtr_alias[csrRowIdx_tmp[i]] + csrRowPtr_counter[csrRowIdx_tmp[i]];
            csrColIdx_alias[offset] = csrColIdx_tmp[i];
            csrAx_alias[offset] = csrAx_tmp[i];
            csrRowPtr_counter[csrRowIdx_tmp[i]]++;
        }
    }
    
    *m = m_tmp;
    *n = n_tmp;
    *nnz = nnz_tmp;
    *isSymmetric = isSymmetric_tmp;

    *csrRowPtr = csrRowPtr_alias;
    *csrColIdx = csrColIdx_alias;
    *csrAx = csrAx_alias;

    // free tmp space
    free(csrColIdx_tmp);
    free(csrAx_tmp);
    free(csrRowIdx_tmp);
    free(csrRowPtr_counter);

    return 0;
}


#endif
