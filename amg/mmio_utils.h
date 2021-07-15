#ifndef _TRANS_
#define _TRANS_

void exclusive_scan(int *input, int length)
{
    if(length == 0 || length == 1)
        return;

    int old_Ax, new_Ax;

    old_Ax = input[0];
    input[0] = 0;
    for (int i = 1; i < length; i++)
    {
        new_Ax = input[i];
        input[i] = old_Ax + input[i-1];
        old_Ax = new_Ax;
    }
}

void matrix_transposition(const int         m,
                          const int         n,
                          const int         nnz,
                          const int        *csrRowPtr,
                          const int        *csrColIdx,
                          const float *csrAx,
                                int        *cscRowIdx,
                                int        *cscColPtr,
                                float *cscAx)
{
    // histogram in column pointer
    memset (cscColPtr, 0, sizeof(int) * (n+1));
    for (int i = 0; i < nnz; i++)
    {
        cscColPtr[csrColIdx[i]]++;
    }

    // prefix-sum scan to get the column pointer
    exclusive_scan(cscColPtr, n + 1);

    int *cscColIncr = (int *)malloc(sizeof(int) * (n+1));
    memcpy (cscColIncr, cscColPtr, sizeof(int) * (n+1));

    // insert nnz to csc
    for (int row = 0; row < m; row++)
    {
        for (int j = csrRowPtr[row]; j < csrRowPtr[row+1]; j++)
        {
            int col = csrColIdx[j];

            cscRowIdx[cscColIncr[col]] = row;
            cscAx[cscColIncr[col]] = csrAx[j];
            cscColIncr[col]++;
        }
    }

    free (cscColIncr);
}

#endif
