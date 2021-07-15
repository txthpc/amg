#include "SSS_coarsen.h"


//Create an node using Item for its data field
static LinkList create_node(int Item)
{
    LinkList new_node_ptr;

    /* Allocate memory space for the new node.
     * return with error if no space available
     */
    new_node_ptr = (LinkList) SSS_calloc(1, sizeof(ListElement));
    new_node_ptr->data = Item;
    new_node_ptr->next_node = NULL;
    new_node_ptr->prev_node = NULL;
    new_node_ptr->head = LIST_TAIL;
    new_node_ptr->tail = LIST_HEAD;

    return (new_node_ptr);
}
//Places point in new list
static void enter_list(LinkList *head_ptr, LinkList *tail_ptr,
        int measure, int index, int *lists, int *where)
{
    LinkList head = *head_ptr;
    LinkList tail = *tail_ptr;
    LinkList list_ptr;
    LinkList new_ptr;
    int old_tail;

    list_ptr = head;

    if (head == NULL) { /* no lists exist yet */
        new_ptr = create_node(measure);
        new_ptr->head = index;
        new_ptr->tail = index;
        lists[index] = LIST_TAIL;
        where[index] = LIST_HEAD;
        head = new_ptr;
        tail = new_ptr;

        *head_ptr = head;
        *tail_ptr = tail;

        return;
    }
    else {
        do {
            if (measure > list_ptr->data) {
                new_ptr = create_node(measure);

                new_ptr->head = index;
                new_ptr->tail = index;

                lists[index] = LIST_TAIL;
                where[index] = LIST_HEAD;

                if (list_ptr->prev_node != NULL) {
                    new_ptr->prev_node = list_ptr->prev_node;
                    list_ptr->prev_node->next_node = new_ptr;
                    list_ptr->prev_node = new_ptr;
                    new_ptr->next_node = list_ptr;
                }
                else {
                    new_ptr->next_node = list_ptr;
                    list_ptr->prev_node = new_ptr;
                    new_ptr->prev_node = NULL;
                    head = new_ptr;
                }

                *head_ptr = head;
                *tail_ptr = tail;

                return;
            }
            else if (measure == list_ptr->data) {
                old_tail = list_ptr->tail;
                lists[old_tail] = index;
                where[index] = old_tail;
                lists[index] = LIST_TAIL;
                list_ptr->tail = index;

                return;
            }

            list_ptr = list_ptr->next_node;
        } while (list_ptr != NULL);

        new_ptr = create_node(measure);
        new_ptr->head = index;
        new_ptr->tail = index;
        lists[index] = LIST_TAIL;
        where[index] = LIST_HEAD;
        tail->next_node = new_ptr;
        new_ptr->prev_node = tail;
        new_ptr->next_node = NULL;
        tail = new_ptr;

        *head_ptr = head;
        *tail_ptr = tail;

        return;
    }
}
//Generate the set of all strong negative couplings    //生成所有强负耦合的集合
static void strong_couplings(SSS_MAT *A, SSS_IMAT *S, SSS_AMG_PARS *pars)
{
    const double max_row_sum = pars->max_row_sum;   //maximal row sum parseter          //最大行和分析器
    const double epsilon_str = pars->strong_threshold;                                  //强度阈值
    const int row = A->num_rows, col = A->num_cols, row1 = row + 1;
    const int nnz = A->num_nnzs;

    int *ia = A->row_ptr, *ja = A->col_idx;
    double *Aj = A->val;

    // local variables
    int i, j, begin_row, end_row;
    double row_scl, row_sum;

    // get the diagonal entry of A: assume all connections are strong       //假设所有的联系都strong
    SSS_VEC diag;

    //diag =SSS_mat_get_diag(A, 0);        //Get first n diagonal entries of a CSR matrix A    //得到一个CSR矩阵a的前n个对角线项,0则为全部对角项
    diag = SSS_mat_get_diag(A,0);
    // copy the structure of A to S
    S->num_rows = row;
    S->num_cols = col;
    S->num_nnzs = nnz;
    S->val = NULL;
    S->row_ptr = (int *) SSS_calloc(row1, sizeof(int));
    S->col_idx = (int *) SSS_calloc(nnz, sizeof(int));
    SSS_iarray_cp(row1, ia, S->row_ptr);
    SSS_iarray_cp(nnz, ja, S->col_idx);

    for (i = 0; i < row; ++i) {

        // Compute row scale and row sum
        row_scl = row_sum = 0.0;
        begin_row = ia[i];
        end_row = ia[i + 1];

        for (j = begin_row; j < end_row; j++) {

            // Originally: Not consider positive entries
            // row_sum += Aj[j];
            row_sum += SSS_ABS(Aj[j]);

            // Originally: Not consider positive entries
            // row_scl = max(row_scl, -Aj[j]); // smallest negative
            if (ja[j] != i)
                row_scl = SSS_max(row_scl, SSS_ABS(Aj[j]));   // largest abs

        }

        // Multiply by the strength threshold
        row_scl *= epsilon_str;

        // Find diagonal entries of S and remove them later
        for (j = begin_row; j < end_row; j++) {
            if (ja[j] == i) {
                S->col_idx[j] = -1;
                break;
            }
        }

        // Mark entire row as weak couplings if strongly diagonal-dominant
        // Originally: Not consider positive entries
        // if ( ABS(row_sum) > max_row_sum * ABS(diag.d[i]) ) {
        if (row_sum < (2 - max_row_sum) * SSS_ABS(diag.d[i])) {
            for (j = begin_row; j < end_row; j++)
                S->col_idx[j] = -1;
        }
        else {
            for (j = begin_row; j < end_row; j++) {
                if (-A->val[j] <= row_scl) S->col_idx[j] = -1;      // only n-couplings
            }
        }
    }

    SSS_vec_destroy(&diag);
} 


//Remove weak couplings from S (marked as -1)
static int compress_S(SSS_IMAT * S)
{
    const int row = S->num_rows;
    int *ia = S->row_ptr;

    // local variables
    int index, i, j, begin_row, end_row;

    // compress S: remove weak connections and form strong coupling matrix
    for (index = i = 0; i < row; ++i) {
        begin_row = ia[i];
        end_row = ia[i + 1];

        ia[i] = index;
        for (j = begin_row; j < end_row; j++) {
            if (S->col_idx[j] > -1) S->col_idx[index++] = S->col_idx[j];      // strong couplings
        }
    }

    S->num_nnzs = S->row_ptr[row] = index;

    if (S->num_nnzs <= 0) {
        return -99;//SSS_ERROR_CODE.ERROR_UNKNOWN
    }
    else {
        return 0;
    }
}

static void dispose_node(LinkList node_ptr)
{
    if (node_ptr) SSS_free(node_ptr);
}

//Removes a point from the lists
static void remove_node(LinkList * head_ptr, LinkList * tail_ptr,
        int measure, int index, int * lists, int * where)
{
    LinkList head = *head_ptr;
    LinkList tail = *tail_ptr;
    LinkList list_ptr = head;

    do {
        if (measure == list_ptr->data) {
            /* point to be removed is only point on list,
               which must be destroyed */
            if (list_ptr->head == index && list_ptr->tail == index) {
                /* removing only list, so num_left better be 0! */
                if (list_ptr == head && list_ptr == tail) {
                    head = NULL;
                    tail = NULL;
                    dispose_node(list_ptr);

                    *head_ptr = head;
                    *tail_ptr = tail;
                    return;
                }
                else if (head == list_ptr) {        /*removing 1st (max_measure) list */
                    list_ptr->next_node->prev_node = NULL;
                    head = list_ptr->next_node;
                    dispose_node(list_ptr);

                    *head_ptr = head;
                    *tail_ptr = tail;
                    return;
                }
                else if (tail == list_ptr) {        /* removing last list */
                    list_ptr->prev_node->next_node = NULL;
                    tail = list_ptr->prev_node;
                    dispose_node(list_ptr);

                    *head_ptr = head;
                    *tail_ptr = tail;
                    return;
                }
                else {
                    list_ptr->next_node->prev_node = list_ptr->prev_node;
                    list_ptr->prev_node->next_node = list_ptr->next_node;
                    dispose_node(list_ptr);

                    *head_ptr = head;
                    *tail_ptr = tail;
                    return;
                }
            }
            else if (list_ptr->head == index) { /* index is head of list */
                list_ptr->head = lists[index];
                where[lists[index]] = LIST_HEAD;
                return;
            }
            else if (list_ptr->tail == index) { /* index is tail of list */
                list_ptr->tail = where[index];
                lists[where[index]] = LIST_TAIL;
                return;
            }
            else {              /* index is in middle of list */
                lists[where[index]] = lists[index];
                where[lists[index]] = where[index];
                return;
            }
        }

        list_ptr = list_ptr->next_node;
    } while (list_ptr != NULL);

    printf("### ERROR: This list is empty! %s : %d\n", __FILE__, __LINE__);
    return;
}
// Find coarse level variables (classic C/F splitting)
static int cfsplitting_cls(SSS_MAT *A, SSS_IMAT *S, SSS_IVEC *vertices)
{
    const int row = A->num_rows;

    // local variables
    int col = 0;
    int maxmeas, maxnode, num_left = 0;
    int measure, newmeas;
    int i, j, k, l;
    int *vec = vertices->d;
    int *work = (int *) SSS_calloc(3 * row, sizeof(int));
    int *lists = work, *where = lists + row, *lambda = where + row;
    SSS_IMAT ST;

    int set_empty = 1;
    int jkeep = 0, cnt, index;
    int row_end_S, ji, row_end_S_nabor, jj;
    int *grAph_array = lambda;
    LinkList head = NULL, tail = NULL, list_ptr = NULL;

    // 0. Compress S and form S_transpose
    col = compress_S(S);
    if (col < 0) goto eofc;          // compression failed!!!

    ST = SSS_imat_trans(S);

    // 1. Initialize lambda
    for (i = 0; i < row; ++i)
        lambda[i] = ST.row_ptr[i + 1] - ST.row_ptr[i];

    // 2. Before C/F splitting algorithm starts, filter out the variables which
    //    have no connections at all and mark them as special F-variables.
    for (i = 0; i < row; ++i) {
        if (S->row_ptr[i + 1] == S->row_ptr[i]) {
            vec[i] = ISPT;      // set i as an ISOLATED fine node
            lambda[i] = 0;
        }
        else {
            vec[i] = UNPT;      // set i as a undecided node
            num_left++;
        }
    }

    // 3. Form linked list for lambda (max to min)
    for (i = 0; i < row; ++i) {
        if (vec[i] == ISPT) continue;           // skip isolated variables

        measure = lambda[i];

        if (measure > 0) {
            enter_list(&head, &tail, lambda[i], i, lists, where);
        }
        else {
            if (measure < 0) printf("### WARNING: Negative lambda[%d]!\n", i);

            // Set variables with non-positive measure as F-variables
            vec[i] = FGPT;      // no strong connections, set i as fine node
            --num_left;

            // Update lambda and linked list after i->F
            for (k = S->row_ptr[i]; k < S->row_ptr[i + 1]; ++k) {
                j = S->col_idx[k];
                if (vec[j] == ISPT)
                    continue;   // skip isolate variables
                if (j < i) {
                    newmeas = lambda[j];
                    if (newmeas > 0) {
                        remove_node(&head, &tail, newmeas, j, lists,
                                    where);
                    }
                    newmeas = ++(lambda[j]);
                    enter_list(&head, &tail, newmeas, j, lists, where);
                }
                else {
                    newmeas = ++(lambda[j]);
                }
            }
        }                       // end if measure
    }                           // end for i

    // 4. Main loop
    while (num_left > 0) {
        // pick $i\in U$ with $\max\lambda_i: C:=C\cup\{i\}, U:=U\\{i\}$
        maxnode = head->head;
        maxmeas = lambda[maxnode];

        if (maxmeas == 0) printf("### WARNING: Head of the list has measure 0!\n");

        vec[maxnode] = CGPT;    // set maxnode as coarse node
        lambda[maxnode] = 0;
        --num_left;
        remove_node(&head, &tail, maxmeas, maxnode, lists, where);
        col++;

        // for all $j\in S_i^T\cAp U: F:=F\cup\{j\}, U:=U\backslash\{j\}$
        for (i = ST.row_ptr[maxnode]; i < ST.row_ptr[maxnode + 1]; ++i) {
            j = ST.col_idx[i];

            if (vec[j] != UNPT) continue;       // skip decided variables

            vec[j] = FGPT;      // set j as fine node
            remove_node(&head, &tail, lambda[j], j, lists, where);
            --num_left;

            // Update lambda and linked list after j->F
            for (l = S->row_ptr[j]; l < S->row_ptr[j + 1]; l++) {
                k = S->col_idx[l];
                if (vec[k] == UNPT) {   // k is unknown
                    remove_node(&head, &tail, lambda[k], k, lists, where);

                    newmeas = ++(lambda[k]);
                    enter_list(&head, &tail, newmeas, k, lists, where);
                }
            }
        }                       // end for i

        // Update lambda and linked list after maxnode->C
        for (i = S->row_ptr[maxnode]; i < S->row_ptr[maxnode + 1]; ++i) {

            j = S->col_idx[i];

            if (vec[j] != UNPT) continue;       // skip decided variables

            measure = lambda[j];
            remove_node(&head, &tail, measure, j, lists, where);
            lambda[j] = --measure;

            if (measure > 0) {
                enter_list(&head, &tail, measure, j, lists, where);
            }
            else {              // j is the only point left, set as fine variable
                vec[j] = FGPT;
                --num_left;

                // Update lambda and linked list after j->F
                for (l = S->row_ptr[j]; l < S->row_ptr[j + 1]; l++) {
                    k = S->col_idx[l];
                    if (vec[k] == UNPT) {       // k is unknown
                        remove_node(&head, &tail, lambda[k], k, lists, where);
                        newmeas = ++(lambda[k]);
                        enter_list(&head, &tail, newmeas, k, lists, where);
                    }
                }               // end for l
            }                   // end if
        }                       // end for
    }                           // end while

    // C/F splitting of RS coarsening check C1 Criterion
    SSS_iarray_set(row, grAph_array, -1);
    for (i = 0; i < row; i++) {
        if (vec[i] == FGPT) {
            row_end_S = S->row_ptr[i + 1];
            for (ji = S->row_ptr[i]; ji < row_end_S; ji++) {
                j = S->col_idx[ji];
                if (vec[j] == CGPT) {
                    grAph_array[j] = i;
                }
            }
            cnt = 0;
            for (ji = S->row_ptr[i]; ji < row_end_S; ji++) {
                j = S->col_idx[ji];
                if (vec[j] == FGPT) {
                    set_empty = 1;
                    row_end_S_nabor = S->row_ptr[j + 1];
                    for (jj = S->row_ptr[j]; jj < row_end_S_nabor; jj++) {
                        index = S->col_idx[jj];
                        if (grAph_array[index] == i) {
                            set_empty = 0;
                            break;
                        }
                    }
                    if (set_empty) {
                        if (cnt == 0) {
                            vec[j] = CGPT;
                            col++;
                            grAph_array[j] = i;
                            jkeep = j;
                            cnt = 1;
                        }
                        else {
                            vec[i] = CGPT;
                            vec[jkeep] = FGPT;
                            break;
                        }
                    }
                }
            }
        }
    }

    SSS_imat_destroy(&ST);

    if (head) {
        list_ptr = head;
        head->prev_node = NULL;
        head->next_node = NULL;
        head = list_ptr->next_node;
        SSS_free(list_ptr);
    }

    eofc:
    SSS_free(work);

    return col;
}


static int clean_ff_couplings(SSS_IMAT *S, SSS_IVEC *vertices, int row, int col)
{
    // local variables
    int *vec = vertices->d;
    int *cindex = (int *) SSS_calloc(row, sizeof(int));
    int set_empty = TRUE, C_i_nonempty = FALSE;
    int ci_tilde = -1, ci_tilde_mark = -1;
    int ji, jj, i, j, index;

    SSS_iarray_set(row, cindex, -1);

    for (i = 0; i < row; ++i) {

        if (vec[i] != FGPT)
            continue;           // skip non F-variables

        for (ji = S->row_ptr[i]; ji < S->row_ptr[i + 1]; ++ji) {
            j = S->col_idx[ji];
            if (vec[j] == CGPT)
                cindex[j] = i;  // mark C-neighbors
            else
                cindex[j] = -1; // reset cindex
        }

        if (ci_tilde_mark != i)
            ci_tilde = -1;      //???

        for (ji = S->row_ptr[i]; ji < S->row_ptr[i + 1]; ++ji) {

            j = S->col_idx[ji];

            if (vec[j] != FGPT)
                continue;       // skip non F-variables

            // check whether there is a C-connection
            set_empty = TRUE;
            for (jj = S->row_ptr[j]; jj < S->row_ptr[j + 1]; ++jj) {
                index = S->col_idx[jj];
                if (cindex[index] == i) {
                    set_empty = FALSE;
                    break;
                }
            }                   // end for jj

            // change the point i (if only F-F exists) to C
            if (set_empty) {
                if (C_i_nonempty) {
                    vec[i] = CGPT;
                    col++;
                    if (ci_tilde > -1) {
                        vec[ci_tilde] = FGPT;
                        col--;
                        ci_tilde = -1;
                    }
                    C_i_nonempty = FALSE;
                    break;
                }
                else {          // temporary set j->C and roll back
                    vec[j] = CGPT;
                    col++;
                    ci_tilde = j;
                    ci_tilde_mark = i;
                    C_i_nonempty = TRUE;
                    i--;        // roll back to check i-point again
                    break;
                }               // end if C_i_nonempty
            }                   // end if set_empty
        }                       // end for ji
    }                           // end for i

    SSS_free(cindex);

    return col;
}


static void form_P_pattern_dir(SSS_MAT *P, SSS_IMAT *S, SSS_IVEC *vertices, int row, int col)
{
    // local variables
    int i, j, k, index;
    int *vec = vertices->d;

    // Initialize P matrix
    P->num_rows = row;
    P->num_cols = col;
    P->row_ptr = (int *) SSS_calloc(row + 1, sizeof(int));

    // step 1: Find the structure IA of P first: using P as a counter
    for (i = 0; i < row; ++i) {
        switch (vec[i]) {
            case FGPT:         // fine grid points
                for (j = S->row_ptr[i]; j < S->row_ptr[i + 1]; j++) {
                    k = S->col_idx[j];
                    if (vec[k] == CGPT)
                        P->row_ptr[i + 1]++;
                }
                break;

            case CGPT:         // coarse grid points
                P->row_ptr[i + 1] = 1;
                break;

            default:           // treat everything else as isolated
                P->row_ptr[i + 1] = 0;
                break;
        }
    }                           // end for i

    // Form P->row_ptr from the counter P
    for (i = 0; i < P->num_rows; ++i) P->row_ptr[i + 1] += P->row_ptr[i];

    P->num_nnzs = P->row_ptr[P->num_rows] - P->row_ptr[0];

    // step 2: Find the structure JA of P
    P->col_idx = (int *) SSS_calloc(P->num_nnzs, sizeof(int));
    P->val = (double *) SSS_calloc(P->num_nnzs, sizeof(double));

    for (index = i = 0; i < row; ++i) {
        if (vec[i] == FGPT) {   // fine grid point
            for (j = S->row_ptr[i]; j < S->row_ptr[i + 1]; j++) {
                k = S->col_idx[j];
                if (vec[k] == CGPT)
                    P->col_idx[index++] = k;
            }                   // end for j
        }                       // end if
        else if (vec[i] == CGPT) {      // coarse grid point -- one entry only
            P->col_idx[index++] = i;
        }
    }
}

//Generate sparsity pattern of prolongation for standard interPolation
static void form_P_pattern_std(SSS_MAT *P, SSS_IMAT *S, SSS_IVEC *vertices, int row, int col)
{
    // local variables
    int i, j, k, l, h, index;
    int *vec = vertices->d;

    // number of times a C-point is visited
    int *visited = (int *) SSS_calloc(row, sizeof(int));

    P->num_rows = row;
    P->num_cols = col;
    P->row_ptr = (int *) SSS_calloc(row + 1, sizeof(int));

    SSS_iarray_set(row, visited, -1);

    // Step 1: Find the structure IA of P first: use P as a counter
    for (i = 0; i < row; ++i) {

        if (vec[i] == FGPT) {   // if node i is a F point
            for (j = S->row_ptr[i]; j < S->row_ptr[i + 1]; j++) {
                k = S->col_idx[j];

                // if neighbor of i is a C point, good
                if ((vec[k] == CGPT) && (visited[k] != i)) {
                    visited[k] = i;
                    P->row_ptr[i + 1]++;
                }

                // if k is a F point and k is not i, look for indirect C neighbors
                else if ((vec[k] == FGPT) && (k != i)) {
                    for (l = S->row_ptr[k]; l < S->row_ptr[k + 1]; l++) { // neighbors of k
                        h = S->col_idx[l];
                        if ((vec[h] == CGPT) && (visited[h] != i)) {
                            visited[h] = i;
                            P->row_ptr[i + 1]++;
                        }
                    }           // end for(l=S->row_ptr[k];l<S->row_ptr[k+1];l++)
                }               // end if (vec[k]==CGPT)
            }                   // end for (j=S->row_ptr[i];j<S->row_ptr[i+1];j++)
        }
        else if (vec[i] == CGPT) {      // if node i is a C point
            P->row_ptr[i + 1] = 1;
        }
        else {                  // treat everything else as isolated points
            P->row_ptr[i + 1] = 0;
        }                       // end if (vec[i]==FGPT)
    }                           // end for (i=0;i<row;++i)

    // Form P->row_ptr from the counter P
    for (i = 0; i < P->num_rows; ++i) P->row_ptr[i + 1] += P->row_ptr[i];

    P->num_nnzs = P->row_ptr[P->num_rows] - P->row_ptr[0];

    // Step 2: Find the structure JA of P
    P->col_idx = (int *) SSS_calloc(P->num_nnzs, sizeof(int));
    P->val = (double *) SSS_calloc(P->num_nnzs, sizeof(double));

    SSS_iarray_set(row, visited, -1);  // re-init visited array

    for (i = 0; i < row; ++i) {
        if (vec[i] == FGPT) {   // if node i is a F point
            index = 0;

            for (j = S->row_ptr[i]; j < S->row_ptr[i + 1]; j++) {
                k = S->col_idx[j];

                // if neighbor k of i is a C point
                if ((vec[k] == CGPT) && (visited[k] != i)) {
                    visited[k] = i;
                    P->col_idx[P->row_ptr[i] + index] = k;
                    index++;
                }
                // if neighbor k of i is a F point and k is not i
                else if ((vec[k] == FGPT) && (k != i)) {
                    for (l = S->row_ptr[k]; l < S->row_ptr[k + 1]; l++) { // neighbors of k
                        h = S->col_idx[l];
                        if ((vec[h] == CGPT) && (visited[h] != i)) {
                            visited[h] = i;
                            P->col_idx[P->row_ptr[i] + index] = h;
                            index++;
                        }
                    }           // end for (l=S->row_ptr[k];l<S->row_ptr[k+1];l++)
                }               // end if (vec[k]==CGPT)
            }                   // end for (j=S->row_ptr[i];j<S->row_ptr[i+1];j++)
        }
        else if (vec[i] == CGPT) {
            P->col_idx[P->row_ptr[i]] = i;
        }
    }

    // clean up
    SSS_free(visited);
}
int SSS_amg_coarsen(SSS_MAT *A, SSS_IVEC *vertices, SSS_MAT *P, SSS_IMAT *S, SSS_AMG_PARS *pars)
{
    SSS_COARSEN_TYPE coarse_type = pars->cs_type;
    int row = A->num_rows;
    int interp_type = pars->interp_type;
    int col = 0;

    // find strong couplings and return them in S
    strong_couplings(A, S, pars);

    switch (coarse_type) {
        case SSS_COARSE_RS:        // Classical coarsening
            col = cfsplitting_cls(A, S, vertices);
            break;

        case SSS_COARSE_RSP:       // Classical coarsening with positive connections
            //col = cfsplitting_clsp(A, S, vertices);
            break;

        default:
            SSS_exit_on_errcode(ERROR_AMG_COARSE_TYPE, __FUNCTION__);
    }

    if (col <= 0) return ERROR_UNKNOWN;

    switch (interp_type) {
        case intERP_DIR:       // Direct interPolation or ...
            col = clean_ff_couplings(S, vertices, row, col);
            form_P_pattern_dir(P, S, vertices, row, col);
            break;

        case intERP_STD:       // Standard interPolation
            form_P_pattern_std(P, S, vertices, row, col);
            break;

        default:
            SSS_exit_on_errcode(ERROR_AMG_interp_type, __FUNCTION__);
    }

    return 0;
}
