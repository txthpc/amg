#ifndef _SSS_COARSEN_H_
#define _SSS_COARSEN_H_

#include "../SSS_main.h"
#include "../SSS_matvec.h"
#include "../SSS_utils.h"
//#include "SSS_SETUP.h" 

 #if defined (__cplusplus)
extern "C"
{
#endif    

//Create an node using Item for its data field
static LinkList create_node(int Item);

//Places point in new list
static void enter_list(LinkList *head_ptr, LinkList *tail_ptr,int measure, int index, int *lists, int *where);

//Generate the set of all strong negative couplings    //生成所有强负耦合的集合
static void strong_couplings(SSS_MAT *A, SSS_IMAT *S, SSS_AMG_PARS *pars);

//Remove weak couplings from S (marked as -1)
static int compress_S(SSS_IMAT * S);

static void dispose_node(LinkList node_ptr);

//Removes a point from the lists
static void remove_node(LinkList * head_ptr, LinkList * tail_ptr,int measure, int index, int * lists, int * where);

// Find coarse level variables (classic C/F splitting)
static int cfsplitting_cls(SSS_MAT *A, SSS_IMAT *S, SSS_IVEC *vertices);

static int clean_ff_couplings(SSS_IMAT *S, SSS_IVEC *vertices, int row, int col);

static void form_P_pattern_dir(SSS_MAT *P, SSS_IMAT *S, SSS_IVEC *vertices, int row, int col);

//Generate sparsity pattern of prolongation for standard interPolation
static void form_P_pattern_std(SSS_MAT *P, SSS_IMAT *S, SSS_IVEC *vertices, int row, int col);

int SSS_amg_coarsen(SSS_MAT *A, SSS_IVEC *vertices, SSS_MAT *P, SSS_IMAT *S, SSS_AMG_PARS *pars);


#if defined (__cplusplus)
}
#endif


#endif