#ifndef bse_h
#define bse_h

#include <omp.h>
#include <mpi.h>

#include <iostream>
#include <fstream>
#include <string>
#include <algorithm> // min, max, max_element

#include "const.h"
#include "fn.h"
#include "math.h"
#include "wave_base.h"
#include "wpot.h"
#include "optical.h"

using namespace std;

class excitonclass {
    public:
    // field
    int dirnum;
    waveclass *wvc;
    wpotclass *wpc;
    int spinor;
    double omega;               // entire crystal volume
    double omega0; int ncells;  // available when use GW, inherit from wpc
    int kappa;                  // kappa = 2/0 for singlet/triplet state
    int NK_SC[3];               // sanme kpoints in wvc & wpc NK_SC
    int NKSCtot;                // NK_SC[0 x 1 x 2]
    int NKSCtot2;               // NKSCtot x NKSCtot
    int dimC, dimV, dim, nsdim; // dim = NKSCtot x dimC x dimV, nsdim = bsespns x dim
    int dim_loc_row;            // local of dim by row
    int dim_loc_col;            // local of dim by column
    int nsdim_loc_row;          // local of nsdim by row
    int nsdim_loc_col;          // local of nsdim by column
    int dimCC, dimVV;           // dimCC = dimC x dimC, dimVV = dimV x dimV
    int dimCC_loc_row;          // local of dimCC by row
    int dimCC_loc_col;          // local of dimCC by column
    int dimVV_loc_row;          // local of dimVV by row
    int dimVV_loc_col;          // local of dimVV by column
    int dimKKCC, dimKKVV;       // dimKKCC(VV) = NKSCtot x NKSCtot x dimC(V) x dimC(V)
    int dimKKCC_loc_row;        // local of dimKKCC by row
    int dimKKCC_loc_col;        // local of dimKKCC by column
    int dimKKVV_loc_row;        // local of dimKKVV by row
    int dimKKVV_loc_col;        // local of dimKKVV by column
    int NBtot;                  // NBtot = dimC + dimV
    
    int malloc_root, malloc_end, share_memory_len; // inherit from wvc
    int *all_malloc_root;
    int *all_mal_nodeclr;
    
    int ngf[3];                 // inherit from wvc
    int ngftot;                 // inherit from wvc
    int ngf_loc_n0;             // inherit from wvc
    int ngf_loc_0_start;        // inherit from wvc
    int *all_ngf_loc_n0;        // inherit from wvc
    int *all_ngf_loc_0_start;   // inherit from wvc
    int tot_loc_ngf;            // inherit from wvc by Pzfft3SetMallocSize
    int ngftot_loc;             // ngf_loc_n0 x ngf[1] x ngf[2] may < tot_loc_ngf
    int numQ;                   // the same as in wpc: number of kpoints difference: Q = k1 - k2
    double emax;                // inherit from wpc->emax_GW or = wvc->emax x (2/3)
    int *npw = NULL;            // inherit from wpc->Qcs[numQ].npwG
    int *npw_loc = NULL;        // loc of npw by rows
    int ng[3];                  // inherit from wpc->NG_SCGW or build by current GetngBSE
    int ngtot;                  // ng[0] * ng[1] * ng[2]
    double **qptvecs;           // inherit from wpc->Qcs[].Qptvec or made by GetgidxBSE: [numQ][3]
    int **gidx;                 // inherit from wpc->Qcs[].Gind or made by GetgidxBSE:   [numQ][npw]
    int *gidxall;
    MPI_Win window_gidx; int *local_gidx_node;

    double **qgabs;             // distance of q+G vectors: [numQ][npw]
    double *qgabsall;
    MPI_Win window_qgabs;    double *local_qgabs_node;
    double **qgtheta;           // theta    of q+G vectors: [numQ][npw]
    double *qgthetaall;
    MPI_Win window_qgtheta;  double *local_qgtheta_node;
    double **qgphi;             // phi      of q+G vectors: [numQ][npw]
    double *qgphiall;
    MPI_Win window_qgphi;    double *local_qgphi_node;

    int *totnvij;               // sum_{ia = 0}^{current atom} tot_nv_ij
    int *totlmmax;              // sum_{ia = 0}^{current atom} lmmax
    
    double **fPiOverGabs2;      // (4pi / |G|^2) in the unit of (eV x Angstrom^3) for all G of different q: [][npw]
    double *fPiOverGabs2all;
    MPI_Win window_fPiOverGabs2; double *local_fPiOverGabs2;
    complex<double> **fftIntPre;// there should be a prefix factor from discrete transform(sum) to integral: [numQ][npw]
    complex<double> *fftIntPreall;
    MPI_Win window_fftIntPre; complex<double> *local_fftIntPre_node; 
    complex<double> *ccDenMat = NULL;      // [npw x (dimC x dimC)]
    complex<double> *vvDenMat = NULL;      // [npw x (dimV x dimV)]
    complex<double> *cvDenMatLeft = NULL;  // [npw x dim]
    complex<double> *cvDenMatRight = NULL; // [npw x dim]
    complex<double> *kkccDenMat = NULL;    // [npw x dimKKCC]
    complex<double> *kkvvDenMat = NULL;    // [npw x dimKKCC]
    
    // method
    void GetngBSE();
    bool WithinSphereBSE(int i, int j, int k, double *qpoint);
    void GetgidxBSE(); bool isGetgidxBSE = false;
    void Getqgabsdir(); bool isGetqgabsdir = false;
    void GetfftIntPre(const int sign); bool isGetfftIntPre = false;
    void GetfPiOverGabs2(const int nQQ); bool isGetfPiOverGabs2 = false;
    void DenMatColumnPseudoPart(const int s1, const int k1, const int n1,
                                const int s2, const int k2, const int n2,
                                const int out_col_idx, const int out_tot_cols,
                                complex<double> *outmat, const int idxQ = -1);
    
    void DenMatCPLeftMatBlockCyclic(const int iqpt, complex<double> *leftmat);
    void DenMatCPLeftMatBlockCyclicOneAtom(const int iqpt, const int iatom, complex<double> *leftmat);
    void DenMatCPLeftMatAtomAll(const int iqpt, complex<double> *leftmat);
    void DenMatCPLeftMatAtomEach(const int iqpt, const int iatom, complex<double> *leftmat);
    void DenMatCorePartOneKptPairAtomAll(const int spn, const int k1, const int k2,
                                         const int bnd_start, const int nbnds,
                                         complex<double> *denmat);
    void DenMatCorePartOneKptPairAtomEach(const int spn, const int k1, const int k2,
                                          const int bnd_start, const int nbnds,
                                          complex<double> *denmat);
    void DenMatCorePartKCV(const int spn, complex<double> *denmat);
    void DenMatCorePartKCVOneAtom(const int spn, complex<double> *denmat);
    void DensityMatrixOneKptPair(const int spn, const int k1, const int k2, 
                                 const int bnd_start, const int nbnds,
                                 complex<double> *denmat);
    void DensityMatrixKCV(const int spn, complex<double> *denmat);
    void DirectTerms(const int spnL, const int spnR,
                     const complex<double> alpha, const complex<double> beta,
                     complex<double> *scTerms);
    void ExchangeTerms(const int spnL, const int spnR,
                       const complex<double> alpha, const complex<double> beta,
                       complex<double> *exTerms);
    void DirectTermsNoGW_KbyK(const double epsl, const int spnL, const int spnR,
                              const complex<double> alpha, const complex<double> beta,
                              complex<double> *scTerms);
    void DiagonalSetting(const int nspns, const double gapadd,
                         complex<double> *mat);
    void ScTerms2BSEMat(const int nspns, const int spn1, const int spn2,
                        const complex<double> *scTerms,
                        complex<double> *rootmat, complex<double> *bcmat);
    void ExTerms2BSEMat(const int nspns, const int spn1, const int spn2,
                        const complex<double> *exTerms,
                        complex<double> *rootmat, complex<double> *bcmat);
    void ExcitonTDM(const complex<double> *eigenvecs,
                    complex<double> *Xtdm_full);
    void XEnergies(const complex<double> *bcmat);
    void ExcitonMatrix(const int num);
    
    // constructor
    excitonclass(waveclass *wvc_in, wpotclass *wpc_in); // default, dumy
    void Initial(); bool isInitial = false;
    // destructor
    ~excitonclass();
};

#endif
