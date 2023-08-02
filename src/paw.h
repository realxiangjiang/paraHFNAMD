#ifndef paw_h
#define paw_h

#include <string>
#include <sstream>
#include <vector>
#include <fstream>
#include <cmath>
#include <algorithm> // max, max_element, sort, unique, fill_n
#include <cassert>   // assert

#include "const.h"
#include "fn.h"
#include "math.h"
#include "w3j.h"

using namespace std;

void Gradradial(vector<double> &rgrid, vector<double> &f, double *dfdr);

class pawpotclass {
/*      From zqj github
        
        Reciprocal space presentation of the nonlocal projector functions on a plane-waves grid.
        
        p_{l,m; R}(G + k) = ( 1. / sqrt(Omega) ) * i^l * f(G + k) * ylm(G + k) * exp(i(G+k).R)  (1)
update:
        in VASP code, it omit the kpoint vector "k", the projector is:
        p_{l,m; R}(G + k) = ( 1. / sqrt(Omega) ) * i^l * f(G + k) * ylm(G + k) * exp(iG.R)      (2)
        but in my current code, I add the "k" here
        
        where "f(G + k)" is the radial part of the reciprocal projector functions, 
        which are stored in POTCAR file. "ylm(G+k)" is the real spherical harmonics
        with corresponding "l" and "m". "Omega" is the volume of the cell. 
        The phase factor of "exp(i(G+k)*R)" is stored in "crexp"
        The "i^l" is stored in "cqfak".
        The application of the projector functions on the pseudo-wavefunction 
        can then be obtained: C_n = < p_{l,m; R} | \phi_{n,k} >
        
        C_n = \sum_G C_{n,k}(G + k) * p_{l,m; R}(G + k)
*/
    public:
    
    // field
    bool isReadPWFC;
    bool isReadProj;

    string *ppcstr;
    string element;
    double projGmax; // maximal G for reciprocal non local projectors
    double projRmax; // maximal R for real non local projectors
    vector<int> projL;     // L quantum number for each projector functions
    vector<int> projLbeg;  // sum{2 * L + 1 before current L}
    int lmax;              // lmax = projL.size()
    int lmmax;             // lmmax = sum{all 2 * L + 1 for all L in projL}
    int lmmax2;            // lmmax2 = lmmax * lmmax
    int lmmax_loc_r, lmmax_loc_c, lmmax2_loc; // lmmax2_loc = lmmax_loc_r x lmmax_loc_c
    vector<int> each_l;    // size = lmmax, store each l
    vector<int> each_m;    // size = lmmax, store each m
    vector<int> each_idxl; // size = lmmax, store each index of l, see details in paw.cpp
    vector<double> qProjs; // projector functions in reciprocal space, each one in a vector
    vector<double> rProjs; // projector functions in real       space, total lmax vectors
    
    vector<int> kpoints;
    complex<double> **qProjsKpt;
    /* 
       in one vector, the values are i^l * f(G + k) * ylm(G + k) 
       which are about G with size npw[ik] and set values outside projGmax to zero.
       Because usually the cooresponding encut of projGmax > 600 eV
       and no need to set qProjsKpt to compacted
       total totnkpts x lmmax vectors, may empty if current kpoint isn't concerned
   */
    complex<double> *qProjsKpt_all = NULL;
    MPI_Win window_qProjsKpt; complex<double> *local_qProjsKpt_node = NULL;
                                                

    int radnmax;          // # of radial values
    vector<double> rgrid; // radial grid points, logarithmic, rgrid[i] = rgrid[0] * exp(H * i)
    vector<double> *pswfcs = NULL;    // pesudo       wavefunctions, core region
    vector<double> *aewfcs = NULL;    // all-electron wavefunctions, core region
    double *simp_wt = NULL;           // Simpson integration, size radnmax
    double *Qij = NULL;               // Q_{ij} = < \phi_i^{AE} | \phi_j^{AE} > - < \phi_i^{PS} | \phi_j^{PS} >
                                      // column major, distributed storage with size of lmmax_loc_r x lmmax_loc_c
    complex<double> *Qij_z = NULL;    // Qij_z = Qij
    complex<double> *Qij_z_full;
    MPI_Win window_Qijzfull; complex<double> *local_Qijzfull_node;
    double *Gij = NULL;               // G_{ij} = < \phi_i^{AE} | \grad | \phi_j^{AE} > 
                                      //        - < \phi_i^{PS} | \grad | \phi_j^{PS} >
    complex<double> *Gij_z = NULL;    // Gij_z = Gij
    int maxL;                         // maxL = 2 x projL.max
    double **JLij = NULL;             // JL_{ij} = < \phi_i^{AE} | j_L(kr) | \phi_j^{AE} > 
                                      //         - < \phi_i^{PS} | j_L(kr) | \phi_j^{PS} >
                                      // j_L is spherical Bessel function with 0 <= L <= maxL
                                      // each L corresponds to a [NPSQNL x unique_idxlpair.size()] matrix
                                      // total (maxL + 1) matrixes
    double *JLijall = NULL;
    MPI_Win window_JLij; double *local_JLij_node;
    vector<int> unique_idxlpair;      // different for each process column
    vector<int> idx_of_idxlpair;      // the index in unique_idxlpair
    int nqpts;
    complex<double> **Ekrij;
    complex<double> *Ekrij_all;
    MPI_Win window_Ekrij; complex<double> *local_Ekrij_node;
    /* 
        < \psi_i^{AE} | exp(ik.r) | \psi_j^{AE} > - < \psi_i^{PS} | exp(ik.r) | \psi_j^{PS} >
        in which k = q + G, total nqpts matrixes, each with size of [npw(q) x lmmax^2]
    */
    vector<int> all_nv_ij;            // in some ij, Ekrij always vanish, this vector stores all non-vanishing ones
    int tot_nv_ij;                    // tot_nv_ij = all_nv_ij.size()


    // method
    void ReadProj();
    void ReadPartialWFC();
    void SetSimpiWeight();
    double CalcSimpInt(double *f);
    void CalcQij(); bool isCalcQij = false;
    void CalcGij(); bool isCalcGij = false;
    void CalcJLij();
    void CalcQprojKpts(const int totnkpts, const vector<int> &kpoints, const int *npw,
                       double **gkabs, double **gktheta, double **gkphi); bool isCalcQprojKpts = false;
    void CalcEkrij(const int nqpts,
                   const int *npw, const int *npw_loc,
                   double **qgabs, double **qgtheta, double **qgphi);
    bool isCalcEkrij = false;
    void ReadEkrij(const int iqpt, const int *npw);
    void DelEkrij(const int iqpt);

    // constructor
    pawpotclass(); // default, dumy
    void Initial(string &ppcstr_in); // real constructor

    // destructor
    ~pawpotclass();
};

int ReadAllpawpot(pawpotclass *&pawpots, const char* potcar);

#endif
