#ifndef soc_h
#define soc_h

#include <mpi.h>
#include <omp.h>
#include <cmath>
#include <algorithm> // fill_n
#include <cstring>   // memcpy

#include "const.h"
#include "fn.h"
#include "math.h"
#include "wave_base.h"

using namespace std;

class socclass {
    public:
    
    // field
    string dirpath; // VASP results directory path
    waveclass *wvc;
    int malloc_root, malloc_end, share_memory_len; // inherit from wvc
    complex<double> *Lplus_mmp[4];     // L_{+}: [l=1~4, (2l+1) * (2l+1)], row major
    complex<double> *Lminus_mmp[4];    // L_{-}: [l=1~4, (2l+1) * (2l+1)], row major
    complex<double> *Lz_mmp[4];        // L_{z}: [l=1~4, (2l+1) * (2l+1)], row major
    complex<double> *U_C2R[4];         // U_C2R: [l=1~4, (2l+1) * (2l+1)], row major
    complex<double> *LS_mmp[4];        // LS: [l=1~4, 4 * (2l+1) * (2l+1)], row major
                                       // (0~3)[(2l+1)*(2l+1)] for
                                       // <up|SO|up>,<up|SO|dn>,<dn|SO|up>,<dn|SO|dn>,
    int nions;
    int *lmmax = NULL;                 // lmmax[nions] for each ion
    int *lmax  = NULL;                 // lmax[iatom] = wvc->atoms[iatom].potc->projL.size()
    int *lmmax_loc_r = NULL;           // lmmax_loc_r[nions] for each ion of current process row
    int *lmmax_loc_c = NULL;           // lmmax_loc_c[nions] for each ion of current process column
    int *lmmax2_loc  = NULL;           // lmmax_loc_r[iatom] x lmmax_loc_c[iatom]
    
    complex<double> **hsoc_base;       // shape[nions][4(for spin), lmmax, lmmax], not change during dynamics
    complex<double> *hsoc_baseall;
    MPI_Win window_hsoc_base; complex<double> *local_hsoc_base_node;
    complex<double> **hsoc;            // shape[nions][4(for spin), lmmax, lmmax], calculate from hsoc_base
    complex<double> *hsocall;
    MPI_Win window_hsoc; complex<double> *local_hsoc;
    
    double          **socrad;          // shape[nions][lmax, lmax], read from SocRadCar, row major
    int nbnds_loc_r, nbnds_loc_c;      // local # of rows and columns for nbnds
    int twonbnds;                      // global, 2 x wvc->nbnds;
    int twonbnds_loc_r;                // local # of rows    for 2 x nbnds
    int twonbnds_loc_c;                // local # of columns for 2 x nbnds
    complex<double> *socmat = NULL;    // shape[ twonbnds, twonbnds ]
    double *eigenvals;                 // diagonalize socmat to get the eigenvalues eigenvals[twonbnds]
    double totweight;                  // sum{ eigens[first kpt].weight }
    int maxnpw;                        // max{ npw[all kpts], 2 + nbnds x 3 }, the latter represents double values of: 
                                       // 2npw(one number, value x 2), kptvec[3], (energy, 0, weight)[nbnds x 3] x 2
    complex<float> *oneline = NULL;    // for wavecar output, write one line
    complex<double> **spinor_pp = NULL;// projphi for spinor, should be a superposition of <~p^a_i|ps_{n, sigma}>
                                       // shape of [nions][2 x lmmax_loc_r x twonbnds_loc_c], "2" for spin channel(sigma)
    complex<double> **Qijpp = NULL;    // Qij x <~p^a_j|ps_{n, sigma}>: shape[nions][2 * lmmax_loc_r x twonbnds_loc_c]
    complex<double> *band_spin_polarization = NULL; // shape[4(is1, is2) x twonbnds], global
    
    vector<int> read_in_kpts; // for CorrectSocmat
    
    // method
    int  CheckSpinorWavecar();
    void GetProjPhi();
    void SetupLS(); bool isSetupLS = false;
    void Setup_hsoc_base(const int myprow = myprow_group,
                         const int mypcol = mypcol_group,
                         const int nprow = nprow_group,
                         const int npcol = npcol_group); bool isSetup_hsoc_base = false;
    void ReadSocRad(const char *socradcar);
    void Update_hsoc(const bool is_write_socallcar = false); // default false
    void SetSocmatDiag(const int in_kpt);
    void GetSpinorCoeff(const int in_kpt, complex<double> *spinorpw);
    void CorrectSocmatReadIn();
    void CorrectSocmat(const int ikpt);
    void WriteSpinorHead(ofstream &wavecar, ofstream &socout);
    void WriteSpinor(const int in_kpt, const complex<double> *spinorpw, 
                     ofstream &wavecar, ofstream &socout, const char *suffix); bool isSettotweight = false;
    bool MakeSpinor(const char *idirpath, const char *suffix = "");
    
    // constructor
    socclass(waveclass *wvc_aux); // default, dummy
    void Initial(); bool isInitial = false;

    // destructor
    ~socclass();
};

#endif
