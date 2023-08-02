#ifndef wave_base_h
#define wave_base_h

#include <omp.h>
#include <mpi.h>
#include <vector>
#include <fstream>
#include <string>
#include <sstream>
#include <algorithm> // copy find min max unique
#include <numeric>   // iota accumulate
#include <iterator>  // back_inserter
#include <cmath>
#include <complex>

#include "const.h"
#include "fn.h"
#include "math.h"
#include "paw.h"

#ifndef IS_EXCLUDE_BANDS
#define IS_EXCLUDE_BANDS 1
#endif

#ifndef NOT_EXCLUDE_BANDS
#define NOT_EXCLUDE_BANDS 0
#endif

#ifndef IS_MALCOEFF
#define IS_MALCOEFF 1
#endif

#ifndef IS_MALCOEFF_FULL
#define IS_MALCOEFF_FULL 2
#endif

#ifndef NOT_MALCOEFF
#define NOT_MALCOEFF 0
#endif

using namespace std;

class atomclass {
    public:
    // field
    string element;
    double posfrac[3]; // position in fractional coordinates
    double poscart[3]; // position in Cartesian  coordinates
    pawpotclass *potc; // potential file
    int malloc_root, malloc_end, share_memory_len;
    
    int nkpts;
    complex<double> **crexp;
    // exp(i(G+k).R), total nkpts vectors with size npw[ik]
    // here is nkpts, NOT totnkpts, see details in paw.h
    // updata: int VASP code, it use exp(iG.R) not exp(i(G+k).R)
    //         but here I use the latter one
    complex<double> *crexp_forall;
    MPI_Win window_crexp; complex<double> *local_crexp;
    
    // exp(i(q+G).R), for bse calculation, with size nqpts
    int nqpts;
    complex<double> **crexp_q;
    complex<double> *crexp_q_forall;
    MPI_Win window_crexp_q; complex<double> *local_crexp_q;
    
    complex<double> *projphi_bc = NULL; // "bc" for "block-cyclic"
    complex<double> *projphi = NULL;
    // < p_{l,m; R} | \phi_{\sigma, k, n} > for current R
    // nchannels x lmmax x (nkpts x nbnds): if is_spinor, nchannels = 2; else nchannels = nspns
    int nrow_pp_loc = 0;
    int ncol_pp_loc = 0;
    MPI_Win window_projphi;  complex<double> *local_projphi = NULL;

    complex<double> *Aij[4]; // each Aij may update when ions' positions change
                             // Aij[0] = Qij_z, not update. Calculate in pawpotclass
                             // Aij[1~3] = Gij_z, not update. Calculate in pawpotclass

    // method
    void LoadPosition(string posstr, double *a[]);
    void Getcrexp(const int nkpts, double **kptvecs, const int *npw, int ng[], int **gidx,
                  int rk, int sz, MPI_Comm &comm, const int addkptv = 1); bool isGetcrexp = false;
    void Getcrexp_q(const int nqpts, double **qptvecs, const int *npw,
                    int **gidx, const int ng_bse[3], const int addqptv = 1); bool isGetcrexp_q = false;
    void Getprojphi(const int nspns, const int nkpts, const int nbnds, vector<int> &kpoints, 
                    complex<double> **coeffs, const int *npw, const double volume, const int is_spinor); bool isMallocProjphi = false;
    void Writeprojphi(const char *normcar, const int nspns, const int nkpts, const int nbnds, const int is_spinor);
    
    // constructor
    atomclass(); // default, dummy
    void Initial(string posstr, double *a[], pawpotclass *potc,
                 const int malloc_root, const int malloc_end, const int share_memory_len);

    // destructor
    ~atomclass();
};

void VecCross3d(double *vec1, double *vec2, double *res);
double VecDot3d(double *vec1, double *vec2);

class waveclass {
    public:
    // field
    double *a[3];  //       real space lattice constants: a[0], a[1], a[2]
    double *b[3];  // reciprocal space lattice constants: b[0], b[1], b[2]
                   // b[k] = 1 / Omega0 * a[i] x a[j], Omega0 = a[0] x a[1] . a[2]
    double volume; // Omega0
    int numatoms;
    atomclass *atoms; // all atoms

    int spinor;  // for spinor there should be two parts of wavefunction involving spin up and down
    int totnspns, totnkpts, totnbnds; // total spins, kpoints, bands stored in wavefunction file
    size_t rdum; // line length for VASP WAVECAR

    int nspns;   // # of spins   stored in this class, 1 or 2. For spinor, nspns always equals to 1.
    int nkpts;   // # of kpoints stored in this class
    int nbnds;   // # of bands   stored in this class
    int nstates; // states space dimension: nspns x nkpts x nbnds
    int dimC, dimV; // for exciton
    vector<int>   spins; // list all spins
    vector<int> kpoints; // list all kpoints
    double    **kptvecs; // all kptvecs[0..nkpts-1] are 3-element double* vectors, fractional coordinates
    MPI_Win   window_kptvecs; double *local_kptvecs_node;
    vector<int>  *bands; // list all bands, can be discontinuous. bands[0/1] for first/second spin

    // plane-wave basis sets
    int ng[3];        // {ng[0], ng[1], ng[2]} contains all plane-wave reciprocal lattice points {g_i}
    int ngtot;        // ngtot = ng[0] * ng[1] * ng[2]
    int ngf[3];       // fine ng, at least 2 x ng
    int ngftot;       // ngftot = ngf[0 x 1 x 2]
    int ngf_loc_n0, ngf_loc_0_start;           // for mpi-fftw
    int *all_ngf_loc_n0, *all_ngf_loc_0_start; // with size nprow_group(nprow_onecol, col_size)
    int tot_loc_ngf;  // sometimes > ngf_loc_n0 x ngf[1] x ngf[2], fftw may need more memory space
    double emax;      // energy cut in eV
    int *npw;         // npw[0..nkpts-1] stores # of plane-waves for each kpoint within energy cut sphere
                      // only a pointer, no memory
    MPI_Win window_npw; int *local_npw_node;
    int *npw_loc;     // (Local) npw_loc[ik] stores # of pw coefficients of current process
    int **gidx;       // coefficients index, compacted(sphere) to sparse(cubiod), 
                      // may be different size for different kpoints
                      // all elements in gidx[0..nkpts-1] should non-minus integer values
    int *gidxall;
    int **gidxRev;    // reversed of gidx, elements in gidxRev can have "-1" values mean out of energy cut sphere
    int *gidxRevall;
    MPI_Win window_gidx;    int *local_gidx_node;
    MPI_Win window_gidxRev; int *local_gidxRev_node;

    double **gabs;    double *gabsall;    // distance of g   vectors
    MPI_Win window_gabs;    double *local_gabs_node;
    double **gtheta;  double *gthetaall;  // theta    of g   vectors
    MPI_Win window_gtheta;  double *local_gtheta_node;
    double **gphi;    double *gphiall;    // phi      of g   vectors
    MPI_Win window_gphi;    double *local_gphi_node;
    double **gkabs;   double *gkabsall;   // distance of g+k vectors
    MPI_Win window_gkabs;   double *local_gkabs_node;
    double **gktheta; double *gkthetaall; // theta    of g+k vectors
    MPI_Win window_gktheta; double *local_gktheta_node;
    double **gkphi;   double *gkphiall;   // phi      of g+k vectors
    MPI_Win window_gkphi;   double *local_gkphi_node;
    vector<int> *locbeg = NULL; // these three store the ScaLapack local and global begin index
    vector<int> *glbbeg = NULL; // and block length for each kpoint of npw[ik]
    vector<int> *bcklen = NULL; //
    
    bool is_malcoeff = false;
    complex<double> **coeff_malloc_in_node; // shape: [nkpts][nspns x nbnds x (    npw x (1 + spinor) )]
    complex<double> **real_space_c_in_node; // shpae: [nkpts][nspns x nbnds x ( ngftot x (1 + spinor) )]
    complex<double> *coeff_malloc_in_nodeall;
    complex<double> *real_space_c_in_nodeall;
    // memory malloc in node_share for reciprocal/real space coefficients,
    // again, when spinor = 1, nspns = 1; otherwise nspns = 1 or 2
    MPI_Win window_recpc; complex<double> *local_recpc;
    MPI_Win window_realc; complex<double> *local_realc;
    double *energies;
    MPI_Win window_energies; double *local_energies;
    double *weights;
    MPI_Win window_weights; double *local_weights;
    int malloc_root, malloc_end, share_memory_len;
    int *all_malloc_root;
    int *all_mal_nodeclr;
    MPI_Win window_allroot; int *local_allroot;
    MPI_Win window_allnclr; int *local_allnclr;

    class eigenclass { // class for one eigenstate
        public:
        int spinor;
        int tnspns, tnkpts, tnbnds;  // total # of spins, kpoints, bands
        int   spn ,   kpt ,   bnd ;  // spin, kpoint, band # of original wavefunction file
        int  ispn ,  ikpt ,  ibnd ;  // index for current waveclass
                                     // spn = spins[ispn], kpt = kpoints[ikpt], bnd = bands[ispn][ibnd]
        int istate;
        double kptvec[3];       // kpoint in fractional coordinates
        int *gidx, *gidxRev;    // pointer from waveclass, NO memory allocate
        int npw, npw_loc;       // # of plane waves
        int ng[3], ngf[3];      // inherit from waveclass
        int ngftot;
        int ngf_loc_n0, ngf_loc_0_start, tot_loc_ngf;
        int *all_ngf_loc_n0, *all_ngf_loc_0_start;
        vector<int> locbeg;
        vector<int> glbbeg;
        vector<int> bcklen;
        double energy, weight;  // eigen energy and weight. usually weight = 0/2 for conduction/valance band
        complex<double> *coeff; // ONLY a pointer to coeff_malloc_in_node, no memory malloc
                                // coefficient, compacted storage, size = npw
        complex<double> *realc; // ONLY a pointer to real_space_c_in_node, no memory malloc
                                // coefficient in real space, cubiod storage, ngf[0] x ngf[1] x ngf[2]
        complex<double> extraphase;  // phase correction
        
        //method
        void BandReorder(int *wvorder, int info = 0);
        void OnlyGetEn(const char *wavecar);
        void Getcoeff(const char *wavecar, int rk, int sz, MPI_Comm &comm,
                      double *energies, double *weights,
                      const int isgamma = vaspgam);
        void AddPhase(complex<double> newphase, int info = 0);
        void Getrealc();

        //constructor
        eigenclass(); // default one, dummy
        void Initial(waveclass *wvc, const int ie, const int malcoeff = 1); // real constructor

        //destructor
        ~eigenclass();
    };
    eigenclass *eigens;
    
    // method
    void GetRecpLatt();
    void Getng();
    void DetermineShareMemoryRanks();
    void IniAtoms(const char *posfile, pawpotclass *pawpots); bool isIniAtoms;
    void AtomsRefresh(const char *posfile);
    void StatesRead(const vector<int> inSpins, const vector<int> inKpoints,
                    const vector<int> inBmin, const vector<int> inBmax,
                    const vector<int> inExclude);
    void StatesRead(const vector<int> inSpins, const vector<int> inKpoints,
                    const vector<int> inCmin, const vector<int> inCmax,
                    const vector<int> inVmin, const vector<int> inVmax, 
                    const vector<int> inExclude);
    void StatesRead(); bool isStatesRead;
    void ReadKptvecNpw(const char *wavecar, const int isgamma, const int isncl); bool isReadKptvecNpw; // vasp
    bool WithinSphere(int i, int j, int k, double *kpoint); 
    void GetgBegidxBlocklen(int rk, int sz, MPI_Comm &comm);
    void Getgabsdir(int rk, int sz, MPI_Comm &comm);
    void Getgidx(const int isgamma, const int vasp_ver, const int checknpw = 1); bool isGetgidx; // vasp
    void IniEigens(const int malcoeff = IS_MALCOEFF); bool isIniEigens;
    void ReadCoeff(const char *wavecar, const int isgamma, const int isrealc); bool isReadCoeff; bool isRealCoeff;
    void CalcRealc();
    void CalcProjPhi(const char *normcar, const bool is_write_normcar);
    int  ReadProjPhi(const char *normcar);
    void WriteBandEnergies(const char *filename);
    void WriteExtraPhases(const char *filename);
    
    // constructor
    waveclass(const char *wfcFile, const int spinor = 0); // standard constructor, from vasp WAVECAR file
    
    // destructor
    ~waveclass();
};

#endif
