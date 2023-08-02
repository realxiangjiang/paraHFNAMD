#ifndef wpot_h
#define wpot_h

#include <omp.h>
#include <mpi.h>

#include <iostream>
#include <fstream>
#include <string>
#include <algorithm> // min, max, max_element, copy, copy_if, fill_n
#include <cassert>

#include "const.h"
#include "fn.h"
#include "math.h"
#include "wave_base.h"
#include "sym.h"

using namespace std;

void RecpSC2pc(double &x0, double &x1, double &x2,
               double  X0, double  X1, double  X2, double M[][3]);

class wpotclass {
    public:
    // field
    waveclass *wvc;
    double *a_GW[3];       //       real space lattice constants for GW: a_GW[0], a_GW[1], a_GW[2]
    double *b_GW[3];       // reciprocal space lattice constants for GW: b_GW[0], b_GW[1], b_GW[2]
    double omega0_GW;      // cell volume of GW cell
    double emax_SCGW;      // when calculate GW in SC, the encut is usually small
    int      NG_SCGW[3];   // like the ng[3] in wave_base.h
    int   NGtot_SCGW;
    double emax_GW;        // read GW info in wpotdir
    int      ng_GW[3];
    int   ngtot_GW;
    int M[3][3];           // transformation matrix M^T from B(wvc->b) to b(b_GW) in which M: a(a_GW) to A(wvc->a)
    double MTinv[3][3];    // (M^T)^(-1)
    int detM;              // det(M)
    int ncells;            // relative to GW cell: ncells = det(M) x NK_SC
    int NK_SC[3];          // same kpoints in wvc
    int NK_GW[3];          // usually primitive cell
    int nktotgw;           // read in wpotdir/OUTCAR and should equal to NK_GW[0] * NK_GW[1] * NK_GW[2]
    int nkibrgw;           // # of kpoints in ibr
    int *npwGW;            // npwGW[0..nkibrgw-1]
    MPI_Win window_npwGW; int *local_npwGW_node;
    int numQ;              // # of Q: differences of K_SC
    double **allgwkpts;    // all gw kpoints:    [nktotgw][3]
    int     *gwkpinibz;    // gw kpoints in ibz: [nktotgw]
    int     *gwkp_trev;    // gw kpoints if time reversal: [nktotgw]
    int    **gwkpgr2ibz;   // q1 = R(q) + G_R, this varible stores all G_R: [nktotgw][3]
    int     *gwkpoptno;    // gwkp opterator No.: [nktotgw]
    int **gidxRevGW;       // reversed of gidxGW, elements in gidxRevGW with "-1" values mean out of emax_GW
                           // shape: [nkibrgw][ngtot_GW]
    int *gidxRevGWall;
    MPI_Win window_gidxRevGW; int *local_gidxRevGW_node;
    double *gabsGW;        // only need for q = 0, size of ngtot_GW, but only [0, npwGW[0]) is valid
    MPI_Win window_gabsGW; double *local_gabsGW_node;

    int *numq_of_ibrkall;
    MPI_Win window_nqofibr; int *local_nqofibr_node;
    int *Gindall; int *qindall; int *gindall;
    complex<double> *eigtauall; int *time_reve_kfullall;
    MPI_Win window_Gind; int *local_Gind_node;
    MPI_Win window_qind; int *local_qind_node;
    MPI_Win window_gind; int *local_gind_node;
    MPI_Win window_eigtau; complex<double> *local_eigtau_node;
    MPI_Win window_trk; int *local_trk_node;
    
    complex<double> *wggall = NULL;
    MPI_Win window_wgg; complex<double> *local_wgg_node;

    class Qclass {
        public:
        double Qptvec[3];  // Qpoint vector
        int Qpt[3];        // 0 <= Qpt[] <= 2 * NK_SC[] - 1
        int npwSC, npwSC_loc;
        int npwSC_loc_row, npwSC_loc_col;
        // Q + G = q + g, each G cooresponds one q and g
        int *Gind = NULL;  // size of npwSC, pointer, no memory
        int *qind = NULL;  // size of npwSC, pointer, no memory
        int *gind = NULL;  // size of npwSC, pointer, no memory
        complex<double> *eigtau = NULL;  // size of npwSC, stores e^{ig.tau}, pointer, no memory
        int *time_reve_kfull = NULL;     // size of npwSC, stores [+1/-1] x kfull, pointer, no memory
        
        int nkibrgw;               // inherit from parent class
        int *numq_of_ibrk = NULL;  // size of nkibrgw, stroes the num of same q for ith-ibrkpt
        int isInitial = 0;
        
        complex<double> *wgg = NULL;       // size of npwSC
        complex<double> *wfullggp = NULL;  // local: size of npwSC_loc_row x npwSC_loc_col

        complex<double> *head = NULL;      // only valid for Q = 0
        complex<double> *wing = NULL;
        complex<double> *cwing = NULL;
        MPI_Win window_wing;  complex<double> *local_wing_node;
        MPI_Win window_cwing; complex<double> *local_cwing_node;

        //method
        void ReadHeadWing(double omega0, int ncells, double *gabsGW); bool isReadHeadWing = false;

        //constructor
        Qclass();
        //destructor
        ~Qclass();
    };

    Qclass *Qcs = NULL;

    int numopt; // dimension of optcs
    operatorclass *optcs = NULL;
    
    //method
    void ReadGWOUTCAR(const char *gwoutcar);
    bool GetM();
    void GetNG();
    void Getng();
    bool WithinSphere(int i, int j, int k, double *qpoint);
    bool WithinSphereGW(int i, int j, int k, double *kpoint);
    void GetgidxGW();
    void MatchGWkpt(const double q0, const double q1, const double q2, int &ikfull, double &distance);
    void Doublex012toqg(const double x0, const double x1, const double x2,
                        vector<int> &qind, vector<int> &gind,
                        vector< complex<double> > &eigtau, vector<int> &time_reve_opt);
    void QpG2qpg();
    void ReadWxxxx(); bool isReadWxxxx = false;
    void ReadOneWFull(const int iQQ);
    void ReadOneWFullSub(const int iQQ, const int ikpt, int &idxbeg);
    void DelOneWFull(const int iQQ);
    void ReadAllWFull(); bool isReadAllWFull = false;
    
    //constructor
    wpotclass(waveclass *wvc_in); // default, dummy
    void Initial(); bool isInitial = false;
    
    //destructor
    ~wpotclass();
};

#endif
