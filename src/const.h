#ifndef const_h
#define const_h

#ifndef IS_SPINOR
#define IS_SPINOR 1
#endif

#ifndef NOT_SPINOR
#define NOT_SPINOR 0
#endif

#ifndef IS_VASPNCL
#define IS_VASPNCL 1
#endif

#ifndef NOT_VASPNCL
#define NOT_VASPNCL 0
#endif

#ifndef IS_VASPGAM
#define IS_VASPGAM 1
#endif

#ifndef NOT_VASPGAM
#define NOT_VASPGAM 0
#endif

#ifndef NO_NEED_REALC
#define NO_NEED_REALC 0
#endif

#ifndef NEED_REALC
#define NEED_REALC 1
#endif

#include <mpi.h>
#include <omp.h>
#include <iostream>
#include <string>
#include <vector>
#include <complex>
#include <random>

using namespace std;

typedef double TIMECOST;

const int    iZERO = 0;
const int    iONE  = 1; 
const int    iMONE = -1;
const complex<float>  iu_f(0.0, 1.0);
const complex<double> iu_d(0.0, 1.0);
const double hbar = 0.6582119514; // eV * fs
const double kb = 8.6173857e-5;   // eV / K
const double rytoev = 13.605826;  // 1 Ry in Ev
const double autoa = 0.529177249; // 1 a.u. in Angstroem

// random number
extern unsigned seed;
extern default_random_engine generator;
extern uniform_real_distribution<double> random01;

// paw
const int NPSQNL = 100;   // no. of data for projectors in reciprocal space
const int NPSRNL = 100;   // no. of data for projectors in real space
const int NCINPOTCAR = 5; // no. columns in POTCAR for data
const int NL_NPSQNL = NPSQNL / NCINPOTCAR + (NPSQNL % NCINPOTCAR ? 1 : 0); // no. of lines for projectors (reciprocal)
const int NL_NPSRNL = NPSRNL / NCINPOTCAR + (NPSRNL % NCINPOTCAR ? 1 : 0); // no. of lines for projectors (real)

// output default format
extern std::ios iosDefaultState;

// mpi
extern int      mpi_split_num;
extern int      mpi_split_clr;
extern int      world_rk, world_sz, world_root;
extern bool     is_world_root;
extern MPI_Comm world_comm;
extern int      sub_rank, sub_size, sub_root;
extern bool     is_sub_root;
extern MPI_Comm group_comm;
extern int      node_rank, node_size, node_root;
extern int      node_color, node_number;
extern bool     is_node_root;
extern MPI_Comm node_comm;
extern int      col_split_num;
extern int      col_split_clr;
extern int      col_rank, col_size, col_root;
extern bool     is_col_root;
extern MPI_Comm col_comm;

// blacs
const int MB_ROW = 16; // default 64 (BUT may slower than smaller values!), other values for test
const int NB_COL = 16; // default 64, should equal to NB_COL
extern int      ctxt_world,   ctxt_group;  
extern int      nprow_world,  nprow_group;
extern int      npcol_world,  npcol_group;
extern int      myprow_world, mypcol_world;
extern int      myprow_group, mypcol_group;
extern int      ctxt_only_group_root, group_root_prow, group_root_pcol;
extern int      ctxt_onecol, myprow_onecol, mypcol_onecol/*always zero*/;
extern int      nprow_onecol/*= nprow_group*/, npcol_onecol/*always one*/;
extern int      ctxt_only_onecol_root, onecol_root_prow, onecol_root_pcol/*always zero*/;

// vasp
extern int vaspgam;
extern int vaspGAM; // vaspgam && !is_make_spinor ? 1 : 0
extern int vaspncl;

// work
extern string  resdir; // result directory
extern string namddir; // namd directory, store NAC, eigen states/energies etc.

// auxiliary
extern bool   is_sub_calc;      // sub-calculation means there should be a combination task later.
extern int    laststru;         // this tag marks the last structure for calculation
extern bool   is_make_spinor;   // if true, need construct spinor by wavefunction and SOC matrix
extern bool   is_paw_calc;      // if true, need deal with paw relevant
extern bool   is_bse_calc;      // if true, need calculate direct and/or exchange terms
extern int    totdiffspns;      // totdiffspns = 1 if numspns = 1 or Spins[0] = Spins[1]; else totdiffspns = 2
const  int    bckntrajs = 8192; // for dish/dcsh, determine the block lenght of trajectories

//*********************//
//    user settting    //
//*********************//
//
// basic
extern string taskmod;   // task mode, now supported: fssh, dish, dcsh, bse
extern string carrier;   // support: electron hole exciton
extern string dftcode;   // current only VASP
extern int    vaspver;   // VASP version 5.2.x, 5.4.x, 6.x
extern string vaspbin;   // VASP bin: std gam ncl
extern string pawpsae;   // for paw choose, ps("pseudo") or ae("all-electron")
extern int    sexpikr;   // 0-calc Ekrij in momory, 1-store Ekrij in disk
extern int    ispinor;   // 0-Bloch wavefunction. 1-spinor
extern int    memrank;   // machine memory rank: 1-high, 2-medium, >3-low
extern int    probind;   // # of processes(usually nodes) binded to deal with one crystal structure
extern string runhome;   // all scf directories are listed in runhome
extern int    ndigits;   // # of digits for scf directories, e.g. 00136 if ndigits = 5
extern int    totstru;   // total # of sturctures in runhome, listed as {1..totstru} filling by ndigits
extern int    strubeg;   // nac or/and bse are calculated in strutures [strubeg, struend]
extern int    struend;   // if strubeg and struend set, totstru = struend - strubeg + 1
//
// basis sets space
extern int         numspns;
extern vector<int> Spins;     // 0 or 1 or (0,0) or (0,1) or (1,1). 0/1 for spin up/down
extern int         numkpts;
extern vector<int> Kpoints;   // 1, 2, 3 ...
extern vector<int> bandtop;   // band top    # for Hamiltion/NAC storage
extern vector<int> bandmax;   // dynamics basis sets band maximum #
extern vector<int> bandmin;   // dynamics basis sets band minimum #
extern vector<int> bandbot;   // band bottom # for Hamiltion/NAC storage
//                            // vector[0/1] for spin up/down (if have down)
extern int         numbnds;   // bandmax - bandmin + 1
extern int         totbnds;   // bandtop - bandbot + 1
// exciton
extern vector<int> condmax;   // conduction band maximum #
extern vector<int> condmin;   // conduction band minimum #
extern vector<int> valemax;   // valence    band maximum #
extern vector<int> valemin;   // valence    band minimum #
extern vector<int> exclude;   // excluded bands in range (bandmin, bandmax) or (valemin, valemax) or (condmin, condmax) 
//
// dynamics
extern string      hopmech;   // the hopping mechanism, now supported:
//                            // electron/hole: "nac"(default), "nacsoc"(soc can offer inter-spin channel)
//                            //       exciton: "nacbse"(default), "nac"(no inter kpoints scattering)
extern double      dyntemp;   // the temperature of dynamics
extern double      iontime;   // ionic step time, POTIM in vasp
extern int         neleint;   // num of electronic intergral, interval time = iontime / neleint
extern string      intalgo;   // integral algorithm, Magnus or Euler
extern int         nsample;   // # of samples
extern int         ntrajec;   // # of trajectories
extern int         namdtim;   // namd time, [2, +oo)
extern int         nvinidc;   // non-vanishing initial dynamic coefficients
//
// exciton
extern double         epsilon;   // if < 0, use gw calculation, else screen Coulomb potential = 1/epsilon * 1/r
extern string         wpotdir;   // wpot directory path
extern double         iswfull;   // vasp: wxxxx.tmp or wfullxxxx.tmp
extern double         encutgw;   // set encutgw if there is ENCUTGW tag in vasp INCAR file
extern vector<int>    nkptsgw;   // nkpts[0,1,2] for gw
extern vector<int>    nkptssc;   // nkpts[0,1,2] for dynamics/supercell
extern double         gapdiff;   // manually set or calculate in wpot.cpp
extern vector<double> dynchan;   // dynamics channel, size of 4, for K^d, K^x, eph, radiation
extern int            lrecomb;   // recombination option: 0-not recombine, 1-nonradiative, 2-radiative, 3-both
// auxiliary
extern int            bsedone;   // = 1 if bse was calculated before and
                                 // make sure tmpDirect/ and tmpExchange/ exist in calculation directory

#endif
