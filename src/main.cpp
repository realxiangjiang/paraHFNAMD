#include <mpi.h>
#include <omp.h>
#include <iostream>
#include <string>
#include <vector>

#include <mkl.h>
#include <mkl_blacs.h>
#include <mkl_scalapack.h>

#include "const.h"
#include "fn.h"
#include "io.h"
#include "dynamics.h"
#include "tdcft.h"

using namespace std;

// random number
unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
default_random_engine generator(seed);
uniform_real_distribution<double> random01(0.0, 1.0);

// output default format
std::ios iosDefaultState(nullptr);

// mpi
int         mpi_split_num;
int         mpi_split_clr;

int         world_rk, world_sz, world_root;
bool        is_world_root;
MPI_Comm    world_comm;

int         sub_rank, sub_size, sub_root;
bool        is_sub_root;
MPI_Comm    group_comm;

// "node" for shard memory processes
int         node_rank, node_size, node_root;
int         node_color, node_number;
bool        is_node_root;
MPI_Comm    node_comm;

int         col_split_num;
int         col_split_clr;
int         col_rank, col_size, col_root;
bool        is_col_root;
MPI_Comm    col_comm;

// blacs
int         ctxt_world,   ctxt_group;
int         nprow_world,  nprow_group;
int         npcol_world,  npcol_group;
int         myprow_world, mypcol_world;
int         myprow_group, mypcol_group;
int         ctxt_only_group_root, group_root_prow, group_root_pcol;
int         ctxt_onecol, myprow_onecol, mypcol_onecol/*always zero*/;
int         nprow_onecol/*= nprow_group*/, npcol_onecol/*always one*/;
int         ctxt_only_onecol_root, onecol_root_prow, onecol_root_pcol/*always zero*/;

// vasp
int         vaspgam;
int         vaspGAM; // vaspgam && !is_make_spinor ? 1 : 0
int         vaspncl;

// work
string       resdir; // result directory
string      namddir; // namd directory, store NAC, eigen states/energies etc.

// auxiliary
bool        is_sub_calc;    // sub-calculation means there should be a combination task later.
int         laststru;       // this tag marks the last structure for calculation
bool        is_make_spinor; // if true, need construct spinor by wavefunction and SOC matrix
bool        is_paw_calc;    // if true, need deal with paw relevant
bool        is_bse_calc;    // if true, need calculate direct and/or exchange terms
int         totdiffspns;    // totdiffspns = 1 if numspns = 1 or Spins[0] = Spins[1]; else totdiffspns = 2

//*********************//
//    user settting    //
//*********************//
//
// basic
string      taskmod;   // task mode, now supported: fssh, dish, dcsh, bse
string      carrier;   // support: electron hole exciton
string      dftcode;   // current only VASP
   int      vaspver;   // VASP version 5.2.x, 5.4.x, 6.x
string      vaspbin;   // VASP bin: std gam ncl
string      pawpsae;   // for paw choose, ps("pseudo") or ae("all-electron")
   int      sexpikr;   // 0-calc Ekrij in momory, 1-store Ekrij in disk
   int      ispinor;   // 0-Bloch wavefunction. 1-spinor
   int      memrank;   // machine memory rank: 1-high, 2-medium, >3-low
   int      probind;   // # of processes(usually nodes) binded to deal with one crystal structure
string      runhome;   // all scf directories are listed in runhome
   int      ndigits;   // # of digits for scf directories, e.g. 00136 if ndigits = 5
   int      totstru;   // total # of sturctures in runhome, listed as {1..totstru} filling by ndigits
   int      strubeg;   // nac or/and bse are calculated in strutures [strubeg, struend]
   int      struend;   // if strubeg and struend set, totstru = struend - strubeg + 1
//
// basis sets space
       int  numspns;
vector<int> Spins;     // 0 or 1 or (0,0) or (0,1) or (1,1). 0/1 for spin up/down
       int  numkpts;
vector<int> Kpoints;   // 1, 2, 3 ...
vector<int> bandtop;   // band top    # for Hamiltion/NAC storage
vector<int> bandmax;   // dynamics basis sets band maximum #
vector<int> bandmin;   // dynamics basis sets band minimum #
vector<int> bandbot;   // band bottom # for Hamiltion/NAC storage
//                     // vector[0/1] for spin up/down (if have down)
       int  numbnds;   // bandmax - bandmin + 1
       int  totbnds;   // bandtop - bandbot + 1
// exciton
vector<int> condmax;   // conduction band maximum #
vector<int> condmin;   // conduction band minimum #
vector<int> valemax;   // valence    band maximum #
vector<int> valemin;   // valence    band minimum #
//
vector<int> exclude;   // excluded bands in range (bandmin, bandmax) or (valemin, valemax) or (condmin, condmax) 
//
// dynamics
string      hopmech;   // the hopping mechanism, now supported:
//                     // electron/hole: "nac"(default)
//                     //       exciton: "nacbse"(default), "nac"(no inter kpoints scattering)
double      dyntemp;   // the temperature of dynamics
double      iontime;   // ionic step time, POTIM in vasp
int         neleint;   // num of electronic intergral, interval time = iontime / neleint
string      intalgo;   // integral algorithm, Magnus or Euler
int         nsample;   // # of samples
int         ntrajec;   // # of trajectories
int         namdtim;   // namd time, [2, +oo)
int         nvinidc;   // non-vanishing initial dynamic coefficients
//
// exciton
double         epsilon;   // if < 0, use gw calculation, else screen Coulomb potential = 1/epsilon * 1/r
string         wpotdir;   // wpot directory path
double         iswfull;   // vasp: wxxxx.tmp or wfullxxxx.tmp
double         encutgw;   // set encutgw if there is ENCUTGW tag in vasp INCAR file
vector<int>    nkptsgw;   // nkpts[0,1,2] for gw
vector<int>    nkptssc;   // nkpts[0,1,2] for dynamics/supercell
double         gapdiff;   // manually set or calculate in wpot.cpp
vector<double> dynchan;   // dynamics channel, size of 4, for K^d, K^x, eph, radiation
int            lrecomb;   // recombination option: 0-not recombine, 1-nonradiative, 2-radiative, 3-both
// auxiliary
int            bsedone;   // = 1 if bse was calculated before and
                          // make sure tmpDirect/ and tmpExchange/ exist in calculation directory

int main(int argc, char *argv[]) {
    int required = /*MPI_THREAD_SERIALIZED;*/ MPI_THREAD_MULTIPLE;
    int provided;
    MPI_Init_thread(&argc, &argv, required, &provided);
    if(provided < required) {
        cerr << "Error: MPI does not provide the required thread support" << endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
        exit(1);
    }
    iosDefaultState.copyfmt(std::cout); // store the default c++ io

    // world comm
    MPI_Comm_dup(MPI_COMM_WORLD, &world_comm);
    MPI_Comm_rank(world_comm, &world_rk);
    MPI_Comm_size(world_comm, &world_sz);
    world_root = 0;
    is_world_root = false;
    if(world_rk == world_root) is_world_root = true;
    
    /*cout << "MPI_MAX_LIBRARY_VERSION_STRING = " << MPI_MAX_LIBRARY_VERSION_STRING << endl;
    cout << "MPI_MAX_PROCESSOR_NAME = " << MPI_MAX_PROCESSOR_NAME << endl;
    while(1);*/

    // Read "input" file
    ReadInput("input");
    
    // group/sub comm
    //mpi_split_num = world_sz / probind;
    //mpi_split_clr = world_rk % mpi_split_num; 
    DeterminCommNumClr(probind, world_rk, world_sz, mpi_split_num, mpi_split_clr);
                                              // each process has an unchanged global color number
                                              // processes are divided by mpi_split_num groups signed by mpi_split_clr
                                              // processes in one group have the same color(mpi_split_clr)
    probind = world_sz / mpi_split_num;       // correct probind
    MPI_Comm_split(world_comm, mpi_split_clr, world_rk, &group_comm);
    MPI_Comm_rank(group_comm, &sub_rank);
    MPI_Comm_size(group_comm, &sub_size);
    sub_root = 0;
    is_sub_root = false;
    if(sub_rank == sub_root) is_sub_root = true;
    
    // node comm, for shared memory processes
    MPI_Comm_split_type(world_comm, MPI_COMM_TYPE_SHARED, world_rk,
                        MPI_INFO_NULL, &node_comm);
    MPI_Comm_rank(node_comm, &node_rank);
    MPI_Comm_size(node_comm, &node_size);
    node_root = 0;
    is_node_root = false;
    if(node_rank == node_root) is_node_root = true;
    node_color  = world_rk / node_size;
    node_number = world_sz / node_size;
    if(world_sz % node_size) { CERR << "Fatal ERROR: # of cpus should be divisible by # of nodes" << endl; EXIT(1); }
    
    // "world" blacs
    RowColSquareSet(world_sz, nprow_world, npcol_world);
    blacs_get(&iZERO, &iZERO, &ctxt_world);
    blacs_gridinit(&ctxt_world, "R", &nprow_world, &npcol_world);
    blacs_pcoord(&ctxt_world, &world_rk, &myprow_world, &mypcol_world);
    
    // "group" blacs
    int *group_process_id = new int[sub_size];
    MPI_Allgather(&world_rk, 1, MPI_INT, group_process_id, 1, MPI_INT, group_comm);
    RowColSquareSet(sub_size, nprow_group, npcol_group);
    int *group_process_map = new int[sub_size]; // same elements as group_process_id, but with column-major storage
    for(int jcol = 0; jcol < npcol_group; jcol++) {
        for(int irow = 0; irow < nprow_group; irow++) group_process_map[irow + jcol * nprow_group] = group_process_id[irow * npcol_group + jcol];
    }
    blacs_get(&iZERO, &iZERO, &ctxt_group);
    blacs_gridmap(&ctxt_group, group_process_map, &nprow_group, &nprow_group, &npcol_group);
    blacs_pcoord(&ctxt_group, &sub_rank, &myprow_group, &mypcol_group);
    COUT << "There are total " << world_sz << " processes which are divided into " << mpi_split_num << " group(s)." << endl;
    COUT << "The processes distribution is:" << endl;
    MPI_Barrier(world_comm);
    for(int irk = 0; irk < world_sz; irk++) {
        if(world_rk == irk && world_rk == group_process_id[0]) {
            cout << "Group " << setw(3) << setfill(' ') << mpi_split_clr << ", total " << flush;
            cout << nprow_group << " x " << npcol_group << " = " << sub_size << ": "<< flush;
            for(int i = 0; i < sub_size; i++) cout << group_process_id[i] << ' '; cout << endl;
        }
        MPI_Barrier(world_comm);
    }
    COUT << endl;
    MPI_Barrier(world_comm);

    // "only group root" blacs
    group_root_prow = 0; group_root_pcol = 0;
    blacs_get(&iZERO, &iZERO, &ctxt_only_group_root);
    blacs_gridmap(&ctxt_only_group_root, group_process_map + group_root_prow + 
                                                             group_root_pcol * nprow_group, &iONE, &iONE, &iONE);
    MPI_Barrier(world_comm);

    // column comm: sometimes we need divide each column as a comm
    int *ncols_of_eachgroup = new int[mpi_split_num]; // first should record # of columns in each group comm
    if(is_sub_root && (!is_world_root)) {
        MPI_Send(&npcol_group, 1, MPI_INT, world_root, mpi_split_clr, world_comm);
    }
    if(is_world_root) { // receive in world root
        MPI_Status mpi_status;
        ncols_of_eachgroup[0] = npcol_group;
        for(int ip = 1; ip < mpi_split_num; ip++)
        MPI_Recv(ncols_of_eachgroup + ip, 1, MPI_INT,
                 ip < world_sz % mpi_split_num ?
                 ip * (world_sz / mpi_split_num + 1) :
                 (world_sz % mpi_split_num) * (world_sz / mpi_split_num + 1)
                 + (ip - world_sz % mpi_split_num) * (world_sz / mpi_split_num),
                 ip, world_comm, &mpi_status);
    }
    MPI_Bcast(ncols_of_eachgroup, mpi_split_num, MPI_INT, world_root, world_comm);
    int *sum_ncols_of_eachgroup = new int[mpi_split_num]();
    for(int igrp = 1; igrp < mpi_split_num; igrp++) sum_ncols_of_eachgroup[igrp] = 
                                                    sum_ncols_of_eachgroup[igrp - 1] + ncols_of_eachgroup[igrp - 1];
    col_split_num = sum_ncols_of_eachgroup[mpi_split_num - 1] + ncols_of_eachgroup[mpi_split_num - 1];
    col_split_clr = sum_ncols_of_eachgroup[mpi_split_clr] + mypcol_group; // set column color
    MPI_Comm_split(world_comm, col_split_clr, world_rk, &col_comm);
    MPI_Comm_rank(col_comm, &col_rank);
    MPI_Comm_size(col_comm, &col_size);
    col_root = 0;
    is_col_root = false;
    if(col_rank == col_root) is_col_root = true;

    // "onecol" blacs
    blacs_get(&iZERO, &iZERO, &ctxt_onecol);        // MUST be "0" here
    blacs_gridmap(&ctxt_onecol, group_process_map + (0 + mypcol_group * nprow_group), &nprow_group, &nprow_group, &iONE);
    blacs_pcoord(&ctxt_onecol, &col_rank, &myprow_onecol, &mypcol_onecol);
    nprow_onecol = nprow_group; npcol_onecol = 1;
    MPI_Barrier(world_comm);
    
    // "only_onecol_root" blacs
    onecol_root_prow = 0; // could be other value, usually be zero
    onecol_root_pcol = 0; // MUST be zero
    blacs_get(&iZERO, &iZERO, &ctxt_only_onecol_root);
    blacs_gridmap(&ctxt_only_onecol_root, group_process_map + onecol_root_prow + 
                                                              mypcol_group * nprow_group, &iONE, &iONE, &iONE);
    MPI_Barrier(world_comm);
    
    /*COUT << "world, mpi_group, mpi_one_col, blacs_group, blacs_one_col" << endl;
    for(int irk = 0; irk < world_sz; irk++) {
        if(world_rk == irk) {
            cout << world_rk << ' ' << sub_rank << '/' << mpi_split_clr << ' ' 
                 << col_rank << '/' << col_split_clr << ' '
                 << '(' << myprow_group << ',' << mypcol_group << ") "
                 << '(' << myprow_onecol << ',' << mypcol_onecol << ") " << endl;
        }
        MPI_Barrier(world_comm);
    } while(1);*/

    // task
    CreatWorkDir(world_rk, world_sz, world_comm);
    double tstart, tend; tstart = omp_get_wtime();
    if(is_world_root) WriteOutput(0); MPI_Barrier(world_comm);
    if(taskmod == "fssh" || taskmod == "dish" || taskmod == "dcsh") {
        DynamicsMatrixConstruct();
        if(is_sub_calc && (laststru == struend)); // should do phase correction
        if(totstru > 2 && ( !is_sub_calc || (laststru == struend) )) BuildAllTDMatSplineCoeff();
        if(totstru > 2 && namdtim > 1 && nsample > 0) RunDynamics();
    }
    else if(taskmod == "bse") {
        OnlyBSECalc();
    }
    else if(taskmod == "spinor") {
        if(dftcode == "vasp" && vaspbin == "std" && ispinor == 1) OnlySpinorCalc();
    }

    MPI_Comm_free(&group_comm);
    blacs_exit(&iONE);

    delete[] ncols_of_eachgroup; delete[] sum_ncols_of_eachgroup;
    delete[] group_process_id; delete[] group_process_map;
    MPI_Barrier(world_comm);
    COUT << endl << " ================= Job Finished =================" << endl;
    tend = omp_get_wtime(); if(is_world_root) WriteOutput(-1, to_string((int)(tend - tstart)).c_str());
    MPI_Barrier(world_comm);
    MPI_Finalize();

    return 0;
}
