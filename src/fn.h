#ifndef fn_h
#define fn_h

#include <mpi.h>
#include <omp.h>
#include <cstdlib> // exit
#ifndef  EXIT
#define  EXIT(ExitCode) MPI_Barrier(world_comm); exit(ExitCode)
#endif
#include <iostream>
#ifndef  COUT
#define  COUT if(is_world_root) std::cout
#endif
#ifndef  CERR
#define  CERR if(is_world_root) std::cerr
#endif
#ifndef  Cout
#define  Cout if(is_sub_root) std::cout
#endif
#ifndef  Cerr
#define  Cerr if(is_sub_root) std::cerr
#endif
#include <cmath>       // round
#include <fstream>
#include <string>
#include <sstream>
#include <vector>
#include <iomanip>     // setw setfill
#include <numeric>     // iota
#include <algorithm>   // stable_sort find_if fill_n

#include "const.h"

using namespace std;

bool FFTch1(int n);
void FFTch3(int ngf[]);

void DeterminCommNumClr(const int nbinding, const int rk, const int sz, int &num, int &clr);
void RowColSquareSet(const int area, int &nrow, int &ncol); // nrow >= ncol
void ColRowSquareSet(const int area, int &nrow, int &ncol); // ncol >= nrow
int  BlacsIdxloc2glb(const int locidx, const int n, const int nb, const int iproc, const int nproc, const int psrc = 0);
void BlacsIdxglb2loc(const int glbidx, int &iproc, int &locidx,
                     const int psrc, const int n, const int nb, const int nproc);
void BlacsBegidxBlocklen(const int iproc, const int nproc, const int n, const int nb,
                         vector<int> &locbeg, vector<int> &glbbeg, vector<int> &bcklen,
                         const int psrc = 0);

string Int2Str(const int input, const int strsize = ndigits, const char fillc = '0');
string WholeFile2String(ifstream &inf);
long StringSplit(string &ins, string &ots, string words, bool is_contain = true);
vector<string> StringSplitByBlank(string &ins);
int StringMatchCount(string &ins, string target);

int IdxNat1toSym1(const int ii, const int NN); // nature[0, 1, ..., n) to symmetry(-(n+1)/2 , n/2+1)
int IdxSym1toNat1(const int ii, const int NN); // symmetry(-(n+1)/2 , n/2+1) to nature[0, 1, ..., n)
int IdxNat3toNat1(const int ii, const int jj, const int kk, const int NN0, const int NN1, const int NN2);
int IdxSym3toNat1(const int ii, const int jj, const int kk, const int NN0, const int NN1, const int NN2);
void IdxNat1toNat3(const int in, int &out0, int &out1, int &out2, const int NN0, const int NN1, const int NN2);
void IdxNat1toSym3(const int in, int &out0, int &out1, int &out2, const int NN0, const int NN1, const int NN2);
int MatchKptDiff(int ikpt1, int ikpt2, int NK[]);

void BlockCyclicToMpiFFTW3d(const complex<double> *in, const int totin,
                            complex<double> *out,
                            const int nin[], const int nout[],
                            const int loc_nout0, const int loc_out0_start, 
                            const int *idxin);
void MpiFFTW3dToBlockCyclic(const complex<double> *in,
                            complex<double> *out, const int totout,
                            const int nin[], const int nout[],
                            const int loc_nin0, const int loc_in0_start,
                            const int *idxout);
void Compacted2Cubiod(const int totin, const int *idxin,
                      const int nin[], const complex<double> *in,
                      const int nout[],      complex<double> *out);
void Cubiod2Compacted(const int nin[], const complex<double> *in,
                      const int totout, const int *idxout, 
                      const int nout[], complex<double> *out);

template <typename T>
void MpiWindowShareMemoryInitial(const size_t datasize, T *&data, T *&local_data_node, MPI_Win &window_data,
                                 const int malloc_root = node_root) {
    const size_t local_data_size = (node_rank == malloc_root ? datasize : 0);
    MPI_Aint winsize;
    int      windisp;
    int *model, flag;
    MPI_Win_allocate_shared(local_data_size * sizeof(T), sizeof(T), MPI_INFO_NULL,
                            node_comm, &local_data_node, &window_data);
    MPI_Win_get_attr(window_data, MPI_WIN_MODEL, &model, &flag);
    if(flag != 1) { CERR << "ReadKptvecNpw: Attribute MPI_WIN_MODEL not defined" << endl; exit(1); }
    else if(*model != MPI_WIN_UNIFIED) { CERR << "Memory model is NOT MPI_WIN_UNIFIED " << endl; exit(1); }
    data = local_data_node;
    if(node_rank != malloc_root) {
        MPI_Win_shared_query(window_data, malloc_root, &winsize, &windisp, &data);
    }
    MPI_Win_fence(0, window_data);
    // now all data pointers should point to copy of malloc_root(usually node_root)
    MPI_Barrier(node_comm);
    return;
}

template <typename T1, typename T2> // coordinates T1, basis T2
void XYZtoRTP(T1 x, T1 y, T1 z, T2 *a[], T1 &rabs, T1 &theta, T1 &phi, T1 epsilon = 1e-12) { 
    // output: rabs = |x * a1 + y * a2 + z * a3|, theta = arccos(r2/rabs), phi = arctan(r1/r0)
    T1 r[3];
    for(int j = 0; j < 3; j++) r[j] = x * a[0][j] + y * a[1][j] + z * a[2][j];
    rabs = sqrt(r[0] * r[0] + r[1] * r[1] + r[2] * r[2]);
    if(rabs < epsilon) { rabs = 0.0; theta = 0.0; phi = 0.0; }
    else {
        theta = acos(r[2] / rabs);
        phi = atan2(r[1], r[0]);
        if(phi < 0) phi += 2.0 * M_PI;
    }
    return;
}

template <typename T>
void RTPtoXYZ(T rabs, T theta, T phi, T &x, T &y, T &z) {
    x = rabs * sin(theta) * cos(phi);
    y = rabs * sin(theta) * sin(phi);
    z = rabs * cos(theta);
    return;
}

template <typename T>
void PointerSwap(T *&a, T *&b) { T *c = a; a = b; b = c; return; }

template <typename T>
void DivideTot2Loc(const T tot, const T rk, const T sz, T &locbeg, T &loclen) {
    loclen = tot / sz      + (rk < tot % sz ?  1 : 0); 
    locbeg = tot / sz * rk + (rk < tot % sz ? rk : tot % sz);
    return;
}

template <typename T>
void BcastMultiArrays(T **data, const int ndata, const int *datasize, const MPI_Datatype datatype,
                      int rk, int sz, MPI_Comm &comm) {
    MPI_Barrier(comm);
    for(int rt = 0; rt < sz; rt++) {
        for(int ii = rt; ii < ndata; ii += sz) MPI_Bcast((T*)data[ii], datasize[ii], datatype, rt, comm);
    }
    MPI_Barrier(comm);
    return;
}

template <typename T>
vector<int> Argsort(const T *v, const int n) {
    vector<int> idx(n);
    iota(idx.begin(), idx.end(), 0);
    stable_sort(idx.begin(), idx.end(),
                [&v](int i, int j) { return (v[i] < v[j]); });
    return idx;
}

template <typename T>
int FindIndex(const vector<T> &v, const T val) {
    auto it = find(v.begin(), v.end(), val);
    if(it == v.end()) return -1;
    else return it - v.begin();
}

#endif
