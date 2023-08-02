#ifndef io_h
#define io_h

#include <mpi.h>
#include <omp.h>

#include <fstream>
#include <string>
#include <sstream>
#include <algorithm>   // transform copy min find
#include <unistd.h>    // access rmdir
#include <sys/stat.h>  // mkdir 
#include <thread>      // >= c++11
#include <cassert>
#include <ctime>

#include "const.h"
#include "fn.h"
#include "wave_base.h"
#include "math.h"

using namespace std;

void ReadInput(const char *input = "input");
void WriteOutput(const int flag, const char *info = NULL, const int inNKPTS = numkpts,
                 const char *output = (resdir + "/output").c_str());
void CreatWorkDir(int rk, int sz, MPI_Comm &comm);
void CheckBasisSets(waveclass &wvc);
bool CreatNAMDdir(waveclass &wvc);
void ReadInfoTmp(int &nspns, int &nkpts, int &dimC, int &dimV, const char *infofile = (namddir + "/.info.tmp").c_str());
void ReadInfoTmp1(vector<int> &readspns, vector<int> &readkpts, vector<int> &readbnds,
                  const char* infofile = (namddir + "/.info.tmp").c_str());
void ReadInfoTmp2(vector<int> &readspns, vector<int> &readkpts,
                  vector<int> &readcbds, vector<int> &readvbds,
                  const char* infofile = (namddir + "/.info.tmp").c_str());
void CheckIniconFile(vector<int> *&allbands, const char *inicon = "inicon");
void WritePdvec(ofstream &otf, const int dim, const int dim_loc,
                const double *vec, double *fullvec,
                const bool is_new_line = true);
void WritePzvec(ofstream &otf, const int dim, const int dim_loc, 
                const complex<double> *vec, complex<double> *fullvec,
                const bool is_new_line = true);

#endif
