#ifndef dsh_h
#define dsh_h

#include <mpi.h>
#include <omp.h>
#include <complex>
#include <fstream>
#include <algorithm> // max, shuffle, fill, fill_n
#include <cstdio>    // remove
#include <cassert>

#include "const.h"
#include "fn.h"
#include "math.h"

using namespace std;

void ReadAllEnergies(double *allenergies,
                     vector<int> &allispns, vector<int> &allikpts,
                     const int nstates, const int *ibndstart, const int nbnds,
                     const int totnspns, const int totnkpts, const int totnbnds);
void CalcDecoRate(double *decorate,
                  vector<int> &allispns, vector<int> &allikpts,
                  const int nstates, const int *ibndstart, const int dimC, const int dimV,
                  const int totnspns, const int totnkpts, const int totnbnds);
void SetIniCurStates(const double *population, const int nstates,
                     const int ntrajs_loc_col, int *currentstates);
void WritePopuByCurStates(ofstream &otf, const int *currentstates, 
                          const int nstates, const int ntrajs_loc_col);
void CalcDecoTime(const int nstates, const int ntrajs, 
                  const int ndim_loc_row, const int ntrajs_loc_col,
                  const double *decorate, const complex<double> *coeff,
                  double *decotime);
void WhichToDeco(double *decomoments, int *whiches,
                 const int nstates, const int ndim_loc_row, const int ntrajs_loc_col, 
                 const double *decotime);
void Projector(const int *whiches, int *currentstates, const double temp,
               const int nstates, const int ndim_loc_row, const int ntrajs_loc_col,
               complex<double> *coeff, complex<double> **c_onsite);
void DecoCorrectCoeff(const int nstates, const int ndim_loc_row, const int ntrajs_loc_col,
                      const int *currentstates, const double *decorate_col,
                      complex<double> *coeff);
void UpdateCurrentStates(const int nstates, const int ntrajs_loc_col,
                         int *currentstates,
                         complex<double> **c_onsite, complex<double> **c_midsite,
                         const complex<double> *coeff, const double temp);
void MergeAllPopu(const int nstates, const int ntraj_bcks, const int begtime);

#endif
