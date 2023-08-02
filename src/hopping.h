#ifndef hopping_h
#define hopping_h

#include <mpi.h>
#include <omp.h>
#include <complex>
#include <algorithm> // max

#include "const.h"
#include "fn.h"
#include "math.h"

using namespace std;

void NormalizeProbability(double *probmat, const int nstates, const bool is_reset_diag = false);
void FSSHprob(double *probmat, complex<double> *vmat,
              const int nstates,
              complex<double> **c_onsite, complex<double> **c_midsite,
              const complex<double> *coeff);
void DetailBalanceProb(double *probmat, const complex<double> *vmat,
                       const int nstates, const double temp);
void PopuUpdateFSSH(const int nstates, const complex<double> *coeff,
                    double *population, // update in this routine
                    complex<double> **c_onsite, complex<double> **c_midsite,
                    double *probmat, complex<double> *vmat, const double temp);

#endif
