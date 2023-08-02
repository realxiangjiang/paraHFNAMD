#ifndef nac_h
#define nac_h

#include <omp.h>
#include <mpi.h>
#include <complex>
#include <cmath>
#include <algorithm> // max
#include <cassert>

#include "const.h"
#include "fn.h"
#include "math.h"

#include "wave_base.h"

void Calc_psmApsn(const complex<double> alpha,
                  waveclass &wvcL, waveclass &wvcR,    // wvcL/R: waveclass left/right represents ps_m/ps_n
                  waveclass &wvcCore,                  // wvcCore: for Aij
                  complex<double> *Amn, const int lda, // input and output
                  const int Aij_ID,                    // input, listed in pawpotclass
                  const int sigmaL, const int sigmaR,  // spin channels of wvcL/R
                  const int nstates);
void Calc_psmApsn(const complex<double> alpha,
                  waveclass &wvcL, waveclass &wvcR,    // wvcL/R: waveclass left/right represents ps_m/ps_n
                  waveclass &wvcCore,                  // wvcCore: for Aij
                  complex<double> *Amn, const int lda, // input and output
                  const int Aij_ID,                    // input, listed in pawpotclass
                  const int sigmaL, const int sigmaR,  // spin channels of wvcL/R
                  const int nst_row, const int i_beg_row, 
                  const int nst_col, const int i_beg_col, const int ntotst);
void PsWaveDot(const complex<double> alpha, waveclass &wvcL, waveclass &wvcR, // calc alpha x <ps_m|ps_n>
               const int nst_row, const int i_beg_row,
               const int nst_col, const int i_beg_col, const int ntotst,
               const int beta, complex<double> *res);
void CalcNAC(double dt, waveclass &wvcC, waveclass &wvcN,
             const int nst_row, const int i_beg_row,
             const int nst_col, const int i_beg_col,
             complex<double> *res, const char *outfilename, const bool is_correction = true);

#endif
