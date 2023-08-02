#ifndef wave_h
#define wave_h

#include <omp.h>
#include <mpi.h>
#include <cstdio>    // remove
#include <string>
#include <unistd.h>  // access
#include <fstream>
#include <cassert>   // assert

#include "const.h"
#include "fn.h"
#include "wave_base.h"
#include "paw.h"
#include "soc.h"
#include "bse.h"

void LoadStates2WVC(waveclass &wvc, const int is_exclude_bands = IS_EXCLUDE_BANDS);
TIMECOST WVCBasicProcess(waveclass *&wvc, socclass *soccls, const int num, bool isTmp = false);
void WriteEnergyDiff(waveclass *wvc, const int dirnum, const double gapadd);
TIMECOST WVCAdvanProcess(waveclass *&wvc, excitonclass *extc, const int num);
void CalcQprojKptsForAllElements(waveclass &wvc, pawpotclass *&pawpots, const int numelem);
void Calc_psmApsn(waveclass &wvcL, waveclass &wvcR,    // wvcL/R: waveclass left/right represents ps_m/ps_n
                  complex<double> *Amn, const int lda, // input and output
                  const complex<double> *Aij,          // input
                  const int sigmaL, const int sigmaR,  // spin channels of wvcL/R
                  const int nstates);

#endif
