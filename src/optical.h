#ifndef optical_h
#define optical_h

#include <mpi.h>
#include <omp.h>
#include <cmath>

#include "const.h"
#include "fn.h"
#include "math.h"
#include "wave_base.h"
#include "nac.h"

void TDM_PSPart(waveclass &wvc,
                const int cb_start, const int ncb,
                const int vb_start, const int nvb,
                complex<double> *tdm);
void TDM_AEPart(waveclass &wvc,
                const int cb_start, const int ncb,
                const int vb_start, const int nvb,
                complex<double> *tdm);
void TDM_over_dE(waveclass &wvc, complex<double> *tdm);
void CalcCVtdm(waveclass &wvc, const int dirnum,
               complex<double> *cvtdm, complex<double> *cvtdm_full);

#endif
