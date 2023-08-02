#ifndef dynamics_h
#define dynamics_h

#include <mpi.h>
#include <omp.h>
#include <complex>
#include <fstream>
#include <algorithm> // min, max
#include <cassert>

#include "const.h"
#include "fn.h"
#include "math.h"

#include "io.h"
#include "wave_base.h"
#include "wave.h"
#include "paw.h"
#include "soc.h"
#include "nac.h"
#include "wpot.h"
#include "bse.h"
#include "tdcft.h"
#include "hopping.h"
#include "dsh.h"

using namespace std;

TIMECOST InterTimeAction(waveclass &wvc1, waveclass &wvc2, 
                         const double dt, const int num);
void PhaseCorrection(const int stru_beg, const int tot_stru, waveclass *wvc);
void MatrixPhaseModify(const complex<double> *begphase, const char *filename, waveclass *wvc,
                       const bool modifyleft = true, const bool modifyright = true);
void DynamicsMatrixConstruct();
void OnlyBSECalc();
void OnlySpinorCalc();
void CheckBasisSpace(vector<int> *&allbands, int &nstates, int &nspns, int &nkpts, int &dimC, int &dimV,
                     vector<int> &allispns, vector<int> &allikpts, int *&ibndstart,
                     int &totnspns, int &totnkpts, int &totnbnds);
void RunDynamics();

#endif
