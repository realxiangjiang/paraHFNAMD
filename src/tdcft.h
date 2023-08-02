#ifndef tdcft_h
#define tdcft_h

#include <mpi.h>
#include <omp.h>
#include <complex>
#include <cstdio>    // rename, remove
#include <algorithm> // max
#include <unistd.h>  // access
#include <cassert>

#include "const.h"
#include "fn.h"
#include "math.h"
#include "io.h"

using namespace std;

void AddExtraSplineCoeff(const size_t totsize, const double time_interval,
                         const double *first, const double *last,
                         double *extra_first_c0, double *extra_first_c1,
                         double *extra_last_c0,  double *extra_last_c1);
void InterpolationForTDMat(const int ntimes, const double time_interval,
                           const char *mat_dir_name,
                           const long istart, const long nelements,
                           const bool is_istart_suffix = true,
                           const bool is_add_extra_splinecoeff = false);
void AddExtraSplineCoeff(const int ntimes, const size_t totsize,
                         const double time_interval, const char *dirname);
void CombineAllSplineCoeff(const int ntimes, const int dim_row, const int dim_col,
                           const char *dirname,
                           const int num_of_matrix = 1, const bool is_add_extra_splinecoeff = false);
void BuildAllTDMatSplineCoeff();
void Mat_AtoOmega(const int matsize, const double h,
                  const complex<double> *a0, const complex<double> *a1,
                  const complex<double> *a2, const complex<double> *a3,
                  complex<double> *om_oih);
void ReadDiag(const char *filename, const int itime,
              vector<int> &allispns, vector<int> &allikpts, 
              const int *ibndstart, const int nbnds,
              const int totnspns, const int totnkpts, const int totnbnds,
              const complex<double> alpha, complex<double> *locmat);
void ReadNAC(const char *filename, const int itime,
             vector<int> &allispns, vector<int> &allikpts, 
             const int *ibndstart, const int nbnds,
             const int totnspns, const int totnkpts, const int totnbnds,
             const complex<double> alpha, complex<double> *locmat, const bool isConj);
void ReadNACbySK(const char *ccfile, const char *vvfile,
                 const int itime, const int start_idx,
                 const int nspns, const int nkpts, const int dimC, const int dimV,
                 const complex<double> alpha, complex<double> *locmat);
void ReadCtoVNAC(const char *filename, const int itime,
                 const int nspns, const int nkpts, const int dimC, const int dimV,
                 const complex<double> alpha, complex<double> *locmat);
void ReadOnsiteC(const int t_ion, const int matsize,
                 const int nspns, const int nkpts, const int dimC, const int dimV,
                 vector<int> &allispns, vector<int> &allikpts, const int *ibndstart,
                 const int totnspns, const int totnkpts, const int totnbnds,
                 complex<double> **c_onsite);
void ReadMidsiteC(const int t_ion, const int matsize,
                  const int nspns, const int nkpts, const int dimC, const int dimV,
                  vector<int> &allispns, vector<int> &allikpts, const int *ibndstart,
                  const int totnspns, const int totnkpts, const int totnbnds,
                  complex<double> **c_midsite);
void Mat_CtoA(complex<double> **c_onsite, complex<double> **c_midsite,
              const int mirror_onsitec, const int mirror_midsitec,
              const int t_ref, const int t_ele, const double h, const int Nh,
              const int matsize,
              complex<double> *a0, complex<double> *a1,
              complex<double> *a2, complex<double> *a3);
void A0Coeff(const int nstates, const double h, const complex<double> *a0,
             complex<double> *coeff);
void EomegaCoeff(const int nstates, const double h,
                 complex<double> *om_oih, complex<double> *coeff, const int ntrajs = 1);
void CoeffUpdate(const double h, const int t_ion, const int nstates, 
                 const int nspns, const int nkpts, const int dimC, const int dimV,
                 vector<int> &allispns, vector<int> &allikpts, const int *ibndstart,
                 const int totnspns, const int totnkpts, const int totnbnds,
                 complex<double> **c_onsite, complex<double> **c_midsite,
                 complex<double> *coeff, const int ntrajs,
                 complex<double> *a0, complex<double> *a1, complex<double> *a2, complex<double> *a3,
                 complex<double> *om_oih);
void SetIniCoeff(complex<double> *coeff, double *population, 
                 const int nstates, const int ismp, int &begtime,
                 vector<int> *&allbands, const int ntrajs = 1, const char *inicon = "inicon");

#endif
