#ifndef math_h
#define math_h

#include <mpi.h>
#include <omp.h>
#include <cstring> // strcmp
#include <iostream>
#include <cmath>
#include <complex>
#include <vector>
#include <algorithm> // max
#include <numeric> // accumulate, partial_sum
#include <cassert>

#ifndef MKL_Complex16
#define MKL_Complex16 std::complex<double>
#endif
#include <mkl.h>
#include <mkl_blacs.h>
#include <mkl_pblas.h>
#include <mkl_scalapack.h>
#include <mkl_cdft.h>
#include <fftw3.h>
#include <fftw3-mpi.h>

#include "const.h"
#include "fn.h"

using namespace std;

void SplineInterpolationPre(DFTaskPtr &task, double *&scoeff,
                            const MKL_INT nn, const MKL_INT dimy, const double *xx, const double *yy, 
                            const MKL_INT spl_intp_yhint = DF_MATRIX_STORAGE_ROWS,
                            const MKL_INT spl_intp_xhint = DF_UNIFORM_PARTITION,
                            const MKL_INT spl_intp_order = DF_PP_CUBIC,
                            const MKL_INT spl_intp_stype = DF_PP_NATURAL,
                            const MKL_INT spl_intp_bctype = DF_BC_FREE_END);

void SplineInterpolation(DFTaskPtr &task, MKL_INT nsite, double *sites, double *res);


double Dasum(const long n, const double *x, const long incx = 1);
double DZnrm2(const long n, const complex<double> *x, const long incx = 1);
void Dcopy(const long n, const double *x, const long incx, 
                               double *y, const long incy);
void Ccopy(const long n, const complex<float> *x, const long incx, 
                               complex<float> *y, const long incy);
void Zcopy(const long n, const complex<double> *x, const long incx, 
                               complex<double> *y, const long incy);
void Dscal(const long n, const double a, double *x, const long incx = 1);
void Cscal(const long n, const complex<float> a, complex<float> *x, const long incx = 1);
void Zscal(const long n, const complex<double> a, complex<double> *x, const long incx = 1);
void CSscal(const long n, const float a, complex<float> *x, const long incx = 1);
void ZDscal(const long n, const double a, complex<double> *x, const long incx = 1);
void Cconj(const long n, complex<float> *x, const long incx = 1);
void Zconj(const long n, complex<double> *x, const long incx = 1);
void Zvmul(const int n, const complex<double> *z1, const complex<double> *z2, complex<double> *res);
void Daxpy(const long n,
           double a, const double *x, double *y, 
           const long incx = 1, const long incy = 1);
void Zaxpy(const long n,
           const complex<double> a, const complex<double> *x, complex<double> *y, 
           const long incx = 1, const long incy = 1);
void Zaxpby_o(const long n,
              const complex<double> a, const complex<double> *x,
              const complex<double> b, const complex<double> *y,
                                             complex<double> *r,
              const long incx = 1, const long incy = 1, const long incr = 1);
double Ddot(int n, const double *x1, const double *x2, const int inc1 = 1, const int inc2 = 1);
complex<double> Zdot(const int n, const complex<double> *z1, const complex<double> *z2, const int inc1 = 1, const int inc2 = 1);
void Dimatcopy(const char *layout, const char *trans, const int m, const int n,
               const double alpha, double *AB, const int lda, const int ldb);
void Zimatcopy(const char *layout, const char *trans, const int m, const int n,
               const complex<double> alpha, complex<double> *AB, const int lda, const int ldb);
void Zomatcopy(const char *layout, const char *trans, const int m, const int n,
               const complex<double> alpha, const complex<double> *a, const int lda, 
                                                  complex<double> *b, const int ldb);
void Dgemv(const char *layout, const char *trans, const int m, const int n,
           const double alpha, const double *a, const int lda,
                               const double *x, const int incx,
           const double beta,        double *y, const int incy);
void Dgemm(const char *layout, const char *transa, const char *transb, const int m, const int n, const int k,
           const double alpha, const double *a, const int lda, 
                               const double *b, const int ldb,
           const double beta,        double *c, const int ldc);
void Zgemm(const char *layout, const char *transa, const char *transb, const int m, const int n, const int k,
           const complex<double> alpha, const complex<double> *a, const int lda, 
                                        const complex<double> *b, const int ldb,
           const complex<double> beta,        complex<double> *c, const int ldc);
void Zgemm_onsite_L(const char *layout, const char *transa, const char *transb, const int m, const int n, const int k,
                    const complex<double> alpha,       complex<double> *ac, const int lda, const int ldc,
                    const complex<double> beta,  const complex<double> *b,  const int ldb);
void Zgemm_onsite_R(const char *layout, const char *transa, const char *transb, const int m, const int n, const int k,
                    const complex<double> alpha, const complex<double> *a,  const int lda, 
                    const complex<double> beta,        complex<double> *bc, const int ldb, const int ldc);
void Dgetri(const int n, double *a, const int lda);
void Zheev(int layout, const char jobz, const char uplo, const int n, complex<double> *a, const int lda, double *w);
void Dsyev(int layout, const char jobz, const char uplo, const int n, double *a, const int lda, double *w);
double ZaPlusZb(complex<double> *a, complex<double> *b, const int N,
                complex<double> z1, complex<double> z2);
void ReCombineAB(complex<double> *a, complex<double> *b, const int N);

complex<double> Zdotc(const int n, const complex<double> *z1, const complex<double> *z2, const int inc1 = 1, const int inc2 = 1);
void Igather_inplace(const int n, int *x, const int *idx);
void Dgather(const int nd, const double *y, double *x, const int *indx);
void Zgather(const int nz, const complex<double> *y, complex<double> *x, const int *indx);
void Zgather_inplace(const int nz, complex<double> *z, const int *idxz);


int Numroc(const int n, const int nb, const int iproc, const int nprocs, const int psrc = 0);
void Blacs_barrier(const int ConTxt, const char *scope);
void Dgebs2d(const int ConTxt, const char *scope, const char *top, const int m, const int n, const double *A, const int lda);
void Dgebr2d(const int ConTxt, const char *scope, const char *top, const int m, const int n, double *A, const int lda, const int rsrc, const int csrc);
void Zgebs2d(const int ConTxt, const char *scope, const char *top, const int m, const int n, const complex<double> *A, const int lda);
void Zgebr2d(const int ConTxt, const char *scope, const char *top, const int m, const int n, complex<double> *A, const int lda, const int rsrc, const int csrc);
void Blacs_ColZBroadcast(complex<double> **data, const int ndata, const int *m, const int *n, const int *lda,
                         const int ctxt, const int myprow, const int mypcol, const int npcol);
void Blacs_ColZBroadcast(complex<double> **data, const int ndata, const int *m, const int n, const int *lda,
                         const int ctxt, const int myprow, const int mypcol, const int npcol);
void Blacs_ColZBroadcast(vector< complex<double> > *data, const int ndata, const int *m, const int *n, const int *lda,
                         const int ctxt, const int myprow, const int mypcol, const int npcol);
void Blacs_ColZBroadcast(vector< complex<double> > *data, const int ndata, const int *m, const int n, const int *lda,
                         const int ctxt, const int myprow, const int mypcol, const int npcol);
void Blacs_MatrixDGather(const int m, const int n,
                         const double *a, const int lda,
                               double *b, const int ldb,
                         const int m_a = -1, const int n_a = -1, const int ia = 0, const int ja = 0,
                         const int a_rowsrc = 0, const int a_colsrc = 0,
                         const int prow_dest = group_root_prow,
                         const int pcol_dest = group_root_pcol,
                         const int dest_ctxt = ctxt_only_group_root,
                         const int iprow = myprow_group, const int ipcol = mypcol_group,
                         const int nprow = nprow_group, const int npcol = npcol_group, const int ConTxt = ctxt_group);
void Blacs_MatrixZGather(const int m, const int n,
                         const complex<double> *a, const int lda,
                               complex<double> *b, const int ldb,
                         const int m_a = -1, const int n_a = -1, const int ia = 0, const int ja = 0,
                         const int a_rowsrc = 0, const int a_colsrc = 0,
                         const int prow_dest = group_root_prow,
                         const int pcol_dest = group_root_pcol,
                         const int dest_ctxt = ctxt_only_group_root,
                         const int iprow = myprow_group, const int ipcol = mypcol_group,
                         const int nprow = nprow_group, const int npcol = npcol_group, const int ConTxt = ctxt_group);
void Blacs_MatrixDScatter(const int m, const int n,
                          const double *a, const int lda,
                                double *b, const int ldb,
                          const int mb = -1, const int nb = -1, const int ib = 0, const int jb = 0,
                          const int b_rowsrc = 0, const int b_colsrc = 0,
                          const int prow_root = group_root_prow,
                          const int pcol_root = group_root_pcol,
                          const int root_ctxt = ctxt_only_group_root,
                          const int iprow = myprow_group, const int ipcol = mypcol_group,
                          const int nprow = nprow_group, const int npcol = npcol_group, const int ConTxt = ctxt_group);
void Blacs_MatrixZScatter(const int m, const int n,
                          const complex<double> *a, const int lda,
                                complex<double> *b, const int ldb,
                          const int mb = -1, const int nb = -1, const int ib = 0, const int jb = 0,
                          const int b_rowsrc = 0, const int b_colsrc = 0,
                          const int prow_root = group_root_prow,
                          const int pcol_root = group_root_pcol,
                          const int root_ctxt = ctxt_only_group_root,
                          const int iprow = myprow_group, const int ipcol = mypcol_group,
                          const int nprow = nprow_group, const int npcol = npcol_group, const int ConTxt = ctxt_group);
void Blacs_MatrixISum(const int ConTxt, const char *scope, const char *top, const int m, const int n,
                      int *A, const int lda, const int rdest, const int cdest);
void Blacs_ReadDiag2Full(const char *filename, const int itime, 
                         const int matsize, const complex<double> alpha,
                         complex<double> *locmat, const int start_idx = 0);
void Blacs_ReadFullMat(const char *filename, const int itime,
                       const int matsize, const complex<double> alpha,
                       vector<int> &locbeg, vector<int> &glbbeg, vector<int> &bcklen,
                       complex<double> *locmat, const int start_idx = 0, const bool isConj = false);
void Blacs_ReadFullMat(const char *filename, const int itime, 
                       const int matsize, const complex<double> alpha,
                       complex<double> *locmat, const int start_idx = 0, const bool isConj = false);
double Pdasum(const int n, const double *x, const int ldx, const int incx,
              const int ix, const int jx, const int m_x, const int n_x,
              const int x_prowsrc = 0, const int x_pcolsrc = 0,
              const int mb_row = MB_ROW, const int nb_col = NB_COL, const int ConTxt = ctxt_group);
double Pdznrm2(const int n, const complex<double> *x, const int ldx, const int incx,
               const int ix, const int jx, const int m_x, const int n_x,
               const int x_prowsrc = 0, const int x_pcolsrc = 0, 
               const int mb_row = MB_ROW, const int nb_col = NB_COL, const int ConTxt = ctxt_group);
complex<double> Pzdotc_CC(const int n, complex<double> *x, complex<double> *y, 
                          const int x_colsrc = 0, const int y_colsrc = 0,
                          const int x_rowsrc = 0, const int y_rowsrc = 0,
                          const int iprow = myprow_group, const int nprow = nprow_group, const int ConTxt = ctxt_group);
complex<double> Pzdot_CC(const int n, complex<double> *x, complex<double> *y, 
                         const int x_colsrc = 0, const int y_colsrc = 0,
                         const int x_rowsrc = 0, const int y_rowsrc = 0,
                         const int iprow = myprow_group, const int nprow = nprow_group, const int ConTxt = ctxt_group);
complex<double> Pzdotc(const int n, const complex<double> *x, const int ldx, const int incx,
                                    const complex<double> *y, const int ldy, const int incy,
                       const int ix, const int jx, const int iy, const int jy,
                       const int m_x, const int n_x, const int m_y, const int n_y,
                       const int x_prowsrc = 0, const int x_pcolsrc = 0,
                       const int y_prowsrc = 0, const int y_pcolsrc = 0, 
                       const int mb_row = MB_ROW, const int nb_col = NB_COL,
                       const int myprow = myprow_group, const int mypcol = mypcol_group,
                       const int nprow = nprow_group, const int npcol = npcol_group,
                       const int ConTxt = ctxt_group);
void Pzscal(const int n, const complex<double> alpha, complex<double> *x, const int ldx, const int incx,
            const int ix, const int jx, const int m_x, const int n_x,
            const int x_prowsrc = 0, const int x_pcolsrc = 0,
            const int mb_row = MB_ROW, const int nb_col = NB_COL, const int ConTxt = ctxt_group);
void Pdgemv(const char *transa, const int m, const int n, 
            const double alpha, const double *a, const int lda,
                                const double *x, const int ldx,
            const double beta,        double *y, const int ldy,
            const int incx = 1, const int incy = 1,
            const int m_a = -1, const int n_a = -1,
            const int m_x = -1, const int n_x = -1, 
            const int m_y = -1, const int n_y = -1,
            const int ia = 0, const int ja = 0, const int ix = 0, const int jx = 0, const int iy = 0, const int jy = 0,
            const int a_prowsrc = 0, const int a_pcolsrc = 0,
            const int x_prowsrc = 0, const int x_pcolsrc = 0,
            const int y_prowsrc = 0, const int y_pcolsrc = 0, 
            const int mb_row = MB_ROW, const int nb_col = NB_COL, const int ConTxt = ctxt_group);
void Pzgemv(const char *transa, const int m, const int n, 
            const complex<double> alpha, const complex<double> *a, const int lda,
                                         const complex<double> *x, const int ldx,
            const complex<double> beta,        complex<double> *y, const int ldy,
            const int incx = 1, const int incy = 1,
            const int m_a = -1, const int n_a = -1,
            const int m_x = -1, const int n_x = -1, 
            const int m_y = -1, const int n_y = -1,
            const int ia = 0, const int ja = 0, const int ix = 0, const int jx = 0, const int iy = 0, const int jy = 0,
            const int a_prowsrc = 0, const int a_pcolsrc = 0,
            const int x_prowsrc = 0, const int x_pcolsrc = 0,
            const int y_prowsrc = 0, const int y_pcolsrc = 0, 
            const int mb_row = MB_ROW, const int nb_col = NB_COL, const int ConTxt = ctxt_group);
void Pdgemm(const char *transa, const char *transb, const int m, const int n, const int k, 
            const double alpha, const double *a, const int lda,
                                const double *b, const int ldb, 
            const double beta,        double *c, const int ldc,
            const int m_a = -1, const int n_a = -1,
            const int m_b = -1, const int n_b = -1, 
            const int m_c = -1, const int n_c = -1,
            const int ia = 0, const int ja = 0, const int ib = 0, const int jb = 0, const int ic = 0, const int jc = 0,
            const int a_prowsrc = 0, const int a_pcolsrc = 0,
            const int b_prowsrc = 0, const int b_pcolsrc = 0,
            const int c_prowsrc = 0, const int c_pcolsrc = 0, 
            const int mb_row = MB_ROW, const int nb_col = NB_COL, const int ConTxt = ctxt_group);
void Pzgemm(const char *transa, const char *transb, const int m, const int n, const int k, 
            const complex<double> alpha, const complex<double> *a, const int lda,
                                         const complex<double> *b, const int ldb, 
            const complex<double> beta,        complex<double> *c, const int ldc,
            const int m_a = -1, const int n_a = -1,
            const int m_b = -1, const int n_b = -1, 
            const int m_c = -1, const int n_c = -1,
            const int ia = 0, const int ja = 0, const int ib = 0, const int jb = 0, const int ic = 0, const int jc = 0,
            const int a_prowsrc = 0, const int a_pcolsrc = 0,
            const int b_prowsrc = 0, const int b_pcolsrc = 0,
            const int c_prowsrc = 0, const int c_pcolsrc = 0, 
            const int mb_row = MB_ROW, const int nb_col = NB_COL, const int ConTxt = ctxt_group);
void Pzgecom(const int n, const int ldd_loc, const complex<double> alpha,
             const complex<double> *a, const complex<double> *b,
                   complex<double> *r);
void Pzlacpy(const char *uplo, const int m, const int n, // VERY deprecated
             const complex<double> *a, const int lda, complex<double> *b, const int ldb,
             const int m_a = -1, const int n_a = -1, const int m_b = -1, const int n_b = -1,
             const int ia = 0, const int ja = 0, const int ib = 0, const int jb = 0,
             const int a_prowsrc = 0, const int a_pcolsrc = 0,
             const int b_prowsrc = 0, const int b_pcolsrc = 0,
             const int mb_row = MB_ROW, const int nb_col = NB_COL, const int ConTxt = ctxt_group);
void Pzcopy(const int m, const int n,
            const complex<double> *a, const int lda, 
                  complex<double> *b, const int ldb,
            const int m_a = -1, const int n_a = -1,
            const int m_b = -1, const int n_b = -1,
            const int ia = 0, const int ja = 0, const int ib = 0, const int jb = 0,
            const int a_prowsrc = 0, const int a_pcolsrc = 0,
            const int b_prowsrc = 0, const int b_pcolsrc = 0,
            const int mb_row = MB_ROW, const int nb_col = NB_COL, const int ConTxt = ctxt_group);
void Pzgeadd(const char *transa, const int m, const int n,
             const complex<double> alpha, const complex<double> *a, const int lda, 
             const complex<double> beta,        complex<double> *b, const int ldb,
             const int m_a = -1, const int n_a = -1,
             const int m_b = -1, const int n_b = -1,
             const int ia = 0, const int ja = 0, const int ib = 0, const int jb = 0,
             const int a_prowsrc = 0, const int a_pcolsrc = 0,
             const int b_prowsrc = 0, const int b_pcolsrc = 0,
             const int mb_row = MB_ROW, const int nb_col = NB_COL, const int ConTxt = ctxt_group);
void Pzheev(const char *jobz, const int n, const complex<double> *a, const int lda,
            double *w, complex<double> *z, const int ldz, // output: w-eigenvalues, z-eigenvectors
            const int m_a = -1, const int n_a = -1, const int m_z = -1, const int n_z = -1,
            const int ia = 0, const int ja = 0, const int iz = 0, const int jz = 0, const char *uplo = "U",
            const int a_prowsrc = 0, const int a_pcolsrc = 0,
            const int z_prowsrc = 0, const int z_pcolsrc = 0,
            const int mb_row = MB_ROW, const int nb_col = NB_COL, const int ConTxt = ctxt_group);


double RealSphHarFun(const int l, const int m, const double theta, const double phi);


void Pzfft3SetMallocSize(const int n0, const int n1, const int n2,
                         int &local_n0, int &local_0_start, int &total_local_size, MPI_Comm &comm);
void Pzfft3(complex<double> *data, const int n[], const int forback, MPI_Comm &comm);
void BlockCyclicToMpiFFTW3d(const complex<double> *in,
                            const int totin, const int totin_loc,
                            complex<double> *out,
                            const int nin[], const int nout[],
                            const int *loc_nout0, const int *loc_out0_start,
                            const int *idxin);
void MpiFFTW3dToBlockCyclic(const complex<double> *in, 
                            complex<double> *out, 
                            const int totout, const int totout_loc,
                            const int nin[], const int nout[],
                            const int *loc_nin0, const int *loc_in0_start,
                            const int *idxout);
void Zfft_3d(complex<double> *data, const int n[], const int forback);
void Zfft_1d(const int N, complex<double> *in, complex<double> *out, const int forback);
void Zr2cfft_1d(const int N, double *in, complex<double> *out);
void Zc2rfft_1d(const int N, complex<double> *in, double *out);
void DselfCrossCorrelation(const int N, double *inout);
void CumulativeIntegralTwice(const double alpha, const int N, double *inout);

void LinearLSF(const int N, const double *x, const double *y,
               double &a, double &b);
double DirectProportionLSF(const int N, const double *x, const double *y);
double GassuianFitting1(const int N, const double x0, const double dx,
                        const double *mlny);

#endif
