#include "math.h"

//**************************************************************************************
//
//                                 Spline Interpolation
//
//**************************************************************************************
void SplineInterpolationPre(DFTaskPtr &task, double *&scoeff, // IMPORTANT: DO NOT delete within this function
                            const MKL_INT nn, const MKL_INT dimy, const double *xx, const double *yy, 
                            const MKL_INT spl_intp_yhint,
                            const MKL_INT spl_intp_xhint,
                            const MKL_INT spl_intp_order,
                            const MKL_INT spl_intp_stype,
                            const MKL_INT spl_intp_bctype) {
    assert(nn > 0); assert(dimy > 0);
    /* 
       Input:
       task:   Data Fitting operations are task based
       scoeff: coefficients for interpolation polynomials
       nn:     The size of partition
       dimy:   Dimension of vector-valued function yy
       xx:     each breakpoints, total nn
       yy:     function values at the breakpoints, total nn
    */
    int status;                             /* Status of a Data Fitting operation */
    MKL_INT xhint = spl_intp_xhint;         /* The partition is non-uniform. */
    MKL_INT ny = dimy;                      /* The function is ny-dimension. */
    /* Default: Data is stored in row-major format according to C conventions.
       y0[x_0]   y0[x_1]   ... y0[x_n-1]
       y1[x_0]   y1[x_1]   ... y1[x_n-1]
       .         .             .
       .         .             .
       .         .             .
       yn-1[x_0] yn-1[x_1] ... yn-1[x_n-1] */
    MKL_INT yhint = spl_intp_yhint; // default: DF_MATRIX_STORAGE_ROWS
    /* Initialize spline parameters */
    status = dfdNewTask1D(&task, nn, xx, xhint, ny, yy, yhint);

    /* Parameters describing the spline */
    MKL_INT s_order = spl_intp_order;  /* Spline is of the fourth order (cubic spline)-Default. */
    MKL_INT s_type  = spl_intp_stype;  /* Spline is of the natural cubic type. */
    MKL_INT bc_type = spl_intp_bctype; /* Type of boundary conditions: free-end(natural) */
    double *bc      = NULL;            /* Array of boundary conditions */
    MKL_INT ic_type = DF_NO_IC;        /* Type of internal conditions */
    double *ic      = NULL;            /* Array of internal conditions */
    scoeff  = new double[ny * (nn - 1) * s_order]();   /* Array of spline coefficients */
    MKL_INT scoeffhint = DF_NO_HINT;  /* Additional information about the coefficients: No additional information about the spline */
    /* Set spline parameters in the Data Fitting task */
    status = dfdEditPPSpline1D(task, s_order, s_type, bc_type, bc, ic_type, ic, scoeff, scoeffhint);

    /* Use a standard method to construct a cubic natural(Default) spline: */
    /* Pi(x) = ci,0 + ci,1(x - xi)^1 + ci,2(x - xi)^2 + ci,3(x - xi)^3 + ..., default to order 3 */
    status = dfdConstruct1D(task, DF_PP_SPLINE, DF_METHOD_STD);

    return;
}

void SplineInterpolation(DFTaskPtr &task, MKL_INT nsite, double *sites, double *res) {
    // total dimy results, storage order is function by funciton, each function have nsite interpolation values
    // res[y0][nsite], res[y1][nsite], ..., res[yn-1][nsite]
    assert(nsite > 0);
    int status;
    MKL_INT sitehint = DF_NON_UNIFORM_PARTITION;        // Additional information about the structure of interpolation sites
    MKL_INT ndorder = 1, dorder = 1;                    // Parameters defining the type of interpolation
    double *datahint = NULL;                            // Additional information on partition and interpolation sites
    datahint = DF_NO_APRIORI_INFO;
    MKL_INT rhint = DF_MATRIX_STORAGE_FUNCS_SITES_DERS; // Additional information on the structure of the results
    MKL_INT *cell = NULL;                               // Array of cell indices: Not required
    
    status = dfdInterpolate1D(task, DF_INTERP, DF_METHOD_PP, nsite, sites,
                              sitehint, ndorder, &dorder, datahint, res, rhint, cell);
    return;
}

//**************************************************************************************
//
//                                 Linear Algebra
//
//**************************************************************************************
double Dasum(const long n, const double *x, const long incx) {
    if(n > 0) return cblas_dasum(n, x, incx);
}

double DZnrm2(const long n, const complex<double> *x, const long incx) {
    if(n > 0) return cblas_dznrm2(n, x, incx);
}

void Dcopy(const long n, const double *x, const long incx, 
                               double *y, const long incy) {
    if(n > 0) cblas_dcopy(n, x, incx, y, incy);
    return;
}

void Ccopy(const long n, const complex<float> *x, const long incx, 
                               complex<float> *y, const long incy) {
    if(n > 0) cblas_ccopy(n, (const void*)x, incx, (void*)y, incy);
    return;
}

void Zcopy(const long n, const complex<double> *x, const long incx, 
                               complex<double> *y, const long incy) {
    if(n > 0) cblas_zcopy(n, (const void*)x, incx, (void*)y, incy);
    return;
}

void Dscal(const long n, const double a, double *x, const long incx) {
    if(n > 0) cblas_dscal(n, a, x, incx);
    return;
}

void Cscal(const long n, const complex<float> a, complex<float> *x, const long incx) {
    if(n > 0) cblas_cscal(n, (const void*)&a, (void*)x, incx);
    return;
}

void Zscal(const long n, const complex<double> a, complex<double> *x, const long incx) {
    if(n > 0) cblas_zscal(n, (const void*)&a, (void*)x, incx);
    return;
}

void CSscal(const long n, const float a, complex<float> *x, const long incx) {
    if(n > 0) cblas_csscal(n, a, (void*)x, incx);
    return;
}

void ZDscal(const long n, const double a, complex<double> *x, const long incx) {
    if(n > 0) cblas_zdscal(n, a, (void*)x, incx);
    return;
}

void Cconj(const long n, complex<float> *x, const long incx) {
    complex<float> *y = new complex<float>[n];
    Ccopy(n, x, incx, y, 1);
    vcConjI(n, (const MKL_Complex8*)y, 1, (MKL_Complex8*)x, incx);
    delete[] y;
    return;
}

void Zconj(const long n, complex<double> *x, const long incx) {
    complex<double> *y = new complex<double>[n];
    Zcopy(n, x, incx, y, 1);
    vzConjI(n, (const MKL_Complex16*)y, 1, (MKL_Complex16*)x, incx);
    delete[] y;
    return;
}

void Zvmul(const int n, const complex<double> *z1, const complex<double> *z2, complex<double> *res) {
    vzMul(n, (const MKL_Complex16*)z1, (const MKL_Complex16*)z2, (MKL_Complex16*)res);
    return;
}

void Daxpy(const long n,
           double a, const double *x, double *y, 
           const long incx, const long incy) {
    if(n > 0) cblas_zaxpy(n, &a, x, incx, y, incy);
    return;
}

void Zaxpy(const long n,
           const complex<double> a, const complex<double> *x, complex<double> *y, 
           const long incx, const long incy) {
    if(n > 0) cblas_zaxpy(n, (void*)&a, (const void*)x, incx, (void*)y, incy);
    return;
}

void Zaxpby_o(const long n, // o: out-place
              const complex<double> a, const complex<double> *x,
              const complex<double> b, const complex<double> *y,
                                             complex<double> *r,
              const long incx, const long incy, const long incr) {

    /* r = ax + by */
    if(n <= 0) return;
    Zcopy(n, y, incy, r, incr);
    Zscal(n, b, r, incr);
    Zaxpy(n, a, x, r, incx, incr);
    
    return;
}

double Ddot(int n, const double *x1, const double *x2, const int inc1, const int inc2) {
    return cblas_ddot(n, x1, inc1, x2, inc2);
}
complex<double> Zdot(const int n, const complex<double> *z1, const complex<double> *z2, const int inc1, const int inc2) {
    complex<double> res;
    cblas_zdotu_sub(n, z1, inc1, z2, inc2, &res);
    return res;
}

complex<double> Zdotc(const int n, const complex<double> *z1, const complex<double> *z2, const int inc1, const int inc2) {
    complex<double> res;
    cblas_zdotc_sub(n, z1, inc1, z2, inc2, &res);
    return res;
}

void Dimatcopy(const char *layout, const char *trans, const int m, const int n,
               const double alpha, double *AB, const int lda, const int ldb) {
    if(m <= 0 || n <= 0) return;
    assert(lda > 0); assert(ldb > 0);
    char ordering;
    char char_trans;
         if(!strcmp(layout, "CblasRowMajor")) ordering = 'R';
    else if(!strcmp(layout, "CblasColMajor")) ordering = 'C';
         
         if(!strcmp(trans, "CblasNoTrans"))   char_trans = 'N';
    else if(!strcmp(trans, "CblasTrans"))     char_trans = 'T';
    else if(!strcmp(trans, "CblasConjTrans")) char_trans = 'C';
    else if(!strcmp(trans, "CblasConj"))      char_trans = 'R';

    mkl_dimatcopy(ordering, char_trans, m, n, alpha, AB, lda, ldb);
    
    return;
}

void Zimatcopy(const char *layout, const char *trans, const int m, const int n,
               const complex<double> alpha, complex<double> *AB, const int lda, const int ldb) {
    if(m <= 0 || n <= 0) return;
    assert(lda > 0); assert(ldb > 0);
    char ordering;
    char char_trans;
         if(!strcmp(layout, "CblasRowMajor")) ordering = 'R';
    else if(!strcmp(layout, "CblasColMajor")) ordering = 'C';
         
         if(!strcmp(trans, "CblasNoTrans"))   char_trans = 'N';
    else if(!strcmp(trans, "CblasTrans"))     char_trans = 'T';
    else if(!strcmp(trans, "CblasConjTrans")) char_trans = 'C';
    else if(!strcmp(trans, "CblasConj"))      char_trans = 'R';

    mkl_zimatcopy(ordering, char_trans, m, n, alpha, (MKL_Complex16*)AB, lda, ldb);
    
    return;
}

void Zomatcopy(const char *layout, const char *trans, const int m, const int n,
               const complex<double> alpha, const complex<double> *a, const int lda, 
                                                  complex<double> *b, const int ldb) {
    if(m <= 0 || n <= 0) return;
    assert(lda > 0); assert(ldb > 0);
    char ordering;
    char char_trans;
         if(!strcmp(layout, "CblasRowMajor")) ordering = 'R';
    else if(!strcmp(layout, "CblasColMajor")) ordering = 'C';
         
         if(!strcmp(trans, "CblasNoTrans"))   char_trans = 'N';
    else if(!strcmp(trans, "CblasTrans"))     char_trans = 'T';
    else if(!strcmp(trans, "CblasConjTrans")) char_trans = 'C';
    else if(!strcmp(trans, "CblasConj"))      char_trans = 'R';

    mkl_zomatcopy(ordering, char_trans, m, n, alpha, (MKL_Complex16*)a, lda, (MKL_Complex16*)b, ldb);

    return;
}

void Dgemv(const char *layout, const char *trans, const int m, const int n,
           const double alpha, const double *a, const int lda,
                               const double *x, const int incx,
           const double beta,        double *y, const int incy) {
    if(m <=0 || n <= 0) return;
    CBLAS_LAYOUT cblas_layout;
    CBLAS_TRANSPOSE cblas_trans;
         if(!strcmp(layout, "CblasRowMajor")) cblas_layout = CblasRowMajor;
    else if(!strcmp(layout, "CblasColMajor")) cblas_layout = CblasColMajor;
    
         if(!strcmp(trans, "CblasNoTrans"))   cblas_trans = CblasNoTrans;
    else if(!strcmp(trans, "CblasTrans"))     cblas_trans = CblasTrans;
    else if(!strcmp(trans, "CblasConjTrans")) cblas_trans = CblasConjTrans;

    cblas_dgemv(cblas_layout, cblas_trans, m, n,
                alpha, a, lda, x, incx,
                beta,          y, incy);
    return;
}

void Dgemm(const char *layout, const char *transa, const char *transb, const int m, const int n, const int k,
           const double alpha, const double *a, const int lda, 
                               const double *b, const int ldb,
           const double beta,        double *c, const int ldc) {
    if(m <= 0 || n <= 0 || k <= 0) return;
    CBLAS_LAYOUT cblas_layout;
    CBLAS_TRANSPOSE cblas_transa, cblas_transb;
         if(!strcmp(layout, "CblasRowMajor")) cblas_layout = CblasRowMajor;
    else if(!strcmp(layout, "CblasColMajor")) cblas_layout = CblasColMajor;
    
         if(!strcmp(transa, "CblasNoTrans"))   cblas_transa = CblasNoTrans;
    else if(!strcmp(transa, "CblasTrans"))     cblas_transa = CblasTrans;
    else if(!strcmp(transa, "CblasConjTrans")) cblas_transa = CblasConjTrans;

         if(!strcmp(transb, "CblasNoTrans"))   cblas_transb = CblasNoTrans;
    else if(!strcmp(transb, "CblasTrans"))     cblas_transb = CblasTrans;
    else if(!strcmp(transb, "CblasConjTrans")) cblas_transb = CblasConjTrans;

    cblas_dgemm(cblas_layout, cblas_transa, cblas_transb, m, n, k,
                alpha, a, lda, b, ldb,
                beta,  c, ldc);
    return;
}

void Zgemm(const char *layout, const char *transa, const char *transb, const int m, const int n, const int k,
           const complex<double> alpha, const complex<double> *a, const int lda, 
                                        const complex<double> *b, const int ldb,
           const complex<double> beta,        complex<double> *c, const int ldc) {
    if(m <= 0 || n <= 0 || k <= 0) return;
    CBLAS_LAYOUT cblas_layout;
    CBLAS_TRANSPOSE cblas_transa, cblas_transb;
         if(!strcmp(layout, "CblasRowMajor")) cblas_layout = CblasRowMajor;
    else if(!strcmp(layout, "CblasColMajor")) cblas_layout = CblasColMajor;
    
         if(!strcmp(transa, "CblasNoTrans"))   cblas_transa = CblasNoTrans;
    else if(!strcmp(transa, "CblasTrans"))     cblas_transa = CblasTrans;
    else if(!strcmp(transa, "CblasConjTrans")) cblas_transa = CblasConjTrans;

         if(!strcmp(transb, "CblasNoTrans"))   cblas_transb = CblasNoTrans;
    else if(!strcmp(transb, "CblasTrans"))     cblas_transb = CblasTrans;
    else if(!strcmp(transb, "CblasConjTrans")) cblas_transb = CblasConjTrans;

    cblas_zgemm(cblas_layout, cblas_transa, cblas_transb, m, n, k,
                (const void*)(&alpha), (const void*)a, lda, (const void*)b, ldb,
                (const void*)(&beta),        (void*)c, ldc);
    return;
}

void Zgemm_onsite_L(const char *layout, const char *transa, const char *transb, const int m, const int n, const int k,
                    const complex<double> alpha,       complex<double> *ac, const int lda, const int ldc,
                    const complex<double> beta,  const complex<double> *b,  const int ldb) {
    if(m <= 0 || n <= 0 || k <= 0) return;
    const int tmpldc = max(m, n);
    complex<double> *tmpc = new complex<double>[tmpldc * tmpldc];
    Zgemm(layout, transa, transb, m, n, k, alpha, ac, lda, b, ldb, beta, tmpc, tmpldc);
    Zomatcopy(layout, "CblasNoTrans", m, n, 1.0, tmpc, tmpldc, ac, ldc);
    delete[] tmpc;
    return;
}

void Zgemm_onsite_R(const char *layout, const char *transa, const char *transb, const int m, const int n, const int k,
                    const complex<double> alpha, const complex<double> *a,  const int lda, 
                    const complex<double> beta,        complex<double> *bc, const int ldb, const int ldc) {
    if(m <= 0 || n <= 0 || k <= 0) return;
    const int tmpldc = max(m, n);
    complex<double> *tmpc = new complex<double>[tmpldc * tmpldc];
    Zgemm(layout, transa, transb, m, n, k, alpha, a, lda, bc, ldb, beta, tmpc, tmpldc);
    Zomatcopy(layout, "CblasNoTrans", m, n, 1.0, tmpc, tmpldc, bc, ldc);
    delete[] tmpc;
    return;
}

void Igather_inplace(const int n, int *x, const int *idx) {
    if(n <= 0) return;
    int *y = new int[n];
    #pragma omp parallel for
    for(int i = 0; i < n; i++) {
        y[i] = x[ idx[i] ];
    }
    copy(y, y + n, x);
    
    delete[] y;

    return;
}

void Dgather(const int nd, const double *y, double *x, const int *indx) {
    // gather nd elements form y to compacted x, x[i] = y[indx[i]]
    if(nd > 0) cblas_dgthr(nd, y, x, indx);
    return;
}

void Zgather(const int nz, const complex<double> *y, complex<double> *x, const int *indx) {
    // gather nz elements from y to compacted x, x[i] = y[indx[i]]
    if(nz > 0) cblas_zgthr(nz, (const void*)y, (void *)x, indx);
    return;
}

void Zgather_inplace(const int nz, complex<double> *z, const int *idxz) {
    if(nz <= 0) return; 
    complex<double> *newz = new complex<double>[nz];
    Zgather(nz, z, newz, idxz);
    Zcopy(nz, newz, 1, z, 1);
    delete[] newz;
    return;
}

void Dgetri(const int n, double *a, const int lda) { // inverse of nxn matrix a
    int *ipiv = new int[n];
    int info;
    LAPACKE_dgetrf(LAPACK_ROW_MAJOR, n, n, a, lda, ipiv); // LU factorization 
    LAPACKE_dgetri(LAPACK_ROW_MAJOR, n, a, lda, ipiv);    // inverse
    delete[] ipiv;
    return;
}

void Zheev(int layout, const char jobz, const char uplo, const int n, complex<double> *a, const int lda, double *w) {
    int lapack;
    lapack = LAPACKE_zheev(layout, jobz, uplo, n, (lapack_complex_double*)a, lda, w);
    if(lapack) { cerr << "wrong in Zheev calculating eigenvalues: " << lapack << endl; exit(1); }
    return;
}

void Dsyev(int layout, const char jobz, const char uplo, const int n, double *a, const int lda, double *w) {
    int lapack;
    lapack = LAPACKE_dsyev(layout, jobz, uplo, n, a, lda, w);
    if(lapack) { cerr << "wrong in Dsyev calculating eigenvalues: " << lapack << endl; exit(1); }
    return;
}

double ZaPlusZb(complex<double> *a, complex<double> *b, const int N,
                complex<double> z1, complex<double> z2) {
    complex<double> *r = new complex<double>[N]();
    Zaxpby_o(N, z1, a, z2, b, r);
    double res = DZnrm2(N, r);
    delete[] r;
    return res;
}

void ReCombineAB(complex<double> *a, complex<double> *b, const int N) { // a/b are 2N-dimension column vectors
    double *abmat = new double[4 * 4]();
    double *eigval = new double[4];
    complex<double> cdtmp, z1, z2;
    double minf, imin;
    int minInd;
    complex<double> *res1 = new complex<double>[2 * N];
    complex<double> *res2 = new complex<double>[2 * N];
    complex<double> *res = NULL;
    for(int is = 0; is < 2; is++) {
        abmat[0] = DZnrm2(N, a + is * N);
        cdtmp = Zdotc(N, a + is * N, b + is * N);
        abmat[10] = DZnrm2(N, b + is * N);

        abmat[0]*= abmat[0];     abmat[1] = 0.0;         abmat[2]  = real(cdtmp);  abmat[3] = -imag(cdtmp);
        abmat[4] = 0.0;          abmat[5] = abmat[0];    abmat[6]  = imag(cdtmp);  abmat[7] = real(cdtmp);
        abmat[8] = real(cdtmp);  abmat[9] = imag(cdtmp); abmat[10]*= abmat[10];    abmat[11]= 0.0;
        abmat[12]= -imag(cdtmp); abmat[13]= real(cdtmp); abmat[14] = 0.0;          abmat[15]= abmat[10];

        Dsyev(LAPACK_ROW_MAJOR, 'V', 'L', 4, abmat, 4, eigval);

        z1 = complex<double>(abmat[0], abmat[4]); z2 = complex<double>(abmat[8], abmat[12]);
        minf = ZaPlusZb(a + is * N, b + is * N, N, z1, z2);
        minInd = 0;
        for(int ii = 1; ii < 4; ii++) {
            z1 = complex<double>(abmat[ii], abmat[ii + 4]); z2 = complex<double>(abmat[ii + 8], abmat[ii + 12]);
            imin = ZaPlusZb(a + is * N, b + is * N, N, z1, z2);
            if(imin < minf) { minf = imin; minInd = ii; }
        }

        z1 = complex<double>(abmat[minInd], abmat[minInd + 4]);
        z2 = complex<double>(abmat[minInd + 8], abmat[minInd + 12]);

        if(is) res = res1;
        else res = res2;
        Zaxpby_o(2 * N, z1, a, z2, b, res);
    }
    Zcopy(2 * N, res1, 1, a, 1);
    Zcopy(2 * N, res2, 1, b, 1);

    delete[] abmat; delete[] eigval;
    delete[] res1; delete[] res2; 

    return;
}

//**************************************************************************************
//
//                           Linear Algebra Parrallel
//
//**************************************************************************************
int Numroc(const int n, const int nb, const int iproc, const int nprocs, const int psrc) {
    return numroc(&n, &nb, &iproc, &psrc, &nprocs);
}

void Blacs_barrier(const int ConTxt, const char *scope) {
    blacs_barrier(&ConTxt, scope);
    return;
}

void Dgebs2d(const int ConTxt, const char *scope, const char *top, const int m, const int n, const double *A, const int lda) {
    dgebs2d(&ConTxt, scope, top, &m, &n, A, &lda);
    return;
}

void Dgebr2d(const int ConTxt, const char *scope, const char *top, const int m, const int n, double *A, const int lda, const int rsrc, const int csrc) {
    dgebr2d(&ConTxt, scope, top, &m, &n, A, &lda, &rsrc, &csrc);
    return;
}

void Zgebs2d(const int ConTxt, const char *scope, const char *top, const int m, const int n, const complex<double> *A, const int lda) {
    zgebs2d(&ConTxt, scope, top, &m, &n, (const double*)A, &lda);
    return;
}

void Zgebr2d(const int ConTxt, const char *scope, const char *top, const int m, const int n, complex<double> *A, const int lda, const int rsrc, const int csrc) {
    zgebr2d(&ConTxt, scope, top, &m, &n, (double*)A, &lda, &rsrc, &csrc);
    return;
}

void Blacs_ColZBroadcast(complex<double> **data, const int ndata, const int *m, const int *n, const int *lda, // data[ndata][m][n]
                         const int ctxt, const int myprow, const int mypcol, const int npcol) {
    if(npcol == 1) {
        Blacs_barrier(ctxt, "R");
        return;
    }
    for(int rt_col = 0; rt_col < npcol; rt_col++) {
        for(int ii = rt_col; ii < ndata; ii += npcol) {
            if(mypcol == rt_col) Zgebs2d(ctxt, "ROW", "I", m[ii], n[ii], (complex<double>*)(data[ii]), lda[ii]);
            else Zgebr2d(ctxt, "ROW", "I", m[ii], n[ii], (complex<double>*)(data[ii]), lda[ii], myprow, rt_col);
        }
    }
    Blacs_barrier(ctxt, "R");
    return;
}

void Blacs_ColZBroadcast(complex<double> **data, const int ndata, const int *m, const int n, const int *lda, // data[ndata][m][n]
                         const int ctxt, const int myprow, const int mypcol, const int npcol) {
    int *nn = new int[ndata];
    for(int i = 0; i < ndata; i++) nn[i] = n;
    Blacs_ColZBroadcast(data, ndata, m, nn, lda, ctxt, myprow, mypcol, npcol);
    return;
}

void Blacs_ColZBroadcast(vector< complex<double> > *data, const int ndata, const int *m, const int *n, const int *lda, // data[ndata][m][n]
                         const int ctxt, const int myprow, const int mypcol, const int npcol) {
    if(npcol == 1) {
        Blacs_barrier(ctxt, "R");
        return;
    }
    for(int rt_col = 0; rt_col < npcol; rt_col++) {
        for(int ii = rt_col; ii < ndata; ii += npcol) {
            if(mypcol == rt_col) Zgebs2d(ctxt, "ROW", "I", m[ii], n[ii], (complex<double>*)(&data[ii][0]), lda[ii]);
            else Zgebr2d(ctxt, "ROW", "I", m[ii], n[ii], (complex<double>*)(&data[ii][0]), lda[ii], myprow, rt_col);
        }
    }
    Blacs_barrier(ctxt, "R");
    return;
}

void Blacs_ColZBroadcast(vector< complex<double> > *data, const int ndata, const int *m, const int n, const int *lda, // data[ndata][m][n]
                         const int ctxt, const int myprow, const int mypcol, const int npcol) {
    int *nn = new int[ndata];
    for(int i = 0; i < ndata; i++) nn[i] = n;
    Blacs_ColZBroadcast(data, ndata, m, nn, lda, ctxt, myprow, mypcol, npcol);
    return;
}

void Blacs_MatrixDGather(const int m, const int n,
                         const double *a, const int lda, 
                               double *b, const int ldb,
                         const int m_a, const int n_a, const int ia, const int ja,
                         const int a_rowsrc, const int a_colsrc,
                         const int prow_dest, const int pcol_dest, const int dest_ctxt,
                         const int iprow, const int ipcol,
                         const int nprow, const int npcol, const int ConTxt) {
    // matrix a[ia:ia+m][ja:ja+n] are distributed stored
    // matrix b[m][n] are local stroed at (prow_dest, pcol_dest) in current ctxt
    const int iia = ia + 1; const int jja = ja + 1; // C to Fortran
    const int mm_a = (m_a < 0 ? m : m_a);
    const int nn_a = (n_a < 0 ? n : n_a);
    int *desca = new int[9];
    int *descb = new int[9];
    int infoa, infob;
    const int lld_a = max(lda, 1);
    const int lld_b = max(ldb, 1);
    descinit(desca, &mm_a, &nn_a, &MB_ROW, &NB_COL, &a_rowsrc, &a_colsrc, &ConTxt,    &lld_a, &infoa);
    if(iprow == prow_dest && ipcol == pcol_dest)
    descinit(descb, &m,    &n,    &MB_ROW, &NB_COL, &iZERO,    &iZERO,    &dest_ctxt, &lld_b, &infob);
    else descb[1] = dest_ctxt;
    Blacs_barrier(ConTxt, "A");
    pdgemr2d(&m, &n, a, &iia,  &jja,  desca,
                     b, &iONE, &iONE, descb, &ConTxt);
    delete[] desca; delete[] descb;
    
    Blacs_barrier(ConTxt, "A");
    return;
}

void Blacs_MatrixZGather(const int m, const int n,
                         const complex<double> *a, const int lda, 
                               complex<double> *b, const int ldb,
                         const int m_a, const int n_a, const int ia, const int ja,
                         const int a_rowsrc, const int a_colsrc,
                         const int prow_dest, const int pcol_dest, const int dest_ctxt,
                         const int iprow, const int ipcol,
                         const int nprow, const int npcol, const int ConTxt) {
    // matrix a[ia:ia+m][ja:ja+n] are distributed stored
    // matrix b[m][n] are local stroed at (prow_dest, pcol_dest) in current ctxt
    const int iia = ia + 1; const int jja = ja + 1; // C to Fortran
    const int mm_a = (m_a < 0 ? m : m_a);
    const int nn_a = (n_a < 0 ? n : n_a);
    int *desca = new int[9];
    int *descb = new int[9];
    int infoa, infob;
    const int lld_a = max(lda, 1);
    const int lld_b = max(ldb, 1);
    /*MPI_Barrier(group_comm); for(int irk = 0; irk < sub_size; irk++) {
        if(irk == sub_rank) cout << "irk = " << irk << '/' << sub_size << ": " << endl;
        MPI_Barrier(group_comm);
    } MPI_Barrier(group_comm);*/
    descinit(desca, &mm_a, &nn_a, &MB_ROW, &NB_COL, &a_rowsrc, &a_colsrc, &ConTxt,    &lld_a, &infoa);
    if(iprow == prow_dest && ipcol == pcol_dest)
    descinit(descb, &m,    &n,    &MB_ROW, &NB_COL, &iZERO,    &iZERO,    &dest_ctxt, &lld_b, &infob);
    else descb[1] = dest_ctxt;
    Blacs_barrier(ConTxt, "A");
    pzgemr2d(&m, &n, (MKL_Complex16*)a, &iia,  &jja,  desca,
                     (MKL_Complex16*)b, &iONE, &iONE, descb, &ConTxt);
    delete[] desca; delete[] descb;
    
    Blacs_barrier(ConTxt, "A");
    return;
}

void Blacs_MatrixDScatter(const int m, const int n,
                          const double *a, const int lda,
                                double *b, const int ldb,
                          const int m_b, const int n_b, const int ib, const int jb,
                          const int b_rowsrc, const int b_colsrc,
                          const int prow_root, const int pcol_root, const int root_ctxt,
                          const int iprow, const int ipcol,
                          const int nprow, const int npcol, const int ConTxt) {
    // matrix a[m][n] are local stroed at (prow_root, pcol_root) in current ctxt
    // matrix b[ib:ib + m][jb:jb + n] are distributed stored
    const int iib = ib + 1; const int jjb = jb + 1; // C to Fortran
    const int mm_b = (m_b < 0 ? m : m_b);
    const int nn_b = (n_b < 0 ? n : n_b);
    int *desca = new int[9];
    int *descb = new int[9];
    int infoa, infob;
    const int lld_a = max(lda, 1);
    const int lld_b = max(ldb, 1);
    if(iprow == prow_root && ipcol == pcol_root)
    descinit(desca, &m,    &n,    &MB_ROW, &NB_COL, &iZERO,    &iZERO,    &root_ctxt, &lld_a, &infoa);
    else desca[1] = root_ctxt;
    descinit(descb, &mm_b, &nn_b, &MB_ROW, &NB_COL, &b_rowsrc, &b_colsrc, &ConTxt,    &lld_b, &infob);
    Blacs_barrier(ConTxt, "A");
    pdgemr2d(&m, &n, a, &iONE, &iONE, desca,
                     b, &iib,  &jjb,  descb, &ConTxt);
    
    delete[] desca; delete[] descb;
    Blacs_barrier(ConTxt, "A");
    return;
}

void Blacs_MatrixZScatter(const int m, const int n,
                          const complex<double> *a, const int lda,
                                complex<double> *b, const int ldb,
                          const int m_b, const int n_b, const int ib, const int jb,
                          const int b_rowsrc, const int b_colsrc,
                          const int prow_root, const int pcol_root, const int root_ctxt,
                          const int iprow, const int ipcol,
                          const int nprow, const int npcol, const int ConTxt) {
    // matrix a[m][n] are local stroed at (prow_root, pcol_root) in current ctxt
    // matrix b[ib:ib + m][jb:jb + n] are distributed stored
    const int iib = ib + 1; const int jjb = jb + 1; // C to Fortran
    const int mm_b = (m_b < 0 ? m : m_b);
    const int nn_b = (n_b < 0 ? n : n_b);
    int *desca = new int[9];
    int *descb = new int[9];
    int infoa, infob;
    const int lld_a = max(lda, 1);
    const int lld_b = max(ldb, 1);
    if(iprow == prow_root && ipcol == pcol_root)
    descinit(desca, &m,    &n,    &MB_ROW, &NB_COL, &iZERO,    &iZERO,    &root_ctxt, &lld_a, &infoa);
    else desca[1] = root_ctxt;
    descinit(descb, &mm_b, &nn_b, &MB_ROW, &NB_COL, &b_rowsrc, &b_colsrc, &ConTxt,    &lld_b, &infob);
    Blacs_barrier(ConTxt, "A");
    pzgemr2d(&m, &n, (MKL_Complex16*)a, &iONE, &iONE, desca,
                     (MKL_Complex16*)b, &iib,  &jjb,  descb, &ConTxt);
    
    delete[] desca; delete[] descb;
    Blacs_barrier(ConTxt, "A");
    return;
}

void Blacs_MatrixISum(const int ConTxt, const char *scope, const char *top, const int m, const int n,
                      int *A, const int lda, const int rdest, const int cdest) {
    if(m > 0 && n > 0 && lda > 0)
    igsum2d(&ConTxt, scope, top, &m, &n, A, &lda, &rdest, &cdest);
    return;
}

void Blacs_ReadDiag2Full(const char *filename, const int itime, 
                         const int matsize, const complex<double> alpha,
                         complex<double> *locmat, const int start_idx) {
    /* locmat is the distributed part of fullmat, fullmat[i, i] = alpha * diagdata[i] */
    ifstream inf(filename, ios::in|ios::binary);
    if(!inf.is_open()) { cerr << "ERROR: " << filename << " can't open in Blacs_ReadDiag2Full" << endl; exit(1); }
    const int readsize = matsize - start_idx;
    double *diagdata = new double[readsize];
    inf.seekg(sizeof(double) * itime * readsize, ios::beg);
    inf.read((char*)diagdata, sizeof(double) * readsize);
    inf.close();
    
    const int ndim_loc_row = Numroc(matsize, MB_ROW, myprow_group, nprow_group);
    int ii_loc_row, jj_loc_col;
    int iprow, jpcol;
    #pragma omp parallel for private(ii_loc_row, jj_loc_col, iprow, jpcol)
    for(int ii = start_idx; ii < matsize; ii++) {
        BlacsIdxglb2loc(ii, iprow, ii_loc_row, 0, matsize, MB_ROW, nprow_group);
        BlacsIdxglb2loc(ii, jpcol, jj_loc_col, 0, matsize, NB_COL, npcol_group);
        if(myprow_group == iprow && mypcol_group == jpcol) {
            locmat[ii_loc_row + jj_loc_col * ndim_loc_row] = alpha * diagdata[ii - start_idx];
        }
    }
    delete[] diagdata;
    
    MPI_Barrier(group_comm);
    return;
}

void Blacs_ReadFullMat(const char *filename, const int itime,
                       const int matsize, const complex<double> alpha,
                       vector<int> &locbeg, vector<int> &glbbeg, vector<int> &bcklen, // used for row
                       complex<double> *locmat, const int start_idx, const bool isConj) {
/*
 * read date from fullmat and update to locmat, 
 * which is the distributed part of fullmat 
 * locmat += alpha x locmat
*/
    ifstream inf(filename, ios::in|ios::binary);
    if(!inf.is_open()) { cerr << "ERROR: " << filename << " can't open in Blacs_ReadFullMat" << endl; exit(1); }
    const int readsize = matsize - start_idx;
    const int ndim_loc_row = Numroc(matsize, MB_ROW, myprow_group, nprow_group);
    int iprow, ii_loc;
    int jpcol, jj_loc;
    complex<double> *cdtmp = new complex<double>[MB_ROW];
    for(int jj = start_idx; jj < matsize; jj++) {
        BlacsIdxglb2loc(jj, jpcol, jj_loc, 0, matsize, NB_COL, npcol_group);
        if(mypcol_group != jpcol) continue;
        for(int irow_bck = 0; irow_bck < glbbeg.size(); irow_bck++) {
            const int readbcksize = bcklen[irow_bck] + min(glbbeg[irow_bck] - start_idx, 0);
            const int readglbbeg  = max(glbbeg[irow_bck] - start_idx, 0);
            const int writelogbeg = locbeg[irow_bck] - min(glbbeg[irow_bck] - start_idx, 0);
            inf.seekg(sizeof(complex<double>) * ( (size_t)itime * readsize * readsize
                                                + readglbbeg + (size_t)(jj - start_idx) * readsize ), ios::beg);
            inf.read((char*)cdtmp, sizeof(complex<double>) * readbcksize); // bcklen <= MB_ROW
            if(isConj) Zconj(readbcksize, cdtmp);
            Zaxpy(readbcksize, alpha, cdtmp, locmat + (writelogbeg + (size_t)jj_loc * ndim_loc_row));
        }
    }
    delete[] cdtmp;
    inf.close();

    MPI_Barrier(group_comm);
    return;
}

void Blacs_ReadFullMat(const char *filename, const int itime,
                       const int matsize, const complex<double> alpha,
                       complex<double> *locmat, const int start_idx, const bool isConj) {
    const int readsize = matsize - start_idx;
    complex<double> *fullmat = NULL;
    if(is_sub_root) {
        ifstream inf(filename, ios::in|ios::binary);
        if(!inf.is_open()) { cerr << "ERROR: " << filename << " can't read in Blacs_ReadFullMat" << endl; exit(1); }
        fullmat = new complex<double>[(size_t)readsize * readsize];
        inf.seekg(sizeof(complex<double>) * itime * readsize * readsize, ios::beg);
        inf.read((char*)fullmat, sizeof(complex<double>) * readsize * readsize);
        inf.close();
    }
    MPI_Barrier(group_comm);
    const int ndim_loc_row = Numroc(matsize, MB_ROW, myprow_group, nprow_group);
    const int ndim_loc_col = Numroc(matsize, NB_COL, mypcol_group, npcol_group);
    const size_t mn_loc = (size_t)ndim_loc_row * ndim_loc_col;
    complex<double> *tmploc = new complex<double>[max(mn_loc, (size_t)1)];
    Blacs_MatrixZScatter(readsize, readsize, fullmat, readsize, tmploc, ndim_loc_row,
                         matsize, matsize, start_idx, start_idx);
    if(isConj) Zconj(mn_loc, tmploc);
    Zaxpy(mn_loc, alpha, tmploc, locmat);
    delete[] tmploc;
    if(is_sub_root) delete[] fullmat;
    MPI_Barrier(group_comm);
    return;
}

double Pdasum(const int n, const double *x, const int ldx, const int incx,
              const int ix, const int jx, const int m_x, const int n_x,
              const int x_prowsrc, const int x_pcolsrc, 
              const int mb_row, const int nb_col, const int ConTxt) {
    if( (incx != m_x) && (incx != 1) )  { cerr << "ERROR in Pdasum: incx =/= m_x or 1" << endl; exit(1); }
    const int iix = ix + 1; const int jjx = jx + 1; // C to Fortran
    int *descx = new int[9];
    int infox;
    const int lld_x = max(ldx, 1);
    descinit(descx, &m_x, &n_x, &mb_row, &nb_col, &x_prowsrc, &x_pcolsrc, &ConTxt, &lld_x, &infox);
    double res;
    pdasum(&n, &res, x, &iix, &jjx, descx, &incx);
    delete[] descx;
    return res;
}

double Pdznrm2(const int n, const complex<double> *x, const int ldx, const int incx,
               const int ix, const int jx, const int m_x, const int n_x,
               const int x_prowsrc, const int x_pcolsrc, 
               const int mb_row, const int nb_col, const int ConTxt) {
    if( (incx != m_x) && (incx != 1) )  { cerr << "ERROR in Pdznrm2: incx =/= m_x or 1" << endl; exit(1); }
    const int iix = ix + 1; const int jjx = jx + 1; // C to Fortran
    int *descx = new int[9];
    int infox;
    const int lld_x = max(ldx, 1);
    descinit(descx, &m_x, &n_x, &mb_row, &nb_col, &x_prowsrc, &x_pcolsrc, &ConTxt, &lld_x, &infox);
    double res;
    pdznrm2(&n, &res, x, &iix, &jjx, descx, &incx);
    delete[] descx;
    return res;
}
                         
complex<double> Pzdotc_CC(const int n, complex<double> *x, complex<double> *y, 
                          const int x_colsrc, const int y_colsrc,
                          const int x_rowsrc, const int y_rowsrc,
                          const int iprow, const int nprow, const int ConTxt) {
    assert(n > 0);
    int *descx = new int[9];
    int *descy = new int[9];
    int infox, infoy;
    const int loc_nrow_x = numroc(&n, &MB_ROW, &iprow, &x_rowsrc, &nprow);
    const int loc_nrow_y = numroc(&n, &MB_ROW, &iprow, &y_rowsrc, &nprow);
    const int lld_x = max(loc_nrow_x, 1);
    const int lld_y = max(loc_nrow_y, 1);
    descinit(descx, &n, &iONE, &MB_ROW, &NB_COL, &x_rowsrc, &x_colsrc, &ConTxt, &lld_x, &infox);
    descinit(descy, &n, &iONE, &MB_ROW, &NB_COL, &y_rowsrc, &y_colsrc, &ConTxt, &lld_y, &infoy);
    if(infox != 0 || infoy != 0) { cerr << "ERROR in Pzdot_CC of descinit" << endl; exit(1); }
    complex<double> res;
    pzdotc(&n, (MKL_Complex16*)&res, (const MKL_Complex16*)x, &iONE, &iONE, descx, &iONE,
                                     (const MKL_Complex16*)y, &iONE, &iONE, descy, &iONE);
    delete[] descx; delete[] descy;
    Blacs_barrier(ConTxt, "A");
    return res;
}


complex<double> Pzdot_CC(const int n, complex<double> *x, complex<double> *y, 
                         const int x_colsrc, const int y_colsrc,
                         const int x_rowsrc, const int y_rowsrc,
                         const int iprow, const int nprow, const int ConTxt) {
    assert(n > 0);
    int *descx = new int[9];
    int *descy = new int[9];
    int infox, infoy;
    const int loc_nrow_x = numroc(&n, &MB_ROW, &iprow, &x_rowsrc, &nprow);
    const int loc_nrow_y = numroc(&n, &MB_ROW, &iprow, &y_rowsrc, &nprow);
    const int lld_x = max(loc_nrow_x, 1);
    const int lld_y = max(loc_nrow_y, 1);
    descinit(descx, &n, &iONE, &MB_ROW, &NB_COL, &x_rowsrc, &x_colsrc, &ConTxt, &lld_x, &infox);
    descinit(descy, &n, &iONE, &MB_ROW, &NB_COL, &y_rowsrc, &y_colsrc, &ConTxt, &lld_y, &infoy);
    if(infox != 0 || infoy != 0) { cerr << "ERROR in Pzdot_CC of descinit" << endl; exit(1); }
    complex<double> res;
    pzdotu(&n, (MKL_Complex16*)&res, (const MKL_Complex16*)x, &iONE, &iONE, descx, &iONE,
                                     (const MKL_Complex16*)y, &iONE, &iONE, descy, &iONE);
    delete[] descx; delete[] descy;
    Blacs_barrier(ConTxt, "A");
    return res;
}

complex<double> Pzdotc(const int n, const complex<double> *x, const int ldx, const int incx,
                                    const complex<double> *y, const int ldy, const int incy,
                       const int ix, const int jx, const int iy, const int jy,
                       const int m_x, const int n_x, const int m_y, const int n_y,
                       const int x_prowsrc, const int x_pcolsrc,
                       const int y_prowsrc, const int y_pcolsrc, 
                       const int mb_row, const int nb_col,
                       const int myprow, const int mypcol, const int nprow, const int npcol,
                       const int ConTxt) {
    assert(n > 0);
    if( (incx != m_x) && (incx != 1) )  { cerr << "ERROR in Pzdotc: incx =/= m_x or 1" << endl; exit(1); }
    if( (incy != m_y) && (incy != 1) )  { cerr << "ERROR in Pzdotc: incy =/= m_y or 1" << endl; exit(1); }
    const int iix = ix + 1; const int jjx = jx + 1; // C to Fortran
    const int iiy = iy + 1; const int jjy = jy + 1;
    int *descx = new int[9];
    int *descy = new int[9];
    int infox, infoy;
    const int lld_x = max(ldx, 1);
    const int lld_y = max(ldy, 1);
    descinit(descx, &m_x, &n_x, &mb_row, &nb_col, &x_prowsrc, &x_pcolsrc, &ConTxt, &lld_x, &infox);
    descinit(descy, &m_y, &n_y, &mb_row, &nb_col, &y_prowsrc, &y_pcolsrc, &ConTxt, &lld_y, &infoy);
    
    complex<double> res;
    pzdotc(&n, &res, x, &iix, &jjx, descx, &incx, y, &iiy, &jjy, descy, &incy);
    int ix_loc, iprow, jx_loc, jpcol;
    BlacsIdxglb2loc(ix, iprow, ix_loc, x_prowsrc, m_x, mb_row, nprow);
    BlacsIdxglb2loc(jx, jpcol, jx_loc, x_pcolsrc, n_x, nb_col, npcol);

    if(myprow == iprow && mypcol == jpcol) Zgebs2d(ConTxt, "ALL", "I", 1, 1, &res, 1); 
    else Zgebr2d(ConTxt, "ALL", "I", 1, 1, &res, 1, iprow, jpcol);
    Blacs_barrier(ConTxt, "A");
    
    delete[] descx; delete[] descy;
    Blacs_barrier(ConTxt, "A");
    return res;
}

void Pzscal(const int n, const complex<double> alpha, complex<double> *x, const int ldx, const int incx,
            const int ix, const int jx, const int m_x, const int n_x,
            const int x_prowsrc, const int x_pcolsrc, 
            const int mb_row, const int nb_col, const int ConTxt) {
    assert(n > 0);
    if( (incx != m_x) && (incx != 1) )  { cerr << "ERROR in Pzdotc: incx =/= m_x or 1" << endl; exit(1); }
    const int iix = ix + 1; const int jjx = jx + 1; // C to Fortran
    int *descx = new int[9];
    int infox;
    const int lld_x = max(ldx, 1);
    descinit(descx, &m_x, &n_x, &mb_row, &nb_col, &x_prowsrc, &x_pcolsrc, &ConTxt, &lld_x, &infox);
    pzscal(&n, &alpha, x, &iix, &jjx, descx, &incx);
    delete[] descx;
    return;
}

void Pdgemv(const char *transa, const int m, const int n, 
            const double alpha, const double *a, const int lda,
                                const double *x, const int ldx,
            const double beta,        double *y, const int ldy,
            const int incx, const int incy,
            const int m_a, const int n_a, const int m_x, const int n_x, const int m_y, const int n_y,
            const int ia, const int ja, const int ix, const int jx, const int iy, const int jy,
            const int a_prowsrc, const int a_pcolsrc,
            const int x_prowsrc, const int x_pcolsrc,
            const int y_prowsrc, const int y_pcolsrc, 
            const int mb_row, const int nb_col, const int ConTxt) {
    assert(m > 0); assert(n > 0);
    const int iia = ia + 1; const int jja = ja + 1; // C to Fortran
    const int iix = ix + 1; const int jjx = jx + 1;
    const int iiy = iy + 1; const int jjy = jy + 1;
    int mm_a = (m_a < 0 ? m : m_a);
    int nn_a = (n_a < 0 ? n : n_a);
    if(m_a < 0 || n_a < 0) { if( (!strcmp(transa, "T")) || (!strcmp(transa, "C")) ) { mm_a = n; nn_a = m; } }
    const int mm_x = (m_x < 0 ? mm_a : m_x);
    const int nn_x = (n_x < 0 ? 1 : n_x);
    const int mm_y = (m_y < 0 ? mm_a : m_y);
    const int nn_y = (n_y < 0 ? 1 : n_y);
    if( (incx != mm_x) && (incx != 1) )  { cerr << "ERROR in Pdgemv: incx =/= m_x or 1" << endl; exit(1); }
    if( (incy != mm_y) && (incy != 1) )  { cerr << "ERROR in Pdgemv: incy =/= m_y or 1" << endl; exit(1); }
    
    int *desca = new int[9];
    int *descx = new int[9];
    int *descy = new int[9];
    int infoa, infox, infoy;
    const int lld_a = max(lda, 1);
    const int lld_x = max(ldx, 1);
    const int lld_y = max(ldy, 1);
    descinit(desca, &mm_a, &nn_a, &mb_row, &nb_col, &a_prowsrc, &a_pcolsrc, &ConTxt, &lld_a, &infoa);
    descinit(descx, &mm_x, &nn_x, &mb_row, &nb_col, &x_prowsrc, &x_pcolsrc, &ConTxt, &lld_x, &infox);
    descinit(descy, &mm_y, &nn_y, &mb_row, &nb_col, &y_prowsrc, &y_pcolsrc, &ConTxt, &lld_y, &infoy);

    pdgemv(transa, &m, &n, &alpha, a, &iia, &jja, desca,
                                   x, &iix, &jjx, descx, &incx,
                            &beta, y, &iiy, &jjy, descy, &incy);
    
    delete[] desca; delete[] descx; delete[] descy;
    Blacs_barrier(ConTxt, "A");
    return;
}

void Pzgemv(const char *transa, const int m, const int n, 
            const complex<double> alpha, const complex<double> *a, const int lda,
                                         const complex<double> *x, const int ldx,
            const complex<double> beta,        complex<double> *y, const int ldy,
            const int incx, const int incy,
            const int m_a, const int n_a, const int m_x, const int n_x, const int m_y, const int n_y,
            const int ia, const int ja, const int ix, const int jx, const int iy, const int jy,
            const int a_prowsrc, const int a_pcolsrc,
            const int x_prowsrc, const int x_pcolsrc,
            const int y_prowsrc, const int y_pcolsrc, 
            const int mb_row, const int nb_col, const int ConTxt) {
    assert(m > 0); assert(n > 0);
    const int iia = ia + 1; const int jja = ja + 1; // C to Fortran
    const int iix = ix + 1; const int jjx = jx + 1;
    const int iiy = iy + 1; const int jjy = jy + 1;
    int mm_a = (m_a < 0 ? m : m_a);
    int nn_a = (n_a < 0 ? n : n_a);
    if(m_a < 0 || n_a < 0) { if( (!strcmp(transa, "T")) || (!strcmp(transa, "C")) ) { mm_a = n; nn_a = m; } }
    const int mm_x = (m_x < 0 ? mm_a : m_x);
    const int nn_x = (n_x < 0 ? 1 : n_x);
    const int mm_y = (m_y < 0 ? mm_a : m_y);
    const int nn_y = (n_y < 0 ? 1 : n_y);
    if( (incx != mm_x) && (incx != 1) )  { cerr << "ERROR in Pzgemv: incx =/= m_x or 1" << endl; exit(1); }
    if( (incy != mm_y) && (incy != 1) )  { cerr << "ERROR in Pzgemv: incy =/= m_y or 1" << endl; exit(1); }
    
    int *desca = new int[9];
    int *descx = new int[9];
    int *descy = new int[9];
    int infoa, infox, infoy;
    const int lld_a = max(lda, 1);
    const int lld_x = max(ldx, 1);
    const int lld_y = max(ldy, 1);
    descinit(desca, &mm_a, &nn_a, &mb_row, &nb_col, &a_prowsrc, &a_pcolsrc, &ConTxt, &lld_a, &infoa);
    descinit(descx, &mm_x, &nn_x, &mb_row, &nb_col, &x_prowsrc, &x_pcolsrc, &ConTxt, &lld_x, &infox);
    descinit(descy, &mm_y, &nn_y, &mb_row, &nb_col, &y_prowsrc, &y_pcolsrc, &ConTxt, &lld_y, &infoy);

    pzgemv(transa, &m, &n, &alpha, a, &iia, &jja, desca,
                                   x, &iix, &jjx, descx, &incx,
                            &beta, y, &iiy, &jjy, descy, &incy);
    
    delete[] desca; delete[] descx; delete[] descy;
    Blacs_barrier(ConTxt, "A");
    return;
}

void Pzgemm(const char *transa, const char *transb, const int m, const int n, const int k, 
            const complex<double> alpha, const complex<double> *a, const int lda,
                                         const complex<double> *b, const int ldb, 
            const complex<double> beta,        complex<double> *c, const int ldc,
            const int m_a, const int n_a, const int m_b, const int n_b, const int m_c, const int n_c,
            const int ia, const int ja, const int ib, const int jb, const int ic, const int jc,
            const int a_prowsrc, const int a_pcolsrc,
            const int b_prowsrc, const int b_pcolsrc,
            const int c_prowsrc, const int c_pcolsrc, 
            const int mb_row, const int nb_col, const int ConTxt) {
    assert(m > 0); assert(n > 0); assert(k > 0);
    const int iia = ia + 1; const int jja = ja + 1; // C to Fortran
    const int iib = ib + 1; const int jjb = jb + 1;
    const int iic = ic + 1; const int jjc = jc + 1;
    int mm_a = (m_a < 0 ? m : m_a);
    int nn_a = (n_a < 0 ? k : n_a);
    int mm_b = (m_b < 0 ? k : m_b);
    int nn_b = (n_b < 0 ? n : n_b);
    if(m_a < 0 || n_a < 0) { if( (!strcmp(transa, "T")) || (!strcmp(transa, "C")) ) { mm_a = k; nn_a = m; } }
    if(m_b < 0 || n_b < 0) { if( (!strcmp(transb, "T")) || (!strcmp(transb, "C")) ) { mm_b = n; nn_b = k; } }
    const int mm_c = (m_c < 0 ? m : m_c);
    const int nn_c = (n_c < 0 ? n : n_c);
    int *desca = new int[9];
    int *descb = new int[9];
    int *descc = new int[9];
    int infoa, infob, infoc;
    const int lld_a = max(lda, 1);
    const int lld_b = max(ldb, 1);
    const int lld_c = max(ldc, 1);
    descinit(desca, &mm_a, &nn_a, &mb_row, &nb_col, &a_prowsrc, &a_pcolsrc, &ConTxt, &lld_a, &infoa);
    descinit(descb, &mm_b, &nn_b, &mb_row, &nb_col, &b_prowsrc, &b_pcolsrc, &ConTxt, &lld_b, &infob);
    descinit(descc, &mm_c, &nn_c, &mb_row, &nb_col, &c_prowsrc, &c_pcolsrc, &ConTxt, &lld_c, &infoc);
    
    pzgemm(transa, transb, &m, &n, &k, &alpha, a, &iia, &jja, desca, 
                                               b, &iib, &jjb, descb,
                                        &beta, c, &iic, &jjc, descc);
    
    delete[] desca; delete[] descb; delete[] descc;
    Blacs_barrier(ConTxt, "A");
    return;
}

void Pdgemm(const char *transa, const char *transb, const int m, const int n, const int k, 
            const double alpha, const double *a, const int lda,
                                const double *b, const int ldb, 
            const double beta,        double *c, const int ldc,
            const int m_a, const int n_a, const int m_b, const int n_b, const int m_c, const int n_c,
            const int ia, const int ja, const int ib, const int jb, const int ic, const int jc,
            const int a_prowsrc, const int a_pcolsrc,
            const int b_prowsrc, const int b_pcolsrc,
            const int c_prowsrc, const int c_pcolsrc, 
            const int mb_row, const int nb_col, const int ConTxt) {
    assert(m > 0); assert(n > 0); assert(k > 0);
    const int iia = ia + 1; const int jja = ja + 1; // C to Fortran
    const int iib = ib + 1; const int jjb = jb + 1;
    const int iic = ic + 1; const int jjc = jc + 1;
    int mm_a = (m_a < 0 ? m : m_a);
    int nn_a = (n_a < 0 ? k : n_a);
    int mm_b = (m_b < 0 ? k : m_b);
    int nn_b = (n_b < 0 ? n : n_b);
    if(m_a < 0 || n_a < 0) { if( (!strcmp(transa, "T")) || (!strcmp(transa, "C")) ) { mm_a = k; nn_a = m; } }
    if(m_b < 0 || n_b < 0) { if( (!strcmp(transb, "T")) || (!strcmp(transb, "C")) ) { mm_b = n; nn_b = k; } }
    const int mm_c = (m_c < 0 ? m : m_c);
    const int nn_c = (n_c < 0 ? n : n_c);
    int *desca = new int[9];
    int *descb = new int[9];
    int *descc = new int[9];
    int infoa, infob, infoc;
    const int lld_a = max(lda, 1);
    const int lld_b = max(ldb, 1);
    const int lld_c = max(ldc, 1);
    descinit(desca, &mm_a, &nn_a, &mb_row, &nb_col, &a_prowsrc, &a_pcolsrc, &ConTxt, &lld_a, &infoa);
    descinit(descb, &mm_b, &nn_b, &mb_row, &nb_col, &b_prowsrc, &b_pcolsrc, &ConTxt, &lld_b, &infob);
    descinit(descc, &mm_c, &nn_c, &mb_row, &nb_col, &c_prowsrc, &c_pcolsrc, &ConTxt, &lld_c, &infoc);
    
    pdgemm(transa, transb, &m, &n, &k, &alpha, a, &iia, &jja, desca, 
                                               b, &iib, &jjb, descb,
                                        &beta, c, &iic, &jjc, descc);
    
    delete[] desca; delete[] descb; delete[] descc;
    Blacs_barrier(ConTxt, "A");
    return;
}

void Pzgecom(const int n, const int ldd_loc, const complex<double> alpha, // com: commute
             const complex<double> *a, const complex<double> *b,
                   complex<double> *r) {
    assert(n > 0);
    /* r = [a, b] = ab - ba */
    Pzgemm("N", "N", n, n, n,  1.0, a, ldd_loc, b, ldd_loc, 0.0, r, ldd_loc);
    Pzgemm("N", "N", n, n, n, -1.0, b, ldd_loc, a, ldd_loc, 1.0, r, ldd_loc);
    const int nrows_loc = Numroc(n, MB_ROW, myprow_group, nprow_group);
    const int ncols_loc = Numroc(n, NB_COL, mypcol_group, npcol_group);
    Zscal((long)nrows_loc * ncols_loc, alpha, r, 1);
    MPI_Barrier(group_comm);

    return;
}

void Pzlacpy(const char *uplo, const int m, const int n, 
             const complex<double> *a, const int lda, complex<double> *b, const int ldb,
             const int m_a, const int n_a, const int m_b, const int n_b,
             const int ia, const int ja, const int ib, const int jb,
             const int a_prowsrc, const int a_pcolsrc,
             const int b_prowsrc, const int b_pcolsrc,
             const int mb_row, const int nb_col, const int ConTxt) {
    const int iia = ia + 1; const int jja = ja + 1; // C to Fortran
    const int iib = ib + 1; const int jjb = jb + 1;
    const int mm_a = (m_a < 0 ? m : m_a);
    const int nn_a = (n_a < 0 ? n : n_a);
    const int mm_b = (m_b < 0 ? m : m_b);
    const int nn_b = (n_b < 0 ? n : n_b);
    int *desca = new int[9];
    int *descb = new int[9];
    int infoa, infob;
    const int lld_a = max(lda, 1);
    const int lld_b = max(ldb, 1);
    descinit(desca, &mm_a, &nn_a, &mb_row, &nb_col, &a_prowsrc, &a_pcolsrc, &ConTxt, &lld_a, &infoa);
    descinit(descb, &mm_b, &nn_b, &mb_row, &nb_col, &b_prowsrc, &b_pcolsrc, &ConTxt, &lld_b, &infob);
    
    pzlacpy(uplo, &m, &n, a, &iia, &jja, desca, b, &iib, &jjb, descb);
    delete[] desca; delete[] descb;
    Blacs_barrier(ConTxt, "A");
    
    return;
}

void Pzcopy(const int m, const int n,
            const complex<double> *a, const int lda, 
                  complex<double> *b, const int ldb,
            const int m_a, const int n_a,
            const int m_b, const int n_b,
            const int ia, const int ja, const int ib, const int jb,
            const int a_prowsrc, const int a_pcolsrc,
            const int b_prowsrc, const int b_pcolsrc,
            const int mb_row, const int nb_col, const int ConTxt) {
    const int iia = ia + 1; const int jja = ja + 1; // C to Fortran
    const int iib = ib + 1; const int jjb = jb + 1;
    const int mm_a = (m_a < 0 ? m : m_a);
    const int nn_a = (n_a < 0 ? n : n_a);
    const int mm_b = (m_b < 0 ? m : m_b);
    const int nn_b = (n_b < 0 ? n : n_b);
    int *desca = new int[9];
    int *descb = new int[9];
    int infoa, infob;
    const int lld_a = max(lda, 1);
    const int lld_b = max(ldb, 1);
    descinit(desca, &mm_a, &nn_a, &mb_row, &nb_col, &a_prowsrc, &a_pcolsrc, &ConTxt, &lld_a, &infoa);
    descinit(descb, &mm_b, &nn_b, &mb_row, &nb_col, &b_prowsrc, &b_pcolsrc, &ConTxt, &lld_b, &infob);
    pzgemr2d(&m, &n, (MKL_Complex16*)a, &iia, &jja, desca,
                     (MKL_Complex16*)b, &iib, &jjb, descb, &ConTxt);
    delete[] desca; delete[] descb;
    Blacs_barrier(ConTxt, "A");
    
    return;
}

void Pzgeadd(const char *transa, const int m, const int n,
             const complex<double> alpha, const complex<double> *a, const int lda, 
             const complex<double> beta,        complex<double> *b, const int ldb,
             const int m_a, const int n_a,
             const int m_b, const int n_b,
             const int ia, const int ja, const int ib, const int jb,
             const int a_prowsrc, const int a_pcolsrc,
             const int b_prowsrc, const int b_pcolsrc,
             const int mb_row, const int nb_col, const int ConTxt) {
// b = alpha x a + beta x b
    const int iia = ia + 1; const int jja = ja + 1; // C to Fortran
    const int iib = ib + 1; const int jjb = jb + 1;
    int mm_a = (m_a < 0 ? m : m_a);
    int nn_a = (n_a < 0 ? n : n_a);
    const int mm_b = (m_b < 0 ? m : m_b);
    const int nn_b = (n_b < 0 ? n : n_b);
    if(m_a < 0 || n_a < 0) { if( (!strcmp(transa, "T")) || (!strcmp(transa, "C")) ) { mm_a = n; nn_a = m; } }
    int *desca = new int[9];
    int *descb = new int[9];
    int infoa, infob;
    const int lld_a = max(lda, 1);
    const int lld_b = max(ldb, 1);
    descinit(desca, &mm_a, &nn_a, &mb_row, &nb_col, &a_prowsrc, &a_pcolsrc, &ConTxt, &lld_a, &infoa);
    descinit(descb, &mm_b, &nn_b, &mb_row, &nb_col, &b_prowsrc, &b_pcolsrc, &ConTxt, &lld_b, &infob);
    pzgeadd(transa, &m, &n, &alpha, a, &iia, &jja, desca,
                            &beta,  b, &iib, &jjb, descb);
    delete[] desca; delete[] descb;
    Blacs_barrier(ConTxt, "A");
    
    return;
}

void Pzheev(const char *jobz, const int n, const complex<double> *a, const int lda,
            double *w, complex<double> *z, const int ldz, // output: w-eigenvalues, z-eigenvectors
            const int m_a, const int n_a, const int m_z, const int n_z,
            const int ia, const int ja, const int iz, const int jz, const char *uplo,
            const int a_prowsrc, const int a_pcolsrc,
            const int z_prowsrc, const int z_pcolsrc,
            const int mb_row, const int nb_col, const int ConTxt) {
    const int iia = ia + 1; const int jja = ja + 1; // C to Fortran
    const int iiz = iz + 1; const int jjz = jz + 1;
    const int mm_a = (m_a < 0 ? n : m_a);
    const int nn_a = (n_a < 0 ? n : n_a);
    const int mm_z = (m_z < 0 ? n : m_z);
    const int nn_z = (n_z < 0 ? n : n_z);
    int *desca = new int[9];
    int *descz = new int[9];
    int infoa, infoz;
    const int lld_a = max(lda, 1);
    const int lld_z = max(ldz, 1);
    descinit(desca, &mm_a, &nn_a, &mb_row, &nb_col, &a_prowsrc, &a_pcolsrc, &ConTxt, &lld_a, &infoa);
    descinit(descz, &mm_z, &nn_z, &mb_row, &nb_col, &z_prowsrc, &z_pcolsrc, &ConTxt, &lld_z, &infoz);

    complex<double> worktmp;
    double rworktmp;
    int info;
    pzheev(jobz, uplo, &n, a, &iia, &jja, desca, w, z, &iiz, &jjz, descz, &worktmp, &iMONE, &rworktmp, &iMONE, &info);
    const int lwork = round(real(worktmp));
    const int lrwork = round(rworktmp);
    complex<double> *work = new complex<double>[lwork];
    double         *rwork = new         double[lrwork];
    
    pzheev(jobz, uplo, &n, a, &iia, &jja, desca, w, z, &iiz, &jjz, descz, work, &lwork, rwork, &lrwork, &info);
    
    delete[] work; delete[] rwork;
    delete[] desca; delete[] descz;
    Blacs_barrier(ConTxt, "A");

    return;
}

//**************************************************************************************
//
//                            Spherical Harmonic Function
//
//**************************************************************************************

double RealSphHarFun(const int l, const int m, const double theta, const double phi) { // real spherical harmonics function
         if(m == 0)           return sph_legendre(l, m, theta);
    else if(m >= -l && m < 0) return sqrt(2.0) * pow(-1, m) * sph_legendre(l, -m, theta) * sin(-m * phi);
    else if(m <=  l && m > 0) return sqrt(2.0) * pow(-1, m) * sph_legendre(l,  m, theta) * cos( m * phi);
    else { cout << "RealSphHarFun ERROR: m = " << m << ", not belongs to " << "[-" << l << ",+" << l << ']' << endl; exit(1); }
}

//**************************************************************************************
//
//                            Fast Fourier Transform
//
//**************************************************************************************
void Pzfft3SetMallocSize(const int n0, const int n1, const int n2,
                         int &local_n0, int &local_0_start, int &total_local_size, MPI_Comm &comm) {
    ptrdiff_t n0_loc, n0_start_loc, data_loc_size;
    fftw_mpi_init();
    data_loc_size = fftw_mpi_local_size_3d(n0, n1, n2, comm, &n0_loc, &n0_start_loc);
    fftw_mpi_cleanup();
    local_n0 = n0_loc; local_0_start = n0_start_loc; total_local_size = data_loc_size;
    return;
}

void Pzfft3(complex<double> *data, const int n[], const int forback, MPI_Comm &comm) { 
/*
    local fft, data -> fft(data)
    data[loc_n0 x n1 x n2] row-major stored in one column processes
    in which loc_n0 is determined by fftw_mpi_local_size_3d(n0, n1, n2, comm, &n0_loc, &n0_start_loc) somewhere
    forback: FFTW_FORWARD( e^(-iRG) ) or FFTW_BACKWARD( e^(+IRG) )
*/
    fftw_plan plan = NULL;
    fftw_mpi_init(); 
   
    plan = fftw_mpi_plan_dft_3d(n[0], n[1], n[2], (fftw_complex*)data, (fftw_complex*)data, comm, forback, FFTW_ESTIMATE);
    fftw_execute(plan);
    fftw_destroy_plan(plan);
    
    fftw_mpi_cleanup();
    return;
}

void BlockCyclicToMpiFFTW3d(const complex<double> *in, // compacted with global/local size totin/totin_loc
                            const int totin, const int totin_loc,
                            complex<double> *out,      // local with mpi-fftw data structure
                            const int nin[], const int nout[],
                            const int *loc_nout0, const int *loc_out0_start,
                            const int *idxin) { // global index of in
/*
    This routine scatter the compacted "in" data with one column block-cyclic storage
    in "onecol_root_prow" rank to other ranks within "col_comm"
    First gather data from all processes in one column to the "onecol_root_prow" rank.
    Second rearrange.
    Third scatter the root date to "out" in one-column ranks.
*/
    complex<double> *rootcomp = NULL;
    complex<double> *rootfull = NULL;
    if(myprow_onecol == onecol_root_prow) { 
        rootcomp = new complex<double>[totin]();
        rootfull = new complex<double>[ nout[0] * nout[1] * nout[2] ]();
    }
    MPI_Barrier(col_comm);
    // First, gather to the root process
    Blacs_MatrixZGather(totin, 1, in, totin_loc, rootcomp, totin,
                        totin, 1, 0, 0, 0, 0,
                        onecol_root_prow, onecol_root_pcol, ctxt_only_onecol_root,
                        myprow_onecol, mypcol_onecol, nprow_onecol, npcol_onecol, ctxt_onecol);
    if(myprow_onecol == onecol_root_prow) {
        int ii, jj, kk;
        // Second, set values in the root process
        #pragma omp parallel for private(ii, jj, kk)
        for(int ijk = 0; ijk < totin; ijk++) {
            IdxNat1toSym3(idxin[ijk], ii, jj, kk, nin[0], nin[1], nin[2]); //  - (nin + 1) / 2 < ii/jj/kk < nin / 2 + 1
            rootfull[IdxSym3toNat1(ii, jj, kk, nout[0], nout[1], nout[2])] = rootcomp[ijk];
        }
        // Finally, scatter to other processes
        Zcopy(loc_nout0[onecol_root_prow] * nout[1] * nout[2],
              rootfull + loc_out0_start[onecol_root_prow] * nout[1] * nout[2], 1,
              out, 1);
        for(int rkrow = 0; rkrow < nprow_onecol; rkrow++) if(rkrow != onecol_root_prow)
        MPI_Send(rootfull + loc_out0_start[rkrow] * nout[1] * nout[2], loc_nout0[rkrow] * nout[1] * nout[2],
                 MPI_CXX_DOUBLE_COMPLEX, rkrow, loc_out0_start[rkrow], col_comm);
    }
    else {
        MPI_Status mpi_status;
        MPI_Recv(out, loc_nout0[myprow_onecol] * nout[1] * nout[2], MPI_CXX_DOUBLE_COMPLEX, onecol_root_prow,
                 loc_out0_start[myprow_onecol], col_comm, &mpi_status);
    }
    MPI_Barrier(col_comm);
    if(myprow_onecol == onecol_root_prow) { delete[] rootcomp; delete[] rootfull; }
    MPI_Barrier(col_comm);
    return;
}

void MpiFFTW3dToBlockCyclic(const complex<double> *in, 
                            complex<double> *out, 
                            const int totout, const int totout_loc,
                            const int nin[], const int nout[],
                            const int *loc_nin0, const int *loc_in0_start,
                            const int *idxout) {
/* reverse of BlockCyclicToMpiFFTW3d */
    complex<double> *rootcomp = NULL;
    complex<double> *rootfull = NULL;
    if(myprow_onecol == onecol_root_prow) { 
        rootcomp = new complex<double>[totout]();
        rootfull = new complex<double>[ nin[0] * nin[1] * nin[2] ]();
    }
    MPI_Barrier(col_comm);
    // First, send and recv to the root process
    if(myprow_onecol == onecol_root_prow) {
        Zcopy(loc_nin0[onecol_root_prow] * nin[1] * nin[2],
              in, 1, rootfull + loc_in0_start[onecol_root_prow] * nin[1] * nin[2], 1);
        MPI_Status mpi_status;
        for(int rkrow = 0; rkrow < nprow_onecol; rkrow++) if(rkrow != onecol_root_prow)
        MPI_Recv(rootfull + loc_in0_start[rkrow] * nin[1] * nin[2], 
                 loc_nin0[rkrow] * nin[1] * nin[2], MPI_CXX_DOUBLE_COMPLEX, rkrow,
                 loc_in0_start[rkrow], col_comm, &mpi_status);
    }
    else {
        MPI_Send(in, loc_nin0[myprow_onecol] * nin[1] * nin[2], MPI_CXX_DOUBLE_COMPLEX, onecol_root_prow,
                 loc_in0_start[myprow_onecol], col_comm);
    }
    MPI_Barrier(col_comm);
    
    // Second, rearrange to compacted
    if(myprow_onecol == onecol_root_prow) {
        int ii, jj, kk;
        #pragma omp parallel for private(ii, jj, kk)
        for(int ijk = 0; ijk < totout; ijk++) {
            IdxNat1toSym3(idxout[ijk], ii, jj, kk, nout[0], nout[1], nout[2]); // - (nout + 1) / 2 < ii/jj/kk < nout/2 + 1
            rootcomp[ijk] = rootfull[IdxSym3toNat1(ii, jj, kk, nin[0], nin[1], nin[2])];
        }
    }
    MPI_Barrier(col_comm);

    // Finally, scatter to all onecol ranks
    Blacs_MatrixZScatter(totout, 1, rootcomp, totout, out, totout_loc, totout, 1, 0, 0, 0, 0,
                         onecol_root_prow, onecol_root_pcol, ctxt_only_onecol_root,
                         myprow_onecol, mypcol_onecol, nprow_onecol, npcol_onecol, ctxt_onecol);
    MPI_Barrier(col_comm);
    if(myprow_onecol == onecol_root_prow) { delete[] rootcomp; delete[] rootfull; }
    MPI_Barrier(col_comm);

    return;
}

void Zfft_3d(complex<double> *data, const int n[], const int forback) {
/*
    forback: FFTW_FORWARD( e^(-iRG) ) or FFTW_BACKWARD( e^(+IRG) )
*/
    fftw_plan p;
    fftw_init_threads();
    fftw_plan_with_nthreads(omp_get_max_threads());
    
    p = fftw_plan_dft_3d(n[0], n[1], n[2], (fftw_complex*)data, (fftw_complex*)data, forback, FFTW_ESTIMATE);
    
    fftw_execute(p);
    fftw_destroy_plan(p);
    fftw_cleanup_threads();

    return;
}

void Zfft_1d(const int N, complex<double> *in, complex<double> *out, const int forback) {
/*  
    "in" and "out" can be the same for a in-place transform
    forback: FFTW_FORWARD( e^(-iRG) ) or FFTW_BACKWARD( e^(+IRG) )
*/
    fftw_plan p;
    fftw_init_threads();
    fftw_plan_with_nthreads(omp_get_max_threads());
    
    p = fftw_plan_dft_1d(N, (fftw_complex*)in, (fftw_complex*)out, forback, FFTW_ESTIMATE);
    
    fftw_execute(p);
    fftw_destroy_plan(p);
    fftw_cleanup_threads();
    return;
}

void Zr2cfft_1d(const int N, double *in, complex<double> *out) {
/*  always FFTW_FORWARD( e^(-iRG) ) 
    "in" with length of N, "out" with length of (N / 2 + 1)
*/
    fftw_plan p;
    fftw_init_threads();
    fftw_plan_with_nthreads(omp_get_max_threads());
    
    p = fftw_plan_dft_r2c_1d(N, in, (fftw_complex*)out, FFTW_ESTIMATE);
    
    fftw_execute(p);
    fftw_destroy_plan(p);
    fftw_cleanup_threads();
    return;
}

void Zc2rfft_1d(const int N, complex<double> *in, double *out) {
/*  always FFTW_BACKWARD( e^(+iRG) )
    "in" with length of (N / 2 + 1), "out" with length of N
*/ 
    fftw_plan p;
    fftw_init_threads();
    fftw_plan_with_nthreads(omp_get_max_threads());
    
    p = fftw_plan_dft_c2r_1d(N, (fftw_complex*)in, out, FFTW_ESTIMATE);
    
    fftw_execute(p);
    fftw_destroy_plan(p);
    fftw_cleanup_threads();
    return;
}

void DselfCrossCorrelation(const int N, double *inout) {
    double *in = new double[2 * N - 1]();
    complex<double> *midout = new complex<double>[(2 * N - 1) / 2 + 1];
    double avg = accumulate(inout, inout + N, 0.0) / N; // DON't use Dasum
    #pragma omp parallel for
    for(int ii = 0; ii < N; ii++) in[ii] = inout[ii] - avg;
    Zr2cfft_1d(2 * N - 1, in, midout);
    #pragma omp parallel for
    for(int ii = 0; ii < (2 * N - 1) / 2 + 1; ii++) midout[ii] *= conj(midout[ii]);
    Zc2rfft_1d(2 * N - 1, midout, in);
    #pragma omp parallel for
    for(int ii = 0; ii < N; ii++) inout[ii] = ( 1.0 / (2 * N - 1) / N ) * in[ii];
    //first "2 x N - 1" for ifft, second "N" for integral to summation

    delete[] in;
    delete[] midout;
    return;
}

void CumulativeIntegralTwice(const double alpha, const int N, double *inout) {
 /* calculate res(x) = alpha x \int_0^x dx1 \int_0^x1 dx2 inout(x2)
    use trapezoidal rule */
    double *tmpsum = new double[N]();
    for(int ii = 1; ii < N; ii++) tmpsum[ii] = tmpsum[ii - 1] + (inout[ii - 1] + inout[ii]) / 2.0;
    inout[0] = 0.0;
    for(int ii = 1; ii < N; ii++) inout[ii] = inout[ii - 1] + (tmpsum[ii - 1] + tmpsum[ii]) / 2.0;
    Dscal(N, alpha, inout);
    delete[] tmpsum;
    return;
}

//**************************************************************************************
//
//                               Least Squares Fitting
//
//**************************************************************************************
void LinearLSF(const int N, const double *x, const double *y,
               double &a, double &b) {
/* y = a + bx, return a and b*/
    const double sumx  = accumulate(x, x + N, 0.0);
    const double sumy  = accumulate(y, y + N, 0.0);
    const double sumx2 = Ddot(N, x, x);
    const double sumxy = Ddot(N, x, y);

    b = (N * sumxy - sumx * sumy) / (N * sumx2 - sumx * sumx);
    a = (sumy - b * sumx) / N;

    return;
}

double DirectProportionLSF(const int N, const double *x, const double *y) {
/* y = kx, return k */
    const double sumx2 = Ddot(N, x, x);
    const double sumxy = Ddot(N, x, y);

    return sumxy / sumx2;
}

double GassuianFitting1(const int N, const double x0, const double dx,
                        const double *mlny) {
/*
       y = exp( - (t / tau)^2 / 2 )   
    mlny = -lny = ( 1 / (2 x tau^2) ) x t^2
     sqrt(mlny) = (1 / sqrt(2) / tau) x t 
       Y = kX, Y = sqrt(mlny), X = t, tau = 1 / ( sqrt(2) k )
*/
    double *X = new double[N];
    double *Y = new double[N];
    #pragma omp parallel for
    for(int i = 0; i < N; i++) {
        X[i] = x0 + i * dx;
        Y[i] = sqrt(mlny[i]);
    }
    double res = 1.0 / sqrt(2.0) / DirectProportionLSF(N, X, Y);
    delete[] X; delete[] Y;
    return res;
}
