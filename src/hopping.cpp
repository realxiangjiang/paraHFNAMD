#include "hopping.h"

void NormalizeProbability(double *probmat, const int nstates, const bool is_reset_diag) {
/* normalize the probability matrix column by column */
    const int ndim_loc_row = Numroc(nstates, MB_ROW, myprow_group, nprow_group);
    double colsum;
    int jcol_loc, jpcol;
    int irow_loc, iprow;
    for(int jcol = 0; jcol < nstates; jcol++) {
        colsum = Pdasum(nstates, probmat, ndim_loc_row, 1, 0, jcol, nstates, nstates);
        BlacsIdxglb2loc(jcol, jpcol, jcol_loc, 0, nstates, NB_COL, npcol_group);
        if(mypcol_group == jpcol) {
            if(is_reset_diag) {
                if(colsum < 1.0) {
                    BlacsIdxglb2loc(jcol, iprow, irow_loc, 0, nstates, MB_ROW, nprow_group);
                    if(myprow_group == iprow)
                    probmat[irow_loc + jcol_loc * ndim_loc_row] = 1.0 - colsum;
                }
            }
            if( !is_reset_diag || !(colsum < 1.0) )
            Dscal(ndim_loc_row, 1.0 / colsum, probmat + jcol_loc * ndim_loc_row);
        }
        MPI_Barrier(group_comm);
    }
    return;
}

void FSSHprob(double *probmat, complex<double> *vmat, // output
              const int nstates,
              complex<double> **c_onsite, complex<double> **c_midsite,
              const complex<double> *coeff) {
/*
    Tully, JCP, 93, 1061(1990)
    current state k, target state j

    dC/dt = H(t)C(t), H(t) = V/(ihbar) - NAC, H_ij = -(H_ji)*

    b_jk = 2 x Im(c*_j x c_k x V_jk) / hbar - 2 x Re(c*_j x c_k x NAC_jk)   Eq.(14)
         = 2 x Re(c*_j x c_k x V_jk / (i x hbar) - c*_j x c_k x NAC_jk)
         = 2 x Re(c*_j x c_k x H_jk)

    g_kj = b_jk / (c_k x c*_k) x dt          Eq.(19)
         = 2 x dt x Re(c*_j x H_jk / c*_k)
         = - 2 x dt x Re(c_j x H_kj / c_k) = 2 x dt x Re(c_j x [H_jk]* / c_k)

    ==> the probability matrix: P_kj = g_jk = - 2 x dt x Re(c_k x H_jk / c_j)
                                            =   2 x dt x Re(c_k x [H_kj]* / c_j)
    
    H = c_onsite[0]  + c_onsite[1](iontime)      + c_onsite[2](iontime)^2      + c_onsite[3](iontime}^3
      + c_midsite[0] + c_midsite[1](iontime / 2) + c_midsite[2](iontime / 2)^2 + c_midsite[3](iontime / 2)^3
    
    this routine will compute probmat, also V_kj will be update
*/
    const double odt = iontime;       const double mdt = iontime / 2.0;
    const double odt2 = odt * odt;    const double mdt2 = mdt * mdt;
    const double odt3 = odt2 * odt;   const double mdt3 = mdt2 * mdt;

    const int ndim_loc_row = Numroc(nstates, MB_ROW, myprow_group, nprow_group);
    const int ndim_loc_col = Numroc(nstates, NB_COL, mypcol_group, npcol_group);
    const int nvec_loc_col = Numroc(1,       NB_COL, mypcol_group, npcol_group); // 1 or 0
    const size_t mn_loc = (size_t)ndim_loc_row * ndim_loc_col;
    complex<double> *one_over_coeff = new complex<double>[max(ndim_loc_row * nvec_loc_col, 1)]();
    complex<double> *coeff_over_coeff = new complex<double>[max(mn_loc, (size_t)1)]();
    complex<double> *auxL = new complex<double>[max(ndim_loc_row * nvec_loc_col, 1)]();
    complex<double> *auxR = new complex<double>[max(ndim_loc_row * nvec_loc_col, 1)]();
    complex<double> *auxMat = new complex<double>[max(mn_loc, (size_t)1)]();
    if(mypcol_group == 0) {
        #pragma omp parallel for
        for(int ist = 0; ist < ndim_loc_row; ist++) {
            auxL[ist] = 1.0;
            if(abs(coeff[ist]) < 1e-30) {
                one_over_coeff[ist] = 1.0;
                auxR[ist] = 0.0;
            }
            else {
                one_over_coeff[ist] = 1.0 / coeff[ist];
                auxR[ist] = 1.0;
            }
        }
    }
    MPI_Barrier(group_comm);
    Pzgemm("N", "T", nstates, nstates, 1, 1.0, coeff, ndim_loc_row, one_over_coeff, ndim_loc_row,
                                          0.0, coeff_over_coeff, ndim_loc_row);
    Pzgemm("N", "T", nstates, nstates, 1, 1.0, auxL, ndim_loc_row, auxR, ndim_loc_row,
                                          0.0, auxMat, ndim_loc_row);
    int k_glb, j_glb;
    #pragma omp parallel for private(k_glb, j_glb)
    for(size_t kj = 0; kj < mn_loc; kj++) {
        vmat[kj] =  c_onsite[0][kj] + c_onsite[1][kj] * odt 
                                    + c_onsite[2][kj] * odt2 + c_onsite[3][kj] * odt3;
        if(abs(auxMat[kj]) < 0.1) { // in fact, equals to 0.0
            probmat[kj] = abs(coeff_over_coeff[kj]); // coeff_over_coeff[kj] = c_k
            probmat[kj] *= probmat[kj];              // |ck|^2
        }
        else {
            complex<double> Hkj = vmat[kj] + c_midsite[0][kj] + c_midsite[1][kj] * mdt
                                           + c_midsite[2][kj] * mdt2 + c_midsite[3][kj] * mdt3;
            probmat[kj] = 2.0 * iontime * real( coeff_over_coeff[kj] * conj(Hkj) );
            probmat[kj] = max(probmat[kj], 0.0);
        }
        vmat[kj] *= (iu_d * hbar);
        k_glb = BlacsIdxloc2glb(kj % ndim_loc_row, nstates, MB_ROW, myprow_group, nprow_group);
        j_glb = BlacsIdxloc2glb(kj / ndim_loc_row, nstates, NB_COL, mypcol_group, npcol_group);
        if(k_glb == j_glb) probmat[kj] = 0.0;
    }
    delete[] one_over_coeff; delete[] coeff_over_coeff;
    delete[] auxL; delete[] auxR; delete[] auxMat;
    MPI_Barrier(group_comm);
    
    //NormalizeProbability(probmat, nstates, true);
    // no need normalize here because there's a detail balance setting below
    
    MPI_Barrier(group_comm);
    return;
}

void DetailBalanceProb(double *probmat, const complex<double> *vmat,
                       const int nstates, const double temp) {
/* 
   probmat will update in this routine by the eq.(17) of 
   J. Chem. Theory Comput. 2013, 9, 4959−4972
*/
    const int ndim_loc_row = Numroc(nstates, MB_ROW, myprow_group, nprow_group);
    const int ndim_loc_col = Numroc(nstates, NB_COL, mypcol_group, npcol_group);
    const size_t mn_loc = (size_t)ndim_loc_row * ndim_loc_col;
    complex<double> *vmatk = new complex<double>[max(mn_loc, (size_t)1)]();
    complex<double> *vmatj = new complex<double>[max(mn_loc, (size_t)1)]();
    int iprow, jpcol, irow_loc, jcol_loc;
    complex<double> cdtmp;
    for(int ii = 0; ii < nstates; ii++) {
        BlacsIdxglb2loc(ii, iprow, irow_loc, 0, nstates, MB_ROW, nprow_group);
        BlacsIdxglb2loc(ii, jpcol, jcol_loc, 0, nstates, NB_COL, npcol_group);
        
        if(myprow_group == iprow && mypcol_group == jpcol) {
            Zgebs2d(ctxt_group, "ROW", "I", 1, 1, vmat + (irow_loc + jcol_loc * ndim_loc_row), 1);
            for(int j = 0; j < ndim_loc_col; j++)
            vmatk[irow_loc + j * ndim_loc_row] = vmat[irow_loc + jcol_loc * ndim_loc_row];
        }
        else if(myprow_group == iprow) {
            Zgebr2d(ctxt_group, "ROW", "I", 1, 1, &cdtmp, 1, iprow, jpcol);
            for(int j = 0; j < ndim_loc_col; j++) vmatk[irow_loc + j * ndim_loc_row] = cdtmp;
        }
        MPI_Barrier(group_comm);
        
        if(myprow_group == iprow && mypcol_group == jpcol) {
            Zgebs2d(ctxt_group, "COL", "I", 1, 1, vmat + (irow_loc + jcol_loc * ndim_loc_row), 1);
            for(int k = 0; k < ndim_loc_row; k++)
            vmatj[k + jcol_loc * ndim_loc_row] = vmat[irow_loc + jcol_loc * ndim_loc_row];
        }
        else if(mypcol_group == jpcol) {
            Zgebr2d(ctxt_group, "COL", "I", 1, 1, &cdtmp, 1, iprow, jpcol);
            for(int k = 0; k < ndim_loc_row; k++) vmatj[k + jcol_loc * ndim_loc_row] = cdtmp;
        }
        MPI_Barrier(group_comm);
    }
    
    double dEkj;
    #pragma omp parallel for private(dEkj)
    for(size_t kj = 0; kj < mn_loc; kj++) {
        dEkj = real(vmatk[kj]) - real(vmatj[kj]);
        if(dEkj > 0) probmat[kj] *= exp( - dEkj / (kb * temp) );
    }
    delete[] vmatk; delete[] vmatj;
    MPI_Barrier(group_comm);

    NormalizeProbability(probmat, nstates, true);
    /*double *mat_full = NULL;
    if(is_sub_root) mat_full = new double[nstates * nstates];
    MPI_Barrier(group_comm);
    Blacs_MatrixDGather(nstates, nstates, probmat, ndim_loc_row,
                                          mat_full, nstates);
    if(is_sub_root) {
        cout << "probmat norm:" << endl;
        for(int i = 0; i < nstates; i++) {
            for(int j = 0; j < nstates; j++) 
            cout << setiosflags(ios::fixed) << setprecision(10) << mat_full[i + j * nstates] << ' ';
            cout << endl;
        }
        delete[] mat_full;
    }*/

    MPI_Barrier(group_comm);
    return;
}

void PopuUpdateFSSH(const int nstates, const complex<double> *coeff,
                    double *population, // update in this routine
                    complex<double> **c_onsite, complex<double> **c_midsite,
                    double *probmat, complex<double> *vmat, const double temp) {
    FSSHprob(probmat, vmat, nstates, c_onsite, c_midsite, coeff);
    DetailBalanceProb(probmat, vmat, nstates, temp);
    const int ndim_loc_row = Numroc(nstates, MB_ROW, myprow_group, nprow_group);
    double *vectmp = new double[mypcol_group == 0 ? max(ndim_loc_row, 1) : 1];
    if(mypcol_group == 0) {
        Dcopy(ndim_loc_row, population, 1, vectmp, 1);
    }
    MPI_Barrier(group_comm);
    
    Pdgemv("N", nstates, nstates, 1.0, probmat,    ndim_loc_row, 
                                       vectmp,     ndim_loc_row,
                                  0.0, population, ndim_loc_row);
    
    MPI_Barrier(group_comm);
    delete[] vectmp;
    return;
}
