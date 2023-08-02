#include "nac.h"

void Calc_psmApsn(const complex<double> alpha,
                  waveclass &wvcL, waveclass &wvcR,    // wvcL/R: waveclass left/right represents ps_m/ps_n
                  waveclass &wvcCore,                  // wvcCore: for Aij
                  complex<double> *Amn, const int lda, // input and output
                  const int Aij_ID,                    // input, listed in pawpotclass
                  const int sigmaL, const int sigmaR,  // spin channels of wvcL/R
                  const int nstates) {
/*
     < ps_m | ~A | ps_n > = < ps_m | A | ps_n > 
                          + sum_{a,i,j} <ps_m|~p^a_i>(<phi^a_i | A | phi^a_j> - <~phi^a_i | A | ~phi^a_j>)<~p^a_j|ps_n>
     Amn = < ps_m | A | ps_n > is calculated somewhere and will update in this routine
     Aij = <phi^a_i | A | phi^a_j> - <~phi^a_i | A | ~phi^a_j>
*/
    const int nions = wvcL.numatoms;
    int lmmax, lmmax_loc_r, nrow_pp_loc, ncol_pp_loc;
    complex<double> *cdtmp = NULL;
    for(int iatom = 0; iatom < nions; iatom++) {
        lmmax       = wvcL.atoms[iatom].potc->lmmax;
        lmmax_loc_r = wvcL.atoms[iatom].potc->lmmax_loc_r;
        nrow_pp_loc = wvcL.atoms[iatom].nrow_pp_loc;
        ncol_pp_loc = wvcL.atoms[iatom].ncol_pp_loc;
        cdtmp = new complex<double>[max(lmmax_loc_r * ncol_pp_loc, 1)];
        Pzgemm("N", "N", lmmax, nstates, lmmax,
               1.0, wvcCore.atoms[iatom].Aij[Aij_ID], lmmax_loc_r, 
                    wvcR.atoms[iatom].projphi_bc + sigmaR * nrow_pp_loc * ncol_pp_loc, nrow_pp_loc,
               0.0, cdtmp, lmmax_loc_r);
        Pzgemm("C", "N", nstates, nstates, lmmax,
               alpha, wvcL.atoms[iatom].projphi_bc + sigmaL * nrow_pp_loc * ncol_pp_loc, nrow_pp_loc, cdtmp, lmmax_loc_r,
               1.0, Amn, lda); // "1.0" for summation
        delete[] cdtmp;
    }
    return;
}

void Calc_psmApsn(const complex<double> alpha,
                  waveclass &wvcL, waveclass &wvcR,    // wvcL/R: waveclass left/right represents ps_m/ps_n
                  waveclass &wvcCore,                  // wvcCore: for Aij
                  complex<double> *Amn, const int lda, // input and output
                  const int Aij_ID,                    // input, listed in pawpotclass
                  const int sigmaL, const int sigmaR,  // spin channels of wvcL/R
                  const int nst_row, const int i_beg_row, 
                  const int nst_col, const int i_beg_col, const int ntotst) {
// nst: num of states, i_beg: state begin index, ntotst = nkpts x nbnds usually
/*
     < ps_m | ~A | ps_n > = < ps_m | A | ps_n > 
                          + sum_{a,i,j} <ps_m|~p^a_i>(<phi^a_i | A | phi^a_j> - <~phi^a_i | A | ~phi^a_j>)<~p^a_j|ps_n>
     Amn = < ps_m | A | ps_n > is calculated somewhere and will update in this routine
     Aij = <phi^a_i | A | phi^a_j> - <~phi^a_i | A | ~phi^a_j>
     <~p^a_i|ps_m>: [lmmax x nst_row]~[:, i_beg_row:i_beg_row + nst_row]
     <~p^a_j|ps_n>: [lmmax x nst_col]~[:, i_beg_col:i_beg_col + nst_col]
*/
    const int nions = wvcL.numatoms;
    const int nst_col_loc = Numroc(nst_col, NB_COL, mypcol_group, npcol_group);
    int lmmax, lmmax_loc_r, nrow_pp_loc, ncol_pp_loc;
    complex<double> *cdtmp = NULL;
    for(int iatom = 0; iatom < nions; iatom++) {
        lmmax       = wvcL.atoms[iatom].potc->lmmax;
        lmmax_loc_r = wvcL.atoms[iatom].potc->lmmax_loc_r;
        nrow_pp_loc = wvcL.atoms[iatom].nrow_pp_loc;
        ncol_pp_loc = wvcL.atoms[iatom].ncol_pp_loc;
        cdtmp = new complex<double>[max(lmmax_loc_r * nst_col_loc, 1)];
        Pzgemm("N", "N", lmmax, nst_col, lmmax,
               1.0, wvcCore.atoms[iatom].Aij[Aij_ID], lmmax_loc_r,
                    wvcR.atoms[iatom].projphi_bc + sigmaR * nrow_pp_loc * ncol_pp_loc, nrow_pp_loc,
               0.0, cdtmp, lmmax_loc_r,
               lmmax, lmmax, lmmax, ntotst, lmmax, nst_col,
               0, 0, 0, i_beg_col % ntotst, 0, 0);
        Pzgemm("C", "N", nst_row, nst_col, lmmax,
               alpha, wvcL.atoms[iatom].projphi_bc + sigmaL * nrow_pp_loc * ncol_pp_loc, nrow_pp_loc, cdtmp, lmmax_loc_r,
               1.0, Amn, lda, // "1.0" for summation
               lmmax, ntotst, lmmax, nst_col, nst_row, nst_col,
               0, i_beg_row % ntotst, 0, 0, 0, 0);
        delete[] cdtmp;
    }
    return;
}

void PsWaveDot(const complex<double> alpha, waveclass &wvcL, waveclass &wvcR, // calc alpha x <ps_m|ps_n>
               const int nst_row, const int i_beg_row,
               const int nst_col, const int i_beg_col, const int ntotst,
               const int beta, complex<double> *res) { // res[nst_row x nst_col] = alpha x <ps_m|ps_n> + beta x res[m, n]
// nst: num of states, i_beg: state begin index, ntotst = nkpts x nbnds usually
// the planewave coefficients store column by column, total ntotst column pw vectors
    const int npw     = wvcL.eigens[i_beg_row].npw;      // i_beg_row and i_beg_col MUST belong to the same kpoint
    const int npw_loc = wvcL.eigens[i_beg_row].npw_loc;
    const int nst_row_loc_r = Numroc(nst_row, MB_ROW, myprow_group, nprow_group);
    const int nst_row_loc = Numroc(nst_row, NB_COL, mypcol_group, npcol_group);
    const int nst_col_loc = Numroc(nst_col, NB_COL, mypcol_group, npcol_group);
    complex<double> *coeffL = new complex<double>[max(npw_loc * nst_row_loc, 1)]();
    complex<double> *coeffR = new complex<double>[max(npw_loc * nst_col_loc, 1)]();
    for(int isigma = 0; isigma < wvcL.spinor + 1; isigma++) {
        Blacs_MatrixZScatter(npw, nst_row,
                             wvcL.coeff_malloc_in_node[i_beg_row % (wvcL.nkpts * wvcL.nbnds) / wvcL.nbnds]
                             + (size_t)(i_beg_row / (wvcL.nkpts * wvcL.nbnds) * wvcL.nbnds + i_beg_row % wvcL.nbnds)
                             * npw * (1 + wvcL.spinor) + isigma * npw, npw * (1 + wvcL.spinor),
                             coeffL, npw_loc);
        Blacs_MatrixZScatter(npw, nst_col,
                             wvcR.coeff_malloc_in_node[i_beg_col % (wvcR.nkpts * wvcR.nbnds) / wvcL.nbnds]
                             + (size_t)(i_beg_col / (wvcR.nkpts * wvcR.nbnds) * wvcR.nbnds + i_beg_col % wvcR.nbnds)
                             * npw * (1 + wvcR.spinor) + isigma * npw, npw * (1 + wvcR.spinor),
                             coeffR, npw_loc);
        Pzgemm("C", "N", nst_row, nst_col, npw,
               alpha, coeffL, npw_loc, coeffR, npw_loc,
               isigma ? 1.0 : beta, res, nst_row_loc_r);
    }

    delete[] coeffL; delete[] coeffR;
    return;
}

void CalcNAC(double dt, waveclass &wvcC, waveclass &wvcN,
             const int nst_row, const int i_beg_row,
             const int nst_col, const int i_beg_col,
             complex<double> *res, const char *outfilename, const bool is_correction) {
/* non-adiabatic coupling (NAC), calculated by
   
     <psi_i(t)| d/dt |(psi_j(t))> ~=~
                                 (<psi_i(t)|psi_j(t+dt)> - <psi_i(t+dt)|psi_j(t)>) / (2dt)
   wvcC for |psi(t)>, wvcN for |psi(t+dt)>, only states within one kpoint have non-vanished NAC
*/
    const int nspns = wvcC.nspns;
    const int nkpts = wvcC.nkpts;
    const int nbnds = wvcC.nbnds;
    const int isSpinor = wvcC.spinor;
    const int ntotst = nkpts * nbnds;
    const int nst_row_loc_r = Numroc(nst_row, MB_ROW, myprow_group, nprow_group);
    const int nst_col_loc_c = Numroc(nst_col, NB_COL, mypcol_group, npcol_group);
    const int nst2_loc = nst_row_loc_r * nst_col_loc_c;
    
    complex<double> *nacfull = NULL;
    ofstream nacout;
    if(is_sub_root) {
        nacfull = new complex<double>[nst_row * nst_col];
        nacout.open(outfilename, ios::out|ios::binary);
        if(!nacout.is_open())  { cerr << "ERROR: " << outfilename << " can't open" << endl; exit(1); }
    }
    MPI_Barrier(group_comm);
    
    complex<double> phase; // for phase correction
    for(int is = 0; is < nspns; is++) // if isSpinor = true, nspns = 1
    for(int ik = 0; ik < nkpts; ik++) {
        // <psi_i(t)|psi_j(t+dt)> / (2dt)
        // pseudo part
        PsWaveDot(0.5 / dt, wvcC, wvcN, 
                  nst_row, is * nkpts * nbnds + ik * nbnds + i_beg_row,
                  nst_col, is * nkpts * nbnds + ik * nbnds + i_beg_col,
                  ntotst, 0.0, res + (is * nkpts + ik) * nst2_loc);
        // core part
        for(int isigma = 0; isigma < isSpinor + 1; isigma++)
        Calc_psmApsn(0.5 / dt, wvcC, wvcN, wvcC, res + (is * nkpts + ik) * nst2_loc, nst_row_loc_r, 0, // "0" for Qij
                     isSpinor ? isigma : is, isSpinor ? isigma : is,
                     nst_row, ik * nbnds + i_beg_row,
                     nst_col, ik * nbnds + i_beg_col, ntotst);
        
        if(is_correction) { // phase correction
            assert(nst_row == nst_col);
            int nst = nst_row;
            int nst_loc_r = nst_row_loc_r;
            int i_beg = i_beg_row;
            for(int ist = 0; ist < nst; ist++) { // nst_row = nst_col = nst
                int ix_loc, jx_loc, iprow, jpcol;
                BlacsIdxglb2loc(ist, iprow, ix_loc, 0, nst, MB_ROW, nprow_group);
                BlacsIdxglb2loc(ist, jpcol, jx_loc, 0, nst, NB_COL, npcol_group);
                if(myprow_group == iprow && mypcol_group == jpcol) {
                    phase = res[(is * nkpts + ik) * nst2_loc + (ix_loc + jx_loc * nst_loc_r)];
                    phase = exp(- iu_d * arg(phase));
                    Zgebs2d(ctxt_group, "ALL", "I", 1, 1, &phase, 1);
                }
                else Zgebr2d(ctxt_group, "ALL", "I", 1, 1, &phase, 1, iprow, jpcol);
                wvcN.eigens[is * ntotst + ik * nbnds + (i_beg + ist)].extraphase = phase;
                // nac
                Pzscal(nst, phase, res + (is * nkpts + ik) * nst2_loc, nst_loc_r, 1,
                       0, ist, nst, nst);
                for(int isigma = 0; isigma < isSpinor + 1; isigma++) {
                    // pw coefficients
                    if(node_rank == wvcN.malloc_root)
                    Zscal(wvcN.npw[ik], phase,
                          wvcN.eigens[is * ntotst + ik * nbnds + (i_beg + ist)].coeff + isigma * wvcN.npw[ik]);
                    MPI_Barrier(group_comm);
                    // <proj|phi>
                    for(int iatom = 0; iatom < wvcN.numatoms; iatom++) {
                        Pzscal(wvcN.atoms[iatom].potc->lmmax, phase,
                               wvcN.atoms[iatom].projphi_bc + (isSpinor ? isigma : is) * wvcN.atoms[iatom].nrow_pp_loc
                                                                                       * wvcN.atoms[iatom].ncol_pp_loc,
                               wvcN.atoms[iatom].nrow_pp_loc, 1,
                               0, ik * nbnds + (i_beg + ist),
                               wvcN.atoms[iatom].potc->lmmax, ntotst);
                        if(node_rank == wvcN.malloc_root)
                        Zscal( wvcN.atoms[iatom].potc->lmmax, phase,
                               wvcN.atoms[iatom].projphi 
                               + (isSpinor ? isigma : is) * wvcN.atoms[iatom].potc->lmmax * (nkpts * nbnds)
                               + wvcN.atoms[iatom].potc->lmmax * (ik * nbnds + (i_beg + ist)) );
                        MPI_Barrier(group_comm);
                    }
                }
            }
        } // is_correction
        
        // -<psi_i(t+dt)|psi_j(t)> / (2dt)
        // pseudo part
        PsWaveDot(-0.5 / dt, wvcN, wvcC,
                  nst_row, is * nkpts * nbnds + ik * nbnds + i_beg_row,
                  nst_col, is * nkpts * nbnds + ik * nbnds + i_beg_col,
                  ntotst, 1.0, res + (is * nkpts + ik) * nst2_loc);
        // core part
        for(int isigma = 0; isigma < isSpinor + 1; isigma++)
        Calc_psmApsn(-0.5 / dt, wvcN, wvcC, wvcN, res + (is * nkpts + ik) * nst2_loc, nst_row_loc_r, 0, // "0" for Qij
                     isSpinor ? isigma : is, isSpinor ? isigma : is,
                     nst_row, ik * nbnds + i_beg_row,
                     nst_col, ik * nbnds + i_beg_col, ntotst);

        
        // write to tmp
        Blacs_MatrixZGather(nst_row, nst_col, res + (is * nkpts + ik) * nst2_loc, nst_row_loc_r,
                                              nacfull, nst_row);
        if(is_sub_root) {
            nacout.write((char*)nacfull, sizeof(complex<double>) * nst_row * nst_col);
        }
        MPI_Barrier(group_comm);
    }
    
    if(is_sub_root) {
        delete[] nacfull;
        nacout.close();
    }
    MPI_Barrier(group_comm);

    return;
}
