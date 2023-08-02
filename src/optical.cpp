#include "optical.h"

void TDM_PSPart(waveclass &wvc,
                const int cb_start, const int ncb,
                const int vb_start, const int nvb,
                complex<double> *tdm) {
/*
    <ps_{kn}|\grad|ps_{kn'}> = i \sum_{G} (k+G)xC*_{nG}xC_{n'G}

    C_{nG}  is a matrix with dimension of npw x N_v
    C_{n'G} is a matrix with dimension of npw x N_c

        tdm_{x,y,z} = i x C_{nG}^dagger x [(k+G)_{x,y,z} x C_{n'G}]

    where {x,y,z} represent three directions, and three matrix products
    will be calculated in this routine

    the output "tdm" is a collection of (nspns x nkpts x 3) matrixes,
    each is a [nvb x ncb] distributed sub matrix
*/
    complex<double> *leftmat  = NULL;
    complex<double> *cbcoeff  = NULL;
    complex<double> *rightmat = NULL;
    const int nvb_loc_row = Numroc(nvb, MB_ROW, myprow_group, nprow_group);
    const int nvb_loc_col = Numroc(nvb, NB_COL, mypcol_group, npcol_group);
    const int ncb_loc_col = Numroc(ncb, NB_COL, mypcol_group, npcol_group);
    for(int ispn = 0; ispn < wvc.nspns; ispn++)
    for(int ikpt = 0; ikpt < wvc.nkpts; ikpt++) {
        const int npw = wvc.npw[ikpt];
        const int npw_loc = wvc.npw_loc[ikpt];
        leftmat  = new complex<double>[max((size_t)npw_loc * nvb_loc_col, (size_t)1)]();
        cbcoeff  = new complex<double>[max((size_t)npw_loc * ncb_loc_col, (size_t)1)]();
        rightmat = new complex<double>[max((size_t)npw_loc * ncb_loc_col, (size_t)1)]();
        for(int isigma = 0; isigma < wvc.spinor + 1; isigma++) {
            Blacs_MatrixZScatter(npw, nvb, // leftmat = C_{nG}:  npw x nvb
                                 wvc.coeff_malloc_in_node[ikpt]
                                 + (size_t)(ispn * wvc.nbnds + vb_start) * npw * (1 + wvc.spinor) + isigma * npw, npw * (1 + wvc.spinor),
                                 leftmat, npw_loc);
            Blacs_MatrixZScatter(npw, ncb, // cbcoeff = C_{n'G}: npw x ncb
                                 wvc.coeff_malloc_in_node[ikpt]
                                 + (size_t)(ispn * wvc.nbnds + cb_start) * npw * (1 + wvc.spinor) + isigma * npw, npw * (1 + wvc.spinor),
                                 cbcoeff, npw_loc);
            int g[3] = {0, 0, 0};
            double kpg; // k + G
            for(int ii = 0; ii < 3; ii++) {
                Zcopy((size_t)npw_loc * ncb_loc_col, cbcoeff, 1, rightmat, 1);
                for(int ig = 0; ig < npw_loc; ig++) {
                    const int ig_glb = BlacsIdxloc2glb(ig, npw, MB_ROW, myprow_group, nprow_group);
                    IdxNat1toSym3(wvc.gidx[ikpt][ig_glb], g[0], g[1], g[2], wvc.ng[0], wvc.ng[1], wvc.ng[2]);
                    kpg = 2.0 * M_PI * ( (g[0] + wvc.kptvecs[ikpt][0]) * wvc.b[0][ii]
                                       + (g[1] + wvc.kptvecs[ikpt][1]) * wvc.b[1][ii] 
                                       + (g[2] + wvc.kptvecs[ikpt][2]) * wvc.b[2][ii] ); 
                    ZDscal(ncb_loc_col, kpg,        // (k + G)_{ii = [x, y, z]}
                           rightmat + ig, npw_loc); // rightmat = (k + G) x cbcoeff
                }
                MPI_Barrier(group_comm);

                // leftmat^dagger x rightmat
                Pzgemm("C", "N", nvb, ncb, npw,
                       iu_d, leftmat, npw_loc, rightmat, npw_loc,
                       1.0 * isigma, tdm + (ispn * wvc.nkpts + ikpt) * 3 * nvb_loc_row * ncb_loc_col 
                                         + ii * nvb_loc_row * ncb_loc_col, nvb_loc_row);
            } // loop for {x,y,z}
        } // isigma
        delete[] leftmat; 
        delete[] cbcoeff;
        delete[] rightmat; 
    }
    return;
}

void TDM_AEPart(waveclass &wvc,
                const int cb_start, const int ncb,
                const int vb_start, const int nvb,
                complex<double> *tdm) {
/* 
    tdm will update in this routine
    the output "tdm" is a collection of (nspns x nkpts x 3) matrixes,
    each is a [nvb x ncb] distributed sub matrix
    tdm = <vb|\grad|cb>
*/
    const int nvb_loc_row = Numroc(nvb, MB_ROW, myprow_group, nprow_group);
    const int ncb_loc_col = Numroc(ncb, NB_COL, mypcol_group, npcol_group);
    const int ntotst = wvc.nkpts * wvc.nbnds;
    
    for(int ispn = 0; ispn < wvc.nspns; ispn++)
    for(int ikpt = 0; ikpt < wvc.nkpts; ikpt++) {
        for(int ii = 0; ii < 3; ii++) {
            for(int isigma = 0; isigma < wvc.spinor + 1; isigma++) {
                Calc_psmApsn(1.0, wvc, wvc, wvc,
                             tdm + (ispn * wvc.nkpts + ikpt) * 3 * nvb_loc_row * ncb_loc_col 
                                 + ii * nvb_loc_row * ncb_loc_col, nvb_loc_row, 1 + ii, // "1 + ii" for Gij_{x,y,z}
                             wvc.spinor ? isigma : ispn, wvc.spinor ? isigma : ispn,
                             nvb, ikpt * wvc.nbnds + vb_start,
                             ncb, ikpt * wvc.nbnds + cb_start, ntotst);
            }
        }
    }
    
    return;
}

void TDM_over_dE(waveclass &wvc, complex<double> *tdm) {
    const int nvb_loc_row = Numroc(wvc.dimV, MB_ROW, myprow_group, nprow_group);
    const int ncb_loc_col = Numroc(wvc.dimC, NB_COL, mypcol_group, npcol_group);
    const double unit_trans = 2.0 * rytoev; // * autoa / autoa, first autoa for grad, second transfer back to Angstrom
    for(int ispn = 0; ispn < wvc.nspns; ispn++)
    for(int ikpt = 0; ikpt < wvc.nkpts; ikpt++) {
        for(int ii = 0; ii < 3; ii++) {
            for(int ivb = 0; ivb < nvb_loc_row; ivb++)
            for(int jcb = 0; jcb < ncb_loc_col; jcb++)
                tdm[(ispn * wvc.nkpts + ikpt) * 3 * nvb_loc_row * ncb_loc_col 
                                             + ii * nvb_loc_row * ncb_loc_col + (ivb + jcb * nvb_loc_row)] *=
                unit_trans / (
                wvc.eigens[
                IdxNat3toNat1(ispn, ikpt, BlacsIdxloc2glb(ivb, wvc.dimV, MB_ROW, myprow_group, nprow_group) + wvc.dimC,
                              wvc.nspns, wvc.nkpts, wvc.nbnds)].energy - 
                wvc.eigens[
                IdxNat3toNat1(ispn, ikpt, BlacsIdxloc2glb(jcb, wvc.dimC, NB_COL, mypcol_group, npcol_group),
                              wvc.nspns, wvc.nkpts, wvc.nbnds)].energy );
        }
    }

    MPI_Barrier(group_comm);
    return;
}

void CalcCVtdm(waveclass &wvc, const int dirnum,
               complex<double> *cvtdm, complex<double> *cvtdm_full) {
    TDM_PSPart(wvc, 0, wvc.dimC, wvc.dimC, wvc.dimV, cvtdm);
    TDM_AEPart(wvc, 0, wvc.dimC, wvc.dimC, wvc.dimV, cvtdm);
    TDM_over_dE(wvc, cvtdm);
    const int dimV_loc_row = Numroc(wvc.dimV, MB_ROW, myprow_group, nprow_group);
    const int dimC_loc_col = Numroc(wvc.dimC, NB_COL, mypcol_group, npcol_group);

    for(int ispn = 0; ispn < wvc.nspns; ispn++)
    for(int ikpt = 0; ikpt < wvc.nkpts; ikpt++) {
        for(int ii = 0; ii < 3; ii++)
        Blacs_MatrixZGather(wvc.dimV, wvc.dimC,
                            cvtdm + (ispn * wvc.nkpts + ikpt) * 3 * dimV_loc_row * dimC_loc_col
                                  + ii * dimV_loc_row * dimC_loc_col, dimV_loc_row,
                            cvtdm_full + (ii * wvc.nspns * wvc.nkpts + ispn * wvc.nkpts + ikpt)
                                       * wvc.dimV * wvc.dimC, wvc.dimV);
        // because of column-major, slow to fast axis is ic->jv
    }
    if(is_sub_root) {
        ofstream tdmout((runhome + '/' + Int2Str(dirnum) + "/tdmout").c_str(), ios::out);
        if(!tdmout.is_open())  { cerr << "ERROR: " << runhome + '/' + Int2Str(dirnum) + "/tdmout" << " can't open" << endl; exit(1); }
        tdmout << "#" << endl
               << "#                                  <v|\\grad|c>" << endl
               << "#   tdm is defined as <v|r|c> = - -------------   with the unit of Angstrom" << endl
               << "#                                 (Ec - Ev) m_e" << endl
               << "#" << endl
               << "#   Nv = " << wvc.dimV << " , Nc = " << wvc.dimC << endl
               << "#" << endl << endl;
        for(int ispn = 0; ispn < wvc.nspns; ispn++)
        for(int ikpt = 0; ikpt < wvc.nkpts; ikpt++) {
            if(wvc.nspns == 2) tdmout << "# spin = " << ispn << " ;";
            else tdmout << '#';
            tdmout << fixed << setprecision(5) << noshowpos;
            tdmout << " kpoint = " << ikpt + 1 << " (" << wvc.kptvecs[ikpt][0] << ','
                                                       << wvc.kptvecs[ikpt][1] << ','
                                                       << wvc.kptvecs[ikpt][2] << ')' << endl;
            tdmout << scientific << setprecision(12) 
                   << setiosflags(ios::scientific) << setiosflags(ios::uppercase) << showpos;
            for(int ii = 0; ii < 3; ii++) {
                for(int ivb = 0; ivb < wvc.dimV; ivb++) {
                    for(int icb = 0; icb < wvc.dimC; icb++)
                    tdmout << cvtdm_full[(ii * wvc.nspns * wvc.nkpts + ispn * wvc.nkpts + ikpt) * wvc.dimV * wvc.dimC +
                                         ivb + icb * wvc.dimV];
                    tdmout << endl;
                }
                if(ispn < wvc.nspns - 1 || ikpt < wvc.nkpts - 1 || ii < 2) tdmout << endl;
            }
        }
        tdmout.close();
    }
    
    MPI_Barrier(group_comm);
    return;
}
