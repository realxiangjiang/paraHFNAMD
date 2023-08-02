#include "bse.h"

void excitonclass::GetngBSE() {
    #pragma omp parallel for
    for(int i = 0; i < 3; i++) {
        double dbtmp = sqrt(emax / rytoev) / (2.0 * M_PI / (sqrt(VecDot3d(wvc->a[i], wvc->a[i])) / autoa));
        ng[i] = (int)(2 * dbtmp + 3);
    }
    ngtot = ng[0] * ng[1] * ng[2];

    return;
}

bool excitonclass::WithinSphereBSE(int i, int j, int k, double *qpoint) {
    // G vector (i, j, k) is or not in the emax sphere at kpoint
    double tmp, res = 0.0;
    for(int n = 0; n < 3; n++) {
        tmp = wvc->b[0][n] * (i + qpoint[0]) + wvc->b[1][n] * (j + qpoint[1]) + wvc->b[2][n] * (k + qpoint[2]);
        res += tmp * tmp * 4.0 * M_PI * M_PI;
    }

    res = res * rytoev * autoa * autoa;
    if(res < emax) return true;
    else return false;
}

void excitonclass::GetgidxBSE() {
    isGetgidxBSE = true;
    int **gidxTmp = new int*[numQ];
    for(int iQQ = 0; iQQ < numQ; iQQ++) gidxTmp[iQQ] = new int[ngtot];

    int nn;
    for(int iQQ = node_rank; iQQ < numQ; iQQ += node_size) {
        int iQ[3];
        IdxNat1toSym3(iQQ, iQ[2], iQ[1], iQ[0], // slow to fast axis: z->y->x
                           2 * NK_SC[2] - 1, 2 * NK_SC[1] - 1, 2 * NK_SC[0] - 1);
        for(int s = 0; s < 3; s++) qptvecs[iQQ][s] = (double)iQ[s] / NK_SC[s];
        
        nn = 0;
        for(int kk = 0; kk < ng[2]; kk++) {
            int kng = IdxNat1toSym1(kk, ng[2]);
            for(int jj = 0; jj < ng[1]; jj++) {
                int jng = IdxNat1toSym1(jj, ng[1]);
                for(int ii = 0; ii < ng[0]; ii++) {    
                    int ing = IdxNat1toSym1(ii, ng[0]);
                    int ijk = IdxNat3toNat1(ii, jj, kk, ng[0], ng[1], ng[2]);
                    if(WithinSphereBSE(ing, jng, kng, qptvecs[iQQ])) gidxTmp[iQQ][nn++] = ijk;
                } // ii
            } // jj
        } // kk
        npw[iQQ] = nn;
    }
    MPI_Barrier(node_comm);
    
    for(int rt = 0; rt < node_comm; rt++) { // loop for root(rt) to broadcast
        for(int iQQ = rt; iQQ < numQ; iQQ += node_size) {
            MPI_Bcast(npw + iQQ, 1, MPI_INT, rt, node_comm);            // broadcast npw
            MPI_Bcast(qptvecs[iQQ], 3, MPI_DOUBLE, rt, node_comm);     // extra broadcast qptvecs
        }
    }
    MPI_Barrier(node_comm);
    
    const size_t totnpw = accumulate(npw, npw + numQ, 0);
    MpiWindowShareMemoryInitial(totnpw, gidxall, local_gidx_node, window_gidx);
    size_t sumnpw = 0;
    for(int iQQ = 0; iQQ < numQ; iQQ++) {
        npw_loc[iQQ] = Numroc(npw[iQQ], MB_ROW, myprow_group, nprow_group);
        gidx[iQQ] = gidxall + sumnpw;
        sumnpw += npw[iQQ];
    }
    MPI_Barrier(node_comm);

    for(int iQQ = node_rank; iQQ < numQ; iQQ += node_size) {
        copy(gidxTmp[iQQ], gidxTmp[iQQ] + npw[iQQ], gidx[iQQ]);
    }
    MPI_Barrier(node_comm);

    for(int iQQ = 0; iQQ < numQ; iQQ++) delete[] gidxTmp[iQQ];
    delete[] gidxTmp;

    MPI_Barrier(world_comm);
    return;
}

void excitonclass::Getqgabsdir() {
    isGetqgabsdir = true;
    qgabs               = new double*[numQ];
    qgtheta             = new double*[numQ];
    qgphi               = new double*[numQ];
    const size_t totnpw = accumulate(npw, npw + numQ, 0);
    MpiWindowShareMemoryInitial(totnpw, qgabsall,   local_qgabs_node,   window_qgabs);
    MpiWindowShareMemoryInitial(totnpw, qgthetaall, local_qgtheta_node, window_qgtheta);
    MpiWindowShareMemoryInitial(totnpw, qgphiall,   local_qgphi_node,   window_qgphi);
    size_t sumnpw = 0;
    for(int iq = 0; iq < numQ; iq++) {
        qgabs[iq] = qgabsall + sumnpw;
        qgtheta[iq] = qgthetaall + sumnpw;
        qgphi[iq] = qgphiall + sumnpw;
        sumnpw += npw[iq];
    }
    MPI_Barrier(node_comm);

    int gx, gy, gz;
    for(int iq = node_rank; iq < numQ; iq += node_size) {
        #pragma omp parallel for private(gx, gy, gz)
        for(int ig = 0; ig < npw[iq]; ig++) {
            IdxNat1toSym3(gidx[iq][ig], gx, gy, gz, ng[0], ng[1], ng[2]);
            XYZtoRTP<double, double>(2 * M_PI * (gx + qptvecs[iq][0]), 
                                     2 * M_PI * (gy + qptvecs[iq][1]),
                                     2 * M_PI * (gz + qptvecs[iq][2]), wvc->b,
                                     qgabs[iq][ig], qgtheta[iq][ig], qgphi[iq][ig]);
        }
    }
    
    MPI_Barrier(world_comm);
    return;
}

void excitonclass::GetfftIntPre(const int sign) {
    isGetfftIntPre = true;
    fftIntPre            = new complex<double>*[numQ];
    const size_t totnpw = accumulate(npw, npw + numQ, 0);
    MpiWindowShareMemoryInitial(totnpw, fftIntPreall, local_fftIntPre_node, window_fftIntPre);
    size_t sumnpw = 0;
    for(int iQQ = 0; iQQ < numQ; iQQ++) {
        fftIntPre[iQQ] = fftIntPreall + sumnpw;
        sumnpw += npw[iQQ];
    }
    MPI_Barrier(node_comm);

    for(int iQQ = node_rank; iQQ < numQ; iQQ += node_size) {
        #pragma omp parallel for
        for(int igg = 0; igg < npw[iQQ]; igg++) {
            int ig[3] = {0, 0, 0};
            IdxNat1toSym3(gidx[iQQ][igg], ig[0], ig[1], ig[2], ng[0], ng[1], ng[2]); // ng, NOT ngf
            complex<double> tmpcd[3] = {0.0, 0.0, 0.0};
            for(int s = 0; s < 3; s++) tmpcd[s] = (ig[s] ? (exp(2 * M_PI * iu_d * ig[s] / ngf[s] * sign) - 1.0) 
                                                         / (2 * M_PI * iu_d * ig[s] * sign) : 1.0 / ngf[s]); // ngf
            fftIntPre[iQQ][igg] = tmpcd[0] * tmpcd[1] * tmpcd[2];
        }
    }

    MPI_Barrier(world_comm);
    return;
}

void excitonclass::GetfPiOverGabs2(const int nQQ) {
    isGetfPiOverGabs2 = true;
    fPiOverGabs2         = new double*[nQQ];
    const size_t totnpw = accumulate(npw, npw + numQ, 0);
    MpiWindowShareMemoryInitial(totnpw, fPiOverGabs2all, local_fPiOverGabs2, window_fPiOverGabs2, malloc_root);
    size_t sumnpw = 0;
    for(int iQQ = 0; iQQ < nQQ; iQQ++) {
        fPiOverGabs2[iQQ] = fPiOverGabs2all + sumnpw;
        sumnpw += npw[iQQ];
    }
    MPI_Barrier(group_comm);

    int gx, gy, gz;
    double gabs, theta, phi;
    for(int iQQ = node_rank - malloc_root; iQQ < nQQ; iQQ += share_memory_len) {
        #pragma omp parallel for private(gx, gy, gz, gabs, theta, phi)
        for(int ig = 0; ig < npw[iQQ]; ig++) {
            IdxNat1toSym3(gidx[iQQ][ig], gx, gy, gz, ng[0], ng[1], ng[2]);
            XYZtoRTP<double, double>(2 * M_PI * gx, 2 * M_PI * gy, 2 * M_PI * gz, wvc->b,
                                     gabs, theta, phi);
            if(ig == 0) fPiOverGabs2[iQQ][ig] = 0.0;
            else fPiOverGabs2[iQQ][ig] // = 4.0 * M_PI / (gabs * autoa)^2 / (Omega / autoa^3) * 2.0 * rytoev;
                                          = 4.0 * M_PI / (gabs * gabs) * autoa * 2.0 * rytoev;
                                       // unit of eV * Angstrom^3, and divided by Omega will get true result
        }
    }

    MPI_Barrier(world_comm);
    return;
}

void excitonclass::DenMatColumnPseudoPart(const int s1, const int k1, const int n1,
                                          const int s2, const int k2, const int n2,
                                          const int out_col_idx, const int out_tot_cols,  // global
                                          complex<double> *outmat, const int idxQ) { 
/*
    this routine calculate pesudo part of density matrix for one column

    B^{k1,n1}_{k2,n2}(G) = \int dr \psi^*_{k1n1}(r) e^{i(G+k1-k2)r} \psi_{k2n2}(r)
                         
                         = (1 / Omega0) \int_{Omega0} dr u*_{k1n1}(r) e^{iGr} \u_{k2n2}(r)   // PS part
                         + \sum_{a,i,j} <psi_{k1n1}|~p^a_i>
                                      x (<phi^a_i | e^{i(G+k1-k2)r} | phi^a_j> - <~phi^a_i | e^{i(G+k1-k2)r} | ~phi^a_j>)
                                      x <~p^a_j|psi_{k2n2}>                                  // core part

    where \psi_{nk} are the AE wavefunctions, u_{nk} are the periodic part of PS wavefunctions

    Current program can find the reciprocal/real coefficients etc. from correct processes according to "k" and "n".
    The out_col_idx and out_tot_cols determine the memory location of outmat.
*/
    complex<double> *kn1realc = NULL;
    complex<double> *kn2realc = NULL;
    complex<double> *u1u2 = new complex<double>[ngftot](); // u^*_{k1n1} x u_{k2n2}
    const int iQQ = (idxQ < 0 ? MatchKptDiff(k1, k2, NK_SC) : idxQ);
    complex<double> *outtmp = new complex<double>[npw[iQQ]]();

    for(int isigma = 0; isigma < (1 + spinor); isigma++) {
        //*** PS part

        // first, prepare kn1realc(u_{k1n1})/kn2realc(u_{k2n2})
        kn1realc = wvc->eigens[s1 * NKSCtot * NBtot + k1 * NBtot + n1].realc + isigma * ngftot;
        kn2realc = wvc->eigens[s2 * NKSCtot * NBtot + k2 * NBtot + n2].realc + isigma * ngftot;
        
        // second, calculate u^*_{k1n1} x u_{k2n2}
        #pragma omp parallel for
        for(int ir = 0; ir < ngftot; ir++) u1u2[ir] = conj(kn1realc[ir]) * kn2realc[ir];

        // third, FFT to reciprocal space and compact
        Zfft_3d(u1u2, ngf, FFTW_BACKWARD);
        Cubiod2Compacted(ngf, u1u2, npw[iQQ], gidx[iQQ], ng, outtmp);
        
        // finally, multiply an prefix factor
        if(isigma == 0) {
            #pragma omp parallel for
            for(int ig = 0; ig < npw[iQQ]; ig++) {
                outmat[(size_t)out_col_idx * npw[iQQ] + ig]  = fftIntPre[iQQ][ig] * outtmp[ig];
            }
        }
        else {
            #pragma omp parallel for
            for(int ig = 0; ig < npw[iQQ]; ig++) {
                outmat[(size_t)out_col_idx * npw[iQQ] + ig] += fftIntPre[iQQ][ig] * outtmp[ig];
            }
        }
    } // isigma
    delete[] u1u2; delete[] outtmp;

    return;
}

void excitonclass::DenMatCPLeftMatBlockCyclic(const int iqpt, complex<double> *leftmat) {
/*
    Left: (<phi^a_i | e^{i(q+G).r} | phi^a_j> - <~phi^a_i | e^{i(q+G).r} | ~phi^a_j>)
          = e^{i(q+G).R_a} x Ekrij            with dimension [npw(q) x totnvij[-1]]
*/
    const int totnvij_loc_col = Numroc(totnvij[wvc->numatoms], NB_COL, mypcol_group, npcol_group);
    for(int jcol = 0; jcol < totnvij_loc_col; jcol++) {
        int jcol_glb = BlacsIdxloc2glb(jcol, totnvij[wvc->numatoms], NB_COL, mypcol_group, npcol_group);
        int iatom = 0; for(iatom = 0; iatom < wvc->numatoms; iatom++) 
        if(totnvij[iatom] <= jcol_glb && jcol_glb < totnvij[iatom + 1]) break;
        if(sexpikr) wvc->atoms[iatom].potc->ReadEkrij(iqpt, npw);
        int jcol_in_atom = jcol_glb - totnvij[iatom];
        #pragma omp parallel for
        for(int ig = 0; ig < npw_loc[iqpt]; ig++) {
            int ig_glb = BlacsIdxloc2glb(ig, npw[iqpt], MB_ROW, myprow_group, nprow_group);
            leftmat[ig + jcol * npw_loc[iqpt]]
            = wvc->atoms[iatom].potc->Ekrij[iqpt][ig_glb + jcol_in_atom * npw[iqpt]]
            * wvc->atoms[iatom].crexp_q[iqpt][ig_glb];
        }
        if(sexpikr) wvc->atoms[iatom].potc->DelEkrij(iqpt);
    }
    
    MPI_Barrier(group_comm);
    return;
}

void excitonclass::DenMatCPLeftMatBlockCyclicOneAtom(const int iqpt, const int iatom, complex<double> *leftmat) {
/*
    Left: (<phi^a_i | e^{i(q+G).r} | phi^a_j> - <~phi^a_i | e^{i(q+G).r} | ~phi^a_j>)
          = e^{i(q+G).R_a} x Ekrij            with dimension [npw(q) x tot_nv_ij(a)]
*/
    const int totij_loc_col = Numroc(wvc->atoms[iatom].potc->tot_nv_ij, NB_COL, mypcol_group, npcol_group);
    if(sexpikr) wvc->atoms[iatom].potc->ReadEkrij(iqpt, npw);
    for(int jcol = 0; jcol < totij_loc_col; jcol++) {
        int jcol_glb = BlacsIdxloc2glb(jcol, wvc->atoms[iatom].potc->tot_nv_ij, NB_COL, mypcol_group, npcol_group);
        #pragma omp parallel for
        for(int ig = 0; ig < npw_loc[iqpt]; ig++) {
            int ig_glb = BlacsIdxloc2glb(ig, npw[iqpt], MB_ROW, myprow_group, nprow_group);
            leftmat[ig + jcol * npw_loc[iqpt]]
            = wvc->atoms[iatom].potc->Ekrij[iqpt][ig_glb + jcol_glb * npw[iqpt]]
            * wvc->atoms[iatom].crexp_q[iqpt][ig_glb];
        }
    }
    if(sexpikr) wvc->atoms[iatom].potc->DelEkrij(iqpt);
    
    MPI_Barrier(group_comm);
    return;
}

void excitonclass::DenMatCPLeftMatAtomAll(const int iqpt, complex<double> *leftmat) {
/*
    Left: (<phi^a_i | e^{i(q+G).r} | phi^a_j> - <~phi^a_i | e^{i(q+G).r} | ~phi^a_j>)
          = e^{i(q+G).R_a} x Ekrij            with dimension [npw(q) x totnvij[-1]]
*/
    for(int iatom = 0; iatom < wvc->numatoms; iatom++) {
        if(sexpikr) wvc->atoms[iatom].potc->ReadEkrij(iqpt, npw);
        for(int jcol = totnvij[iatom]; jcol < totnvij[iatom + 1]; jcol++) {
            int jcol_in_atom = jcol - totnvij[iatom];
            for(int ig = 0; ig < npw[iqpt]; ig++) {
                leftmat[ig + jcol * npw[iqpt]] = wvc->atoms[iatom].potc->Ekrij[iqpt][ig + jcol_in_atom * npw[iqpt]]
                                               * wvc->atoms[iatom].crexp_q[iqpt][ig];
            }
        }
        if(sexpikr) wvc->atoms[iatom].potc->DelEkrij(iqpt);
    }

    MPI_Barrier(group_comm);
    return;
}

void excitonclass::DenMatCPLeftMatAtomEach(const int iqpt, const int iatom, complex<double> *leftmat) {
/*
    Left: (<phi^a_i | e^{i(q+G).r} | phi^a_j> - <~phi^a_i | e^{i(q+G).r} | ~phi^a_j>)
          = e^{i(q+G).R_a} x Ekrij            with dimension [npw(q) x tot_nv_ij(a)]
*/
    if(sexpikr) wvc->atoms[iatom].potc->ReadEkrij(iqpt, npw);
    for(int jcol = 0; jcol < wvc->atoms[iatom].potc->tot_nv_ij; jcol++) {
        for(int ig = 0; ig < npw[iqpt]; ig++) {
            leftmat[ig + jcol * npw[iqpt]] = wvc->atoms[iatom].potc->Ekrij[iqpt][ig + jcol * npw[iqpt]]
                                           * wvc->atoms[iatom].crexp_q[iqpt][ig];
        }
    }
    if(sexpikr) wvc->atoms[iatom].potc->DelEkrij(iqpt);

    return;
}

void excitonclass::DenMatCorePartOneKptPairAtomAll(const int spn, const int k1, const int k2,
                                                   const int bnd_start, const int nbnds,
                                                   complex<double> *denmat) {
/*
    Left: (<phi^a_i | e^{i(q+G).r} | phi^a_j> - <~phi^a_i | e^{i(q+G).r} | ~phi^a_j>)
          = e^{i(q+G).R_a} x Ekrij                with dimension [npw(k1-k2) x totnvij[-1]]
    Right: <psi_{k1n1}|~p^a_i><~p^a_j|psi_{k2n2}> with dimension [totnvij[-1] x N^2]
    res = Left x Right with q = k1 - k2
*/
    const int iqpt = MatchKptDiff(k1, k2, NK_SC);
    //double tstart, tend;

    // Left:
    //tstart = omp_get_wtime();
    complex<double> *leftmat  = new complex<double>[ (size_t)npw[iqpt] * totnvij[wvc->numatoms] ]();
    DenMatCPLeftMatAtomAll(iqpt, leftmat);
    /*tend = omp_get_wtime();
    cout << "LeftMat:  " << k1 << '/' << NKSCtot << ' ' << k2 << '/' << NKSCtot << ": "
         << tend - tstart << " s" << endl;*/

    // Right:
    //tstart = omp_get_wtime();
    complex<double> *rightmat = new complex<double>[(size_t)totnvij[wvc->numatoms] * (nbnds * nbnds)]();
    int n1, n2;
    for(int iatom = 0; iatom < wvc->numatoms; iatom++)
    for(int irow = totnvij[iatom]; irow < totnvij[iatom + 1]; irow++) {
        int irow_in_atom = irow - totnvij[iatom];
        int iijj = (wvc->atoms[iatom].potc->all_nv_ij)[irow_in_atom];
        int ii = iijj / wvc->atoms[iatom].potc->lmmax;
        int jj = iijj % wvc->atoms[iatom].potc->lmmax;
        #pragma omp parallel for private(n1, n2)
        for(int n1n2 = 0; n1n2 < nbnds * nbnds; n1n2++) {
            n1 = n1n2 / nbnds; n2 = n1n2 % nbnds;
            rightmat[irow + n1n2 * totnvij[wvc->numatoms]] =
            conj((wvc->atoms[iatom].projphi + spn * wvc->atoms[iatom].potc->lmmax * (NKSCtot * NBtot)) // <psi_{k1n1}|~p^a_i>
                 [ii + (k1 * NBtot + bnd_start + n1) * wvc->atoms[iatom].potc->lmmax]) * 
                 (wvc->atoms[iatom].projphi + spn * wvc->atoms[iatom].potc->lmmax * (NKSCtot * NBtot)) // <~p^a_j|psi_{k2n2}>
                 [jj + (k2 * NBtot + bnd_start + n2) * wvc->atoms[iatom].potc->lmmax];
            if(spinor)
            rightmat[irow + n1n2 * totnvij[wvc->numatoms]] +=
            conj((wvc->atoms[iatom].projphi + wvc->atoms[iatom].potc->lmmax * (NKSCtot * NBtot)) // <psi_{k1n1}|~p^a_i>
                 [ii + (k1 * NBtot + bnd_start + n1) * wvc->atoms[iatom].potc->lmmax]) * 
                 (wvc->atoms[iatom].projphi + wvc->atoms[iatom].potc->lmmax * (NKSCtot * NBtot)) // <~p^a_j|psi_{k2n2}>
                 [jj + (k2 * NBtot + bnd_start + n2) * wvc->atoms[iatom].potc->lmmax];
        }
    }
    /*tend = omp_get_wtime();
    cout << "RightMat: " << k1 << '/' << NKSCtot << ' ' << k2 << '/' << NKSCtot << ": "
         << tend - tstart << " s" << endl;*/
    
    // denmat += Left x Right
    //tstart = omp_get_wtime();
    Zgemm("CblasColMajor", "CblasNoTrans", "CblasNoTrans",
          npw[iqpt], nbnds * nbnds, totnvij[wvc->numatoms],
          1.0, leftmat, npw[iqpt], rightmat, totnvij[wvc->numatoms],
          1.0, denmat, npw[iqpt]);
    /*tend = omp_get_wtime();
    cout << "L x R:    " << k1 << '/' << NKSCtot << ' ' << k2 << '/' << NKSCtot << ": "
         << tend - tstart << " s" << endl;*/

    delete[] leftmat; delete[] rightmat;

    return;
}

void excitonclass::DenMatCorePartOneKptPairAtomEach(const int spn, const int k1, const int k2,
                                                    const int bnd_start, const int nbnds,
                                                    complex<double> *denmat) {
/*
    Left: (<phi^a_i | e^{i(q+G).r} | phi^a_j> - <~phi^a_i | e^{i(q+G).r} | ~phi^a_j>)
          = e^{i(q+G).R_a} x Ekrij                with dimension [npw(k1-k2) x totnvij(a)]
    Right: <psi_{k1n1}|~p^a_i><~p^a_j|psi_{k2n2}> with dimension [totnvij(a) x N^2]
    res = sum_a Left x Right with q = k1 - k2
*/
    const int iqpt = MatchKptDiff(k1, k2, NK_SC);

    complex<double> *leftmat  = NULL;
    complex<double> *rightmat = NULL;
    int lmmax, tot_nv_ij;
    int n1, n2;
    for(int iatom = 0; iatom < wvc->numatoms; iatom++) {
        lmmax = wvc->atoms[iatom].potc->lmmax;
        tot_nv_ij = wvc->atoms[iatom].potc->tot_nv_ij;
        // Left:
        leftmat  = new complex<double>[(size_t)npw[iqpt] * tot_nv_ij]();
        DenMatCPLeftMatAtomEach(iqpt, iatom, leftmat);

        // Right:
        //tstart = omp_get_wtime();
        rightmat = new complex<double>[tot_nv_ij * (nbnds * nbnds)]();
        for(int irow = 0; irow < tot_nv_ij; irow++) {
            int iijj = (wvc->atoms[iatom].potc->all_nv_ij)[irow];
            int ii = iijj / lmmax;
            int jj = iijj % lmmax;
            #pragma omp parallel for private(n1, n2)
            for(int n1n2 = 0; n1n2 < nbnds * nbnds; n1n2++) {
                n1 = n1n2 / nbnds; n2 = n1n2 % nbnds;
                rightmat[irow + n1n2 * tot_nv_ij] =
                conj((wvc->atoms[iatom].projphi + spn * lmmax * (NKSCtot * NBtot)) // <psi_{k1n1}|~p^a_i>
                     [ii + (k1 * NBtot + bnd_start + n1) * lmmax]) * 
                     (wvc->atoms[iatom].projphi + spn * lmmax * (NKSCtot * NBtot)) // <~p^a_j|psi_{k2n2}>
                     [jj + (k2 * NBtot + bnd_start + n2) * lmmax];
                if(spinor)
                rightmat[irow + n1n2 * tot_nv_ij] +=
                conj((wvc->atoms[iatom].projphi + lmmax * (NKSCtot * NBtot)) // <psi_{k1n1}|~p^a_i>
                     [ii + (k1 * NBtot + bnd_start + n1) * lmmax]) * 
                     (wvc->atoms[iatom].projphi + lmmax * (NKSCtot * NBtot)) // <~p^a_j|psi_{k2n2}>
                     [jj + (k2 * NBtot + bnd_start + n2) * lmmax];
            }
        }
        // denmat += Left x Right
        Zgemm("CblasColMajor", "CblasNoTrans", "CblasNoTrans",
              npw[iqpt], nbnds * nbnds, tot_nv_ij,
              1.0, leftmat, npw[iqpt], rightmat, tot_nv_ij,
              1.0, denmat, npw[iqpt]);

        delete[] leftmat; delete[] rightmat;
    }

    return;
}

void excitonclass::DenMatCorePartKCV(const int spn, complex<double> *denmat) {
/*
    Left: (<phi^a_i | e^{i(q+G).r} | phi^a_j> - <~phi^a_i | e^{i(q+G).r} | ~phi^a_j>)
          = e^{i(q+G).R_a} x Ekrij            with dimension [npw x totnvij[-1]]
    Right: <psi_{kc}|~p^a_i><~p^a_j|psi_{kv}> with dimension [totnvij[-1] x NkNcNv]
    res = Left x Right, here k1 = k2 => q = 0
*/
    
    // Left:
    const int totnvij_loc_col = Numroc(totnvij[wvc->numatoms], NB_COL, mypcol_group, npcol_group);
    complex<double> *leftmat  = new complex<double>[max(npw_loc[0] * totnvij_loc_col, 1)]();
    DenMatCPLeftMatBlockCyclic(0, leftmat);
    
    // Right:
    const int totnvij_loc_row = Numroc(totnvij[wvc->numatoms], MB_ROW, myprow_group, nprow_group);
    complex<double> *rightmat = new complex<double>[max(totnvij_loc_row * dim_loc_col, 1)]();
    int kcv_glb, kk, cc, vv;
    int lmmax;
    for(int irow = 0; irow < totnvij_loc_row; irow++) {
        int irow_glb = BlacsIdxloc2glb(irow, totnvij[wvc->numatoms], MB_ROW, myprow_group, nprow_group);
        int iatom = 0; for(iatom = 0; iatom < wvc->numatoms; iatom++) 
        if(totnvij[iatom] <= irow_glb && irow_glb < totnvij[iatom + 1]) break;
        int lmmax = wvc->atoms[iatom].potc->lmmax;
        int iijj = (wvc->atoms[iatom].potc->all_nv_ij)[ irow_glb - totnvij[iatom] ];
        int ii = iijj / lmmax;
        int jj = iijj % lmmax;
        #pragma omp parallel for private(kcv_glb, kk, cc, vv)
        for(int kcv = 0; kcv < dim_loc_col; kcv++) {
            kcv_glb = BlacsIdxloc2glb(kcv, dim, NB_COL, mypcol_group, npcol_group);
            IdxNat1toNat3(kcv_glb, kk, cc, vv, NKSCtot, dimC, dimV);
            rightmat[irow + kcv * totnvij_loc_row] =
            conj((wvc->atoms[iatom].projphi + spn * lmmax * (NKSCtot * NBtot)) // <psi_{kc}|~p^a_i>
                 [ii + (kk * NBtot + cc) * lmmax]) * 
                 (wvc->atoms[iatom].projphi + spn * lmmax * (NKSCtot * NBtot)) // <~p^a_j|psi_{kv}>
                 [jj + (kk * NBtot + (vv + dimC)) * lmmax];
            if(spinor)
            rightmat[irow + kcv * totnvij_loc_row] +=
            conj((wvc->atoms[iatom].projphi + lmmax * (NKSCtot * NBtot)) // <psi_{kc}|~p^a_i>
                 [ii + (kk * NBtot + cc) * lmmax]) * 
                 (wvc->atoms[iatom].projphi + lmmax * (NKSCtot * NBtot)) // <~p^a_j|psi_{kv}>
                 [jj + (kk * NBtot + (vv + dimC)) * lmmax];
        }
    }
    MPI_Barrier(group_comm);

    // denmat += Left x Right
    Pzgemm("N", "N", npw[0], dim, totnvij[wvc->numatoms],
           1.0, leftmat, npw_loc[0], rightmat, totnvij_loc_row,
           1.0, denmat, npw_loc[0]);

    MPI_Barrier(group_comm);
    delete[] leftmat; delete[] rightmat;
    
    MPI_Barrier(group_comm);
    return;
}

void excitonclass::DenMatCorePartKCVOneAtom(const int spn, complex<double> *denmat) {
/*
    Left: (<phi^a_i | e^{i(q+G).r} | phi^a_j> - <~phi^a_i | e^{i(q+G).r} | ~phi^a_j>)
          = e^{i(q+G).R_a} x Ekrij            with dimension [npw x tot_nv_ij(a)]
    Right: <psi_{kc}|~p^a_i><~p^a_j|psi_{kv}> with dimension [tot_nv_ij(a) x NkNcNv]
    res = Left x Right, here k1 = k2 => q = 0
*/
    complex<double> *leftmat = NULL;
    complex<double> *rightmat = NULL;
    int lmmax, tot_nv_ij;
    int kcv_glb, kk, cc, vv;
    for(int iatom = 0; iatom < wvc->numatoms; iatom++) {
        lmmax = wvc->atoms[iatom].potc->lmmax;
        tot_nv_ij = wvc->atoms[iatom].potc->tot_nv_ij;
        const int totij_loc_row = Numroc(tot_nv_ij, MB_ROW, myprow_group, nprow_group);
        // Left:
        leftmat = new complex<double>[max(npw_loc[0] * totij_loc_row, 1)]();
        DenMatCPLeftMatBlockCyclicOneAtom(0, iatom, leftmat);
        
        // Right:
        rightmat = new complex<double>[max(totij_loc_row * dim_loc_col, 1)]();
        int irow_glb;
        for(int irow = 0; irow < totij_loc_row; irow++) {
            int irow_glb = BlacsIdxloc2glb(irow, tot_nv_ij, MB_ROW, myprow_group, nprow_group);
            int ii = (wvc->atoms[iatom].potc->all_nv_ij)[irow_glb] / lmmax;
            int jj = (wvc->atoms[iatom].potc->all_nv_ij)[irow_glb] % lmmax;
            #pragma omp parallel for private(kcv_glb, kk, cc, vv)
            for(int kcv = 0; kcv < dim_loc_col; kcv++) {
                kcv_glb = BlacsIdxloc2glb(kcv, dim, NB_COL, mypcol_group, npcol_group);
                IdxNat1toNat3(kcv_glb, kk, cc, vv, NKSCtot, dimC, dimV);
                rightmat[irow + kcv * totij_loc_row] =
                conj((wvc->atoms[iatom].projphi + spn * lmmax * (NKSCtot * NBtot)) // <psi_{kc}|~p^a_i>
                     [ii + (kk * NBtot + cc) * lmmax]) * 
                     (wvc->atoms[iatom].projphi + spn * lmmax * (NKSCtot * NBtot)) // <~p^a_j|psi_{kv}>
                     [jj + (kk * NBtot + (vv + dimC)) * lmmax];
                if(spinor)
                rightmat[irow + kcv * totij_loc_row] +=
                conj((wvc->atoms[iatom].projphi + lmmax * (NKSCtot * NBtot)) // <psi_{kc}|~p^a_i>
                     [ii + (kk * NBtot + cc) * lmmax]) * 
                     (wvc->atoms[iatom].projphi + lmmax * (NKSCtot * NBtot)) // <~p^a_j|psi_{kv}>
                     [jj + (kk * NBtot + (vv + dimC)) * lmmax];
            }
        }
        MPI_Barrier(group_comm);

        // denmat += Left x Right:
        Pzgemm("N", "N", npw[0], dim, tot_nv_ij,
               1.0, leftmat, npw_loc[0], rightmat, totij_loc_row,
               1.0, denmat, npw_loc[0]);
        
        MPI_Barrier(group_comm);
        delete[] leftmat; delete[] rightmat;
    }

    MPI_Barrier(group_comm);
    return;
}
    
void excitonclass::DensityMatrixOneKptPair(const int spn, const int k1, const int k2, 
                                           const int bnd_start, const int nbnds,
                                           complex<double> *denmat) {
//calculate the matrix B^k1_k2[npw x nbnds^2], nbnds:bnd_start ~ bnd_start + nbnds
    /*double tstart, tend;
    tstart = omp_get_wtime();*/
    int ib1, ib2;
    for(int ibnd1 = 0; ibnd1 < nbnds; ibnd1++)
    for(int ibnd2 = 0; ibnd2 < nbnds; ibnd2++) {
        ib1 = bnd_start + ibnd1; ib2 = bnd_start + ibnd2;
        DenMatColumnPseudoPart(spn, k1, ib1, spn, k2, ib2,
                               ibnd1 * nbnds + ibnd2, nbnds * nbnds, denmat);
    }
    /*tend = omp_get_wtime();
    cout << "PSpart:   " << k1 << '/' << NKSCtot << ' ' << k2 << '/' << NKSCtot << ": "
         << tend - tstart << " s" << endl;*/

    //tstart = omp_get_wtime();
    //DenMatCorePartOneKptPairAtomAll(spn, k1, k2, bnd_start, nbnds, denmat);
    DenMatCorePartOneKptPairAtomEach(spn, k1, k2, bnd_start, nbnds, denmat);  // this one is memory friendly keeping similar speed
    /*tend = omp_get_wtime();
    cout << "AEpart:   " << k1 << '/' << NKSCtot << ' ' << k2 << '/' << NKSCtot << ": "
         << tend - tstart << " s" << endl;*/
    
    return;
}

void excitonclass::DensityMatrixKCV(const int spn, complex<double> *denmat) {
    int ikpt, icb, ivb;
    complex<double> *kcvdenmat;
    MPI_Win window_kcvdenmat; complex<double> *local_kcvdenmat;
    MpiWindowShareMemoryInitial((size_t)npw[0] * dim,
                                kcvdenmat, local_kcvdenmat, window_kcvdenmat, malloc_root);
    int kcv_glb;
    if(myprow_group == 0) // only col_root do below loop to avoid repeating calculation
    for(int kcv = 0; kcv < dim_loc_col; kcv++) {
        kcv_glb = BlacsIdxloc2glb(kcv, dim, NB_COL, mypcol_group, npcol_group);
        IdxNat1toNat3(kcv_glb, ikpt, icb, ivb, NKSCtot, dimC, dimV);
        DenMatColumnPseudoPart(spn, ikpt, icb, spn, ikpt, ivb + dimC,
                               kcv_glb, dim, kcvdenmat);
    }
    MPI_Barrier(group_comm);
    
    // scatter to block-cyclic
    for(int kcv = 0; kcv < dim_loc_col; kcv++) {
        kcv_glb = BlacsIdxloc2glb(kcv, dim, NB_COL, mypcol_group, npcol_group);
        Blacs_MatrixZScatter(npw[0], 1,
                             kcvdenmat + (size_t)kcv_glb * npw[0], npw[0],
                             denmat + (size_t)kcv * npw_loc[0], npw_loc[0],
                             npw[0], 1, 0, 0, 0, 0,
                             onecol_root_prow, onecol_root_pcol, ctxt_only_onecol_root,
                             myprow_onecol, mypcol_onecol, nprow_onecol, npcol_onecol, ctxt_onecol);
    }
    MPI_Barrier(group_comm);
    MPI_Win_free(&window_kcvdenmat);
    MPI_Barrier(group_comm);
    
    DenMatCorePartKCV(spn, denmat);  
  //DenMatCorePartKCVOneAtom() may slower

    MPI_Barrier(group_comm);
    return;
}

void excitonclass::DirectTerms(const int spnL, const int spnR,
                               const complex<double> alpha, const complex<double> beta,
                               complex<double> *scTerms) {
/*
    this routine calculates
        
        scTerms = alpha x sum_{GG'} B^{kc}_{k'c'}(G) x W_{GG'} x B*^{kv}_{k'v'}(G')
                + beta x scTerms
    
    usually alpha = -1.0 / ncells, beta = 1.0 or 0.0
    if iswfull = true:
        Left:   B^{kc}_{k'c'}(G)  with dimension [npw x dimCC]
        Middle: W_{GG'}           with dimension [npw x npw]
        Right:  B^{kv}_{k'v'}(G') with dimension [npw x dimVV]
        the result should be trans(Left) x Middle x conj(Right)
        for q = 0, W_{GG'} includes head and wing
    if iswfull = false:
        Left:  W_{GG} x B^{kc}_{k'c'}(G) with dimension [npw x dimCC]
        Right: B^{kv}_{k'v'}(G')         with dimension [npw x dimVV]
        the result should be trans(Left) x conj(Right)
        for q = 0, W_{GG'} includes head but should count wing alone
*/
    double tstart, tend;
    //for(int ik12 = 0; ik12 < NKSCtot * NKSCtot; ik12++) {
    for(int ik12 = sub_rank; ik12 < NKSCtot * NKSCtot; ik12 += sub_size) {
        tstart = omp_get_wtime();
        const int ik1 = ik12 / NKSCtot;
        const int ik2 = ik12 % NKSCtot;
        const int iQQ = MatchKptDiff(ik1, ik2, NK_SC);
        ccDenMat = new complex<double>[(size_t)npw[iQQ] * dimCC]();
        vvDenMat = new complex<double>[(size_t)npw[iQQ] * dimVV]();
        //tstart = omp_get_wtime();
        DensityMatrixOneKptPair(spnL, ik1, ik2, 0,    dimC, ccDenMat);
        /*tend = omp_get_wtime();
        cout << "1 Pair1: " << ik12 << '/' << NKSCtot * NKSCtot << ' '
             << "k1, k2 = " << ik1 << '/' << NKSCtot << ' ' << ik2 << '/' << NKSCtot << ": "
             << tend - tstart << " s" << endl;*/
        //tstart = omp_get_wtime();
        DensityMatrixOneKptPair(spnR, ik1, ik2, dimC, dimV, vvDenMat);
        /*tend = omp_get_wtime();
        cout << "2 Pair2: " << ik12 << '/' << NKSCtot * NKSCtot << ' '
             << "k1, k2 = " << ik1 << '/' << NKSCtot << ' ' << ik2 << '/' << NKSCtot << ": "
             << tend - tstart << " s" << endl;*/
        #pragma omp parallel for
        for(size_t ii = 0; ii < (size_t)npw[iQQ] * dimVV; ii++) vvDenMat[ii] = conj(vvDenMat[ii]);
        
        if(iswfull) {
            if(false) { // old version, read the entire wfullggp
            //tstart = omp_get_wtime();
            wpc->ReadOneWFull(iQQ);
            /*tend = omp_get_wtime();
            cout << "3 read one wfull: k1, k2 = " << ik1 << '/' << NKSCtot << ' ' << ik2 << '/' << NKSCtot << ": "
                 << tend - tstart << " s" << endl;*/
            complex<double> *cdtmp = new complex<double>[(size_t)npw[iQQ] * dimVV];
            Zgemm("CblasColMajor", "CblasNoTrans", "CblasNoTrans", // cdtmp = Middle x conf(Right)
                  npw[iQQ], dimVV, npw[iQQ],
                  1.0, wpc->Qcs[iQQ].wfullggp, npw[iQQ], vvDenMat, npw[iQQ],
                  0.0, cdtmp, npw[iQQ]);
            Zgemm("CblasColMajor", "CblasTrans", "CblasNoTrans",  // trans(Left) x cdtmp
                  dimCC, dimVV, npw[iQQ],
                  alpha, ccDenMat, npw[iQQ], cdtmp, npw[iQQ],
                  beta, scTerms + ik12 * (dimCC * dimVV), dimCC);
            delete[] cdtmp;
            wpc->DelOneWFull(iQQ);
            /*tend = omp_get_wtime();
            cout << "3 wfull: k1, k2 = " << ik1 << '/' << NKSCtot << ' ' << ik2 << '/' << NKSCtot << ": "
                 << tend - tstart << " s" << endl;*/
            } // old version

            complex<double> *cdtmp = new complex<double>[(size_t)npw[iQQ] * dimVV]();
            int idxbeg, numq_of_ibrk;
            for(int ikpt = 0; ikpt < wpc->nkibrgw; ikpt++) {
                numq_of_ibrk = wpc->Qcs[iQQ].numq_of_ibrk[ikpt];
                if(numq_of_ibrk == 0) continue;
                wpc->ReadOneWFullSub(iQQ, ikpt, idxbeg);
                Zgemm("CblasColMajor", "CblasNoTrans", "CblasNoTrans", // cdtmp = Middle x conf(Right)
                      numq_of_ibrk, dimVV, numq_of_ibrk,
                      1.0, wpc->Qcs[iQQ].wfullggp, numq_of_ibrk, vvDenMat + idxbeg, npw[iQQ],
                      0.0, cdtmp + idxbeg, npw[iQQ]);
                wpc->DelOneWFull(iQQ);
            }
            if(iQQ == 0) {
                // wing
                for(int irow = 1; irow < npw[0]; irow++) // exclude head
                // cdtmp += wing x conf(Right), row by row
                Zaxpy(dimVV, (double)ncells * wpc->Qcs[0].wing[irow], vvDenMat, cdtmp + irow, npw[0], npw[0]);

                // cwing, include head
                Zgemm("CblasColMajor", "CblasNoTrans", "CblasNoTrans", // cdtmp += cwing x conf(Right)
                      1, dimVV, npw[0],
                      (double)ncells, wpc->Qcs[0].cwing, 1, vvDenMat, npw[0], // must multiply "ncells" here
                      1.0, cdtmp, npw[0]);
            }
            Zgemm("CblasColMajor", "CblasTrans", "CblasNoTrans",  // trans(Left) x cdtmp
                  dimCC, dimVV, npw[iQQ],
                  alpha, ccDenMat, npw[iQQ], cdtmp, npw[iQQ],
                  beta, scTerms + ik12 * (dimCC * dimVV), dimCC);
            delete[] cdtmp;
        }
        else {
            //tstart = omp_get_wtime();
            #pragma omp parallel for
            for(int ig = 0; ig < npw[iQQ]; ig++) 
            Zscal(dimCC, 
                  alpha * wpc->Qcs[iQQ].wgg[ig] /*+  MAY UPDATE FUTURE
                  // ik1 = ik2: alpha x W_GG - WING_{G,0} - CWING_{0,G}
                  (ik1 == ik2 ? wpc->Qcs[0].wing[ig] - wpc->Qcs[0].cwing[ig]: 0.0)*/,
                  ccDenMat + ig, npw[iQQ]);
            Zgemm("CblasColMajor", "CblasTrans", "CblasNoTrans",
                  dimCC, dimVV, npw[iQQ],
                  1.0, ccDenMat, npw[iQQ], vvDenMat, npw[iQQ],
                  beta, scTerms + ik12 * (dimCC * dimVV), dimCC);
            /*tend = omp_get_wtime();
            cout << "3 wdiag: " << ik12 << '/' << NKSCtot * NKSCtot << ' '
                 << "k1, k2 = " << ik1 << '/' << NKSCtot << ' ' << ik2 << '/' << NKSCtot << ": "
                 << tend - tstart << " s" << endl;*/
        }
        delete[] ccDenMat; delete[] vvDenMat;
        /*if( ik1 == 2 && ik2 == 2 ) {
            cout << "(ik1,ik2) = (" << ik1 << ',' << ik2 << "): " << flush;
            for(int icv = 0; icv < dimCC * dimVV; icv++) cout<< (scTerms + ik12 * (dimCC * dimVV))[icv];
            cout << endl;
        }*/
        tend = omp_get_wtime();
        if(ik2 == NKSCtot - 1)
        if(taskmod == "bse" && totstru < 10) {
            cout << "Direct Term  " << ik12 << '/' << NKSCtot * NKSCtot << "  "
                 << "k1, k2 = " << ik1 << '/' << NKSCtot << ' ' << ik2 << '/' << NKSCtot << ": "
                 << setiosflags(ios::fixed) << setprecision(1) << tend - tstart << " s" << endl;
            cout.copyfmt(iosDefaultState);
        }
    } // ik1, ik2
    MPI_Barrier(group_comm);
    return;
}

void excitonclass::ExchangeTerms(const int spnL, const int spnR,
                                 const complex<double> alpha, const complex<double> beta,
                                 complex<double> *exTerms) {
/*
    this routine calculates
        
        exTerms = alpha x sum_{G} B^{kc}_{kv}(G) x (4pi / |G|^2) x B*^{k'c'}_{k'v'}(G)
                + beta x exTerms
    
    usually alpha = +1.0 / Omega, beta = 1.0 or 0.0
    
    Left:  ( 4pi / |G|^2 ) x B^{kc}_{kv}(G) with dimension [npw x dim]
    Right: B^{k'c'}_{k'v'}(G) with dimension [npw x dim]
    the result should be trans(Left) x conj(Right)
*/
    cvDenMatLeft  = new complex<double>[(size_t)npw_loc[0] * dim_loc_col]();
    cvDenMatRight = new complex<double>[(size_t)npw_loc[0] * dim_loc_col]();
    DensityMatrixKCV(spnL, cvDenMatLeft);
    int irow_glb;
    #pragma omp parallel for private(irow_glb)
    for(int irow = 0; irow < npw_loc[0]; irow++) {
        irow_glb = BlacsIdxloc2glb(irow, npw[0], MB_ROW, myprow_group, nprow_group);
        ZDscal(dim_loc_col, fPiOverGabs2[0][irow_glb], cvDenMatLeft + irow, npw_loc[0]);
    }

    DensityMatrixKCV(spnR, cvDenMatRight);
    #pragma omp parallel for
    for(size_t ii = 0; ii < (size_t)npw_loc[0] * dim_loc_col; ii++) cvDenMatRight[ii] = conj(cvDenMatRight[ii]);
    
    MPI_Barrier(group_comm);
    Pzgemm("T", "N", dim, dim, npw[0],
           alpha, cvDenMatLeft, npw_loc[0], cvDenMatRight, npw_loc[0],
           beta, exTerms, dim_loc_row);
    delete[] cvDenMatLeft; delete[] cvDenMatRight;
    return;
}

void excitonclass::DirectTermsNoGW_KbyK(const double epsl, const int spnL, const int spnR,
                                        const complex<double> alpha, const complex<double> beta,
                                        complex<double> *scTerms) {
/*
    see details in DirectTerms, but with a const W_{GG'} = (4pi / |G|^2 / epsl) delta_{GG'}
    usually alpha = -1.0 / Omega, beta = 1.0 or 0.0
*/
    // temporarily modify
    if(node_rank == malloc_root)
    fPiOverGabs2[0][0] = 2.0 * pow(6.0 / (M_PI * omega), 1.0 / 3.0) // 2[6/(pi x omega)]^(1/3)
                       * autoa * 2.0 * rytoev * omega;              // Hartree to eV, times "omega" to cancel alpha
    MPI_Barrier(group_comm);

    for(int ik12 = sub_rank; ik12 < NKSCtot * NKSCtot; ik12 += sub_size) {
        const int ik1 = ik12 / NKSCtot;
        const int ik2 = ik12 % NKSCtot;
        const int iQQ = MatchKptDiff(ik1, ik2, NK_SC);
        ccDenMat = new complex<double>[(size_t)npw[iQQ] * dimCC];
        vvDenMat = new complex<double>[(size_t)npw[iQQ] * dimVV];
        
        DensityMatrixOneKptPair(spnL, ik1, ik2, 0,    dimC, ccDenMat);
        #pragma omp parallel for
        for(int irow = 0; irow < npw[iQQ]; irow++) 
        ZDscal(dimCC, fPiOverGabs2[iQQ][irow], ccDenMat + irow, npw[iQQ]);
        
        DensityMatrixOneKptPair(spnR, ik1, ik2, dimC, dimV, vvDenMat);
        #pragma omp parallel for
        for(long ii = 0; ii < (long)npw[iQQ] * dimVV; ii++) vvDenMat[ii] = conj(vvDenMat[ii]);
        
        MPI_Barrier(group_comm);
        
        Zgemm("CblasColMajor", "CblasTrans", "CblasNoTrans",
              dimCC, dimVV, npw[iQQ],
              alpha / epsl, ccDenMat, npw[iQQ], vvDenMat, npw[iQQ],
              beta, scTerms + ik12 * (dimCC * dimVV), dimCC);
        
        delete[] ccDenMat; delete[] vvDenMat;
    } // ik1, ik2
        
    // modify back
    if(node_rank == malloc_root)
    fPiOverGabs2[0][0] = 0.0;
    MPI_Barrier(group_comm);
    return;
}

void excitonclass::DiagonalSetting(const int nspns, const double gapadd,
                                   complex<double> *mat) { // mat[(nspns x dim) x (nspns x dim)]
    const int ndim_loc_row = Numroc(nspns * dim, MB_ROW, myprow_group, nprow_group);
    int ii_loc_row, ii_loc_col;
    int iprow, jpcol;
    int ss, kk, cb, vb;
    double *energydiff = new double[nspns * dim];
    #pragma omp parallel for private(ii_loc_row, ii_loc_col, iprow, jpcol, ss, kk, cb, vb)
    for(int ii = 0; ii < nspns * dim; ii++) {
        BlacsIdxglb2loc(ii, iprow, ii_loc_row, 0, nspns * dim, MB_ROW, nprow_group);
        BlacsIdxglb2loc(ii, jpcol, ii_loc_col, 0, nspns * dim, NB_COL, npcol_group);
        ss = ii / dim;
        IdxNat1toNat3(ii % dim, kk, cb, vb, NKSCtot, dimC, dimV);
        if(myprow_group == iprow && mypcol_group == jpcol) {
            mat[ii_loc_row + ii_loc_col * ndim_loc_row]
            = wvc->eigens[min(totdiffspns - 1, ss) * NKSCtot * NBtot + kk * NBtot +        cb].energy
            - wvc->eigens[min(totdiffspns - 1, ss) * NKSCtot * NBtot + kk * NBtot + dimC + vb].energy + gapadd;
        }
        if(is_sub_root) {
            energydiff[ii]
            = wvc->eigens[min(totdiffspns - 1, ss) * NKSCtot * NBtot + kk * NBtot +        cb].energy
            - wvc->eigens[min(totdiffspns - 1, ss) * NKSCtot * NBtot + kk * NBtot + dimC + vb].energy + gapadd;
        }
    }
    MPI_Barrier(group_comm);
    
    if(is_sub_root) {
        ofstream diagout((namddir + "/tmpDiagonal/" + Int2Str(dirnum)).c_str(), ios::out|ios::binary);
        if(!diagout.is_open())  { cerr << "ERROR: " << namddir + "/tmpDiagonal/" + Int2Str(dirnum) << " can't open" << endl; exit(1); }
        diagout.write((char*)energydiff, sizeof(double) * nsdim);
        diagout.close();
    }
    delete[] energydiff;
    MPI_Barrier(group_comm);

    return;
}

void excitonclass::ScTerms2BSEMat(const int nspns, const int spn1, const int spn2, // spn1/2 for row/column
                                  const complex<double> *scTerms,
                                  complex<double> *rootmat, complex<double> *bcmat) {
/*
    scTerms has the shape of nkpts^2 x [dimCC x dimVV]
    rootmat is malloc in group_root process with shape of [(nspns x dim) x (nspns x dim)]
    bcmat is the Block Cyclic mat of rootmat with shape [nsdim_loc_row x nsdim_loc_col]
    this routine scatter the date from scTerms to rootmat(rewrite) and bcmat(update)
*/
    const int nsdim_loc_r = Numroc(nspns * dim, MB_ROW, myprow_group, nprow_group);
    int kcv1, kcv2, skcv1_loc, skcv2_loc, recv_prow, recv_pcol, recv_rank;
    int k1, c1, v1, k2, c2, v2, cc, vv, send_rank;
    //#pragma omp parallel for private(kcv1, kcv2, skcv1_loc, skcv2_loc, recv_prow, recv_pcol, k1, c1, v1, k2, c2, v2, cc_loc, vv_loc, send_prow, send_pcol) /* openmp has bug here */
    for(int kcv = 0; kcv < dim * dim; kcv++) {
        kcv1 = kcv % dim; // row
        kcv2 = kcv / dim; // column
        BlacsIdxglb2loc(spn1 * dim + kcv1, recv_prow, skcv1_loc, 0, nspns * dim, MB_ROW, nprow_group);
        BlacsIdxglb2loc(spn2 * dim + kcv2, recv_pcol, skcv2_loc, 0, nspns * dim, NB_COL, npcol_group);
        recv_rank = recv_prow * npcol_group + recv_pcol;
        IdxNat1toNat3(kcv1, k1, c1, v1, NKSCtot, dimC, dimV);
        IdxNat1toNat3(kcv2, k2, c2, v2, NKSCtot, dimC, dimV);
        send_rank = (k1 * NKSCtot + k2) % sub_size;
        cc = c1 * dimC + c2;
        vv = v1 * dimV + v2;

        // first part for bcmat
        if( send_rank == recv_rank || // in same rank
           (all_mal_nodeclr[send_rank] == all_mal_nodeclr[recv_rank] &&
            all_malloc_root[send_rank] == all_malloc_root[recv_rank]) ) { // no need communication thanks to MPI_Win
            if(sub_rank == recv_rank)
            bcmat[skcv1_loc + skcv2_loc * nsdim_loc_r] += 
            scTerms[(k1 * NKSCtot + k2) * (dimCC * dimVV) + (cc + vv * dimCC)];
        }
        else {
            // need communication
            if(sub_rank == send_rank)
            MPI_Send(scTerms + (k1 * NKSCtot + k2) * (dimCC * dimVV) + (cc + vv * dimCC),
                     1, MPI_CXX_DOUBLE_COMPLEX, recv_rank, 0, group_comm);
            else if(sub_rank == recv_rank) {
                MPI_Status mpi_status;
                complex<double> cdtmp;
                MPI_Recv(&cdtmp, 1, MPI_CXX_DOUBLE_COMPLEX, send_rank, 0, group_comm, &mpi_status);
                bcmat[skcv1_loc + skcv2_loc * nsdim_loc_r] += cdtmp;
            }
        }

        MPI_Barrier(group_comm);
        
        // second part for rootmat
        if(send_rank == sub_root || // send process is the group root
           (all_mal_nodeclr[send_rank] == all_mal_nodeclr[sub_root] &&
            all_malloc_root[send_rank] == all_malloc_root[sub_root]) ) { // no need communication thanks to MPI_Win
            if(sub_rank == sub_root) 
            rootmat[(spn1 * dim + kcv1) + (spn2 * dim + kcv2) * (nspns * dim)] =
            scTerms[(k1 * NKSCtot + k2) * (dimCC * dimVV) + (cc + vv * dimCC)];
        }
        else { // need communication
            if(sub_rank == send_rank)
            MPI_Send(scTerms + (k1 * NKSCtot + k2) * (dimCC * dimVV) + (cc + vv * dimCC),
                     1, MPI_CXX_DOUBLE_COMPLEX, sub_root, 1, group_comm);
            else if(sub_rank == sub_root) {
                MPI_Status mpi_status;
                MPI_Recv(rootmat + (spn1 * dim + kcv1) + (spn2 * dim + kcv2) * (nspns * dim),
                         1, MPI_CXX_DOUBLE_COMPLEX, send_rank, 1, group_comm, &mpi_status);
            }
        }
        
        MPI_Barrier(group_comm);
    }
    MPI_Barrier(group_comm);
    
    if(is_sub_root) { // write rootmat
        ofstream rtmatout((namddir + "/tmpDirect/" + Int2Str(dirnum)).c_str(), ios::out|ios::binary);
        if(!rtmatout.is_open())  { cerr << "ERROR: " << namddir + "/tmpDirect/" + Int2Str(dirnum) << " can't open" << endl; exit(1); }
        rtmatout.write((char*)rootmat, sizeof(complex<double>) * nsdim * nsdim);
        rtmatout.close();
    }
    
    MPI_Barrier(group_comm);
    return;
}

void excitonclass::ExTerms2BSEMat(const int nspns, const int spn1, const int spn2,
                                  const complex<double> *exTerms,
                                  complex<double> *rootmat, complex<double> *bcmat) {
/*
    exTerms has the shape of [dim_loc_row x dim_loc_col]
    rootmat is malloc in group_root process with shape of [(nspns x dim) x (nspns x dim)]
    bcmat is the Block Cyclic mat of rootmat with shape [nsdim_loc_row x nsdim_loc_col]
    if nspns = 1, do intra-process copy from exTerms to bcmat and broadcast to rootmat
    if nspns = 2, do inter-process copy to bcmat and broadcast to rootmat
*/
    const int nsdim_loc_r = Numroc(nspns * dim, MB_ROW, myprow_group, nprow_group);
    Pzgeadd("N", dim, dim, (double)kappa, exTerms, dim_loc_row, 1.0, bcmat, nsdim_loc_r,
            dim, dim, nspns * dim, nspns * dim, 0, 0, spn1 * dim, spn2 * dim);
    Blacs_MatrixZGather(dim, dim, exTerms, dim_loc_row, 
                                  rootmat + (spn1 * dim + (spn2 * dim) * (nspns * dim)), nspns * dim);
    
    if(is_sub_root) { // write rootmat
        ofstream rtmatout((namddir + "/tmpExchange/" + Int2Str(dirnum)).c_str(), ios::out|ios::binary);
        if(!rtmatout.is_open())  { cerr << "ERROR: " << namddir + "/tmpExchange/" + Int2Str(dirnum) << " can't open" << endl; exit(1); }
        rtmatout.write((char*)rootmat, sizeof(complex<double>) * nsdim * nsdim);
        rtmatout.close();
    }
    
    return;
}

void excitonclass::ExcitonTDM(const complex<double> *eigenvecs,
                              complex<double> *Xtdm_full) {
    const int dimV_loc_row = Numroc(dimV, MB_ROW, myprow_group, nprow_group);
    const int dimC_loc_col = Numroc(dimC, NB_COL, mypcol_group, npcol_group);
    complex<double> *cvtdm = new complex<double>[max(3 * numspns * NKSCtot * dimV_loc_row * dimC_loc_col, 1)]();
    complex<double> *cvtdm_full = NULL;
    if(is_sub_root) cvtdm_full = new complex<double>[3 * numspns * NKSCtot * dimC * dimV]();
    MPI_Barrier(group_comm);
    CalcCVtdm(*wvc, dirnum, cvtdm, cvtdm_full);
    // in the sub_root process, cvtdm_full can be regard as a column vector
    // with dimension of 3 x nspns x nkpts x dimC x dimV
    
    complex<double> *skcvtdm = NULL;
    complex<double> *Xtdm = NULL;
    if(mypcol_group == 0) {
        skcvtdm = new complex<double>[max(nsdim_loc_row, 1)]();
        Xtdm = new complex<double>[max(nsdim_loc_row, 1)]();
    }
    else {
        skcvtdm = new complex<double>[1]();
        Xtdm = new complex<double>[1]();
    }
    MPI_Barrier(group_comm);
    for(int ii = 0; ii < 3; ii++) {
        Blacs_MatrixZScatter(nsdim, 1, cvtdm_full + ii * nsdim, nsdim, skcvtdm, nsdim_loc_row);

        Pzgemv("T", nsdim, nsdim,
               1.0, eigenvecs, nsdim_loc_row, skcvtdm, nsdim_loc_row,
               0.0, Xtdm, nsdim_loc_row);

        Blacs_MatrixZGather(nsdim, 1, Xtdm, nsdim_loc_row, Xtdm_full + ii * nsdim, nsdim);
        MPI_Barrier(group_comm);
    }

    delete[] skcvtdm; delete[] Xtdm;
    delete[] cvtdm; if(is_sub_root) delete [] cvtdm_full;
    MPI_Barrier(group_comm);
    return;
}

void excitonclass::XEnergies(const complex<double> *bcmat) {
    double *eigenvals = new double[nsdim]();
    complex<double> *eigenvecs = new complex<double>[max(nsdim_loc_row * nsdim_loc_col, 1)]();
    Pzheev("V", nsdim, bcmat, nsdim_loc_row, eigenvals, eigenvecs, nsdim_loc_row);
    complex<double> *root_eigenvecs = NULL;
    complex<double> *Xtdm_full = NULL;
    if(is_sub_root) {
        root_eigenvecs = new complex<double>[nsdim * nsdim]();
        Xtdm_full = new complex<double>[3 * nsdim]();
    }
    MPI_Barrier(group_comm);
    ExcitonTDM(eigenvecs, Xtdm_full);
    Blacs_MatrixZGather(nsdim, nsdim, eigenvecs, nsdim_loc_row, root_eigenvecs, nsdim);
    if(is_sub_root) {
        const int ww = (int)log10(nsdim) + 1;
        ofstream bseout((runhome + '/' + Int2Str(dirnum) + "/bseout").c_str(), ios::out);
        ofstream bsevec((runhome + '/' + Int2Str(dirnum) + "/bsevec").c_str(), ios::out|ios::binary);
        if(!bseout.is_open())  { cerr << "ERROR: " << runhome + '/' + Int2Str(dirnum) + "/bseout" << " can't open" << endl; exit(1); }
        if(!bsevec.is_open())  { cerr << "ERROR: " << runhome + '/' + Int2Str(dirnum) + "/bsevec" << " can't open" << endl; exit(1); }
        bsevec.write((char*)root_eigenvecs, sizeof(complex<double>) * nsdim * nsdim);
        double *one_vec_abs2 = new double[nsdim];
        int spn, kpt, cb, vb;
        for(int ix = 0; ix < nsdim; ix++) {
            #pragma omp parallel for
            for(int icft = 0; icft < nsdim; icft++) one_vec_abs2[icft] = pow(abs(root_eigenvecs[icft + ix * nsdim]), 2);
            vector<int> argsort_onevec = Argsort<double>(one_vec_abs2, nsdim);
            bseout << " No. = " << std::right << setw(ww) << setfill(' ') << ix + 1
                   << " ,  energy = " << std::right << setw(9) << setfill(' ') << fixed << setprecision(4) << eigenvals[ix];
            bseout << " ,  TDM = " << scientific
                   << setprecision(12) << setiosflags(ios::scientific) << setiosflags(ios::uppercase) << showpos
                   << Xtdm_full[ix] << ' ' << Xtdm_full[ix + nsdim] << ' ' << Xtdm_full[ix + 2 * nsdim] << endl;
            bseout << noshowpos;
            if(numspns == 1) {
                bseout << "#        kpt        cb        vb        population" << endl;
                for(int icft = 0; icft < min(50, nsdim); icft++) {
                    IdxNat1toNat3(argsort_onevec[nsdim - 1 - icft], kpt, cb, vb, NKSCtot, dimC, dimV);
                    bseout << std::left << setw(9)  << setfill(' ') << icft + 1
                           << std::left << setw(11) << setfill(' ') << kpt + 1
                           << std::left << setw(10) << setfill(' ') << wvc->bands[0][cb] + 1
                           << std::left << setw(10) << setfill(' ') << wvc->bands[0][vb + dimC] + 1
                           << fixed << setprecision(5) << one_vec_abs2[ argsort_onevec[nsdim - 1 - icft] ] << endl;
                }
            }
            else if(numspns == 2) {
                bseout << "#        spin        kpt        cb        cb        population" << endl;
                for(int icft = 0; icft < min(50, nsdim); icft++) {
                    spn = argsort_onevec[nsdim - 1 - icft] / dim;
                    IdxNat1toNat3(argsort_onevec[nsdim - 1 - icft] % dim, kpt, cb, vb, NKSCtot, dimC, dimV);
                    bseout << std::left << setw(9)  << setfill(' ') << icft + 1
                           << std::left << setw(12) << setfill(' ') << spn
                           << std::left << setw(11) << setfill(' ') << kpt + 1
                           << std::left << setw(10) << setfill(' ') << wvc->bands[spn][cb] + 1
                           << std::left << setw(10) << setfill(' ') << wvc->bands[spn][vb + dimC] + 1
                           << fixed << setprecision(5) << one_vec_abs2[ argsort_onevec[nsdim - 1 - icft] ] << endl;
                }
            }
            if(ix < nsdim - 1) bseout << endl; // add an empty line
        }
        bseout.close(); bsevec.close();
        delete[] root_eigenvecs; delete[] one_vec_abs2;
        delete[] Xtdm_full;
    } MPI_Barrier(group_comm);
    delete[] eigenvals; delete[] eigenvecs;

    MPI_Barrier(group_comm);
    return;
}

void excitonclass::ExcitonMatrix(const int num) {
/*
    there are two cases:
    1) numspns = 1
    mat should be dim x dim: D + K^d + kappa x K^x, kappa = 0 or 2
    in which D is diagonal with QP energies, K^d/K^x is the direct/exchange terms
    
    2) numspns = 2
    mat should be (2dim) x (2dim):
    / D_uu + K^d_uu + K^x_uu     K^x_ud         \
    |                                           |
    \        K^x_du       D_dd + K^d_dd + K^x_dd/
    u=up, d=down
*/
    for(int iatom = 0; iatom < wvc->numatoms; iatom++) {
        wvc->atoms[iatom].Getcrexp_q(numQ, qptvecs, npw, gidx, ng);
    } // refresh crexp_q

    dirnum = num;
    complex<double> *bcmat   = new complex<double>[max(nsdim_loc_row * nsdim_loc_col, 1)]();
    complex<double> *rootmat = NULL;
    if(is_sub_root) rootmat = new complex<double>[nsdim * nsdim]();
    complex<double> *exTerms = new complex<double>[max(dim_loc_row * dim_loc_col, 1)]();
    complex<double> *scTerms = NULL;
    MPI_Win window_scTerms; complex<double> *local_scTerms = NULL;
    MpiWindowShareMemoryInitial(NKSCtot2 * dimCC * dimVV,
                                scTerms, local_scTerms, window_scTerms, malloc_root);
    ofstream rtmatout;
    
    DiagonalSetting(numspns, gapdiff, bcmat);  // D
    //double tstart, tend; tstart = omp_get_wtime();
    for(int spn1 = 0; spn1 < numspns; spn1++) for(int spn2 = 0; spn2 < numspns; spn2++) { // D + K^d
        if(totdiffspns == 1) {
            if(spn1 == 0 && spn2 == 0) {
                if(epsilon < 0) DirectTerms(0, 0, - 1.0 / (double)ncells, 0.0, scTerms);
                else            DirectTermsNoGW_KbyK(epsilon, 0, 0, - 1.0 / omega, 0.0, scTerms);
            }
        }
        else {
            if(epsilon < 0) DirectTerms(spn1, spn2, - 1.0 / (double)ncells, 0.0, scTerms);
            else            DirectTermsNoGW_KbyK(epsilon, spn1, spn2, - 1.0 / omega, 0.0, scTerms);
        }
        ScTerms2BSEMat(numspns, spn1, spn2, scTerms, rootmat, bcmat);
    }
    /*tend = omp_get_wtime(); Cout << "1 DirectTerms: " << tend - tstart << " s" << endl;
    MPI_Barrier(group_comm); tstart = omp_get_wtime();*/

    for(int spn1 = 0; spn1 < numspns; spn1++) for(int spn2 = 0; spn2 < numspns; spn2++) { // D + K^d + K^x
        if(totdiffspns == 1) {
            if(spn1 == 0 && spn2 == 0) ExchangeTerms(0, 0, 1.0 / omega, 0.0, exTerms);
        }
        else ExchangeTerms(spn1, spn2, 1.0 / omega, 0.0, exTerms);
        ExTerms2BSEMat(numspns, spn1, spn2, exTerms, rootmat, bcmat);
    }
    //tend = omp_get_wtime(); Cout << "2 ExchangeTerms: " << tend - tstart << " s" << endl; MPI_Barrier(group_comm);
    
    XEnergies(bcmat);
    
    delete[] bcmat;
    if(is_sub_root) delete[] rootmat;
    delete[] exTerms;
    MPI_Win_free(&window_scTerms);
    
    return;
}

excitonclass::excitonclass(waveclass *wvc_in, wpotclass *wpc_in):wvc(wvc_in), wpc(wpc_in) {
}
void excitonclass::Initial() {
    isInitial = true;
    spinor = wvc->spinor;
    kappa = 2; if(numspns == 2) kappa = 1;
    for(int s = 0; s < 3; s++) { NK_SC[s] = nkptssc[s]; ngf[s] = wvc->ngf[s]; }
    ngftot = wvc->ngftot;
    numQ = (2 * NK_SC[0] - 1) * (2 * NK_SC[1] - 1) * (2 * NK_SC[2] - 1);
    NKSCtot = NK_SC[0] * NK_SC[1] * NK_SC[2];
    NKSCtot2 = NKSCtot * NKSCtot;
    dimC = wvc->dimC; dimV = wvc->dimV;
    
    dim = NKSCtot * dimC * dimV;
    dim_loc_row = Numroc(dim, MB_ROW, myprow_group, nprow_group);
    dim_loc_col = Numroc(dim, NB_COL, mypcol_group, npcol_group);
    nsdim = numspns * dim;
    nsdim_loc_row = Numroc(nsdim, MB_ROW, myprow_group, nprow_group);
    nsdim_loc_col = Numroc(nsdim, NB_COL, mypcol_group, npcol_group);
    dimCC = dimC * dimC;
    dimCC_loc_row = Numroc(dimCC, MB_ROW, myprow_group, nprow_group);
    dimCC_loc_col = Numroc(dimCC, NB_COL, mypcol_group, npcol_group);
    dimVV = dimV * dimV;
    dimVV_loc_row = Numroc(dimVV, MB_ROW, myprow_group, nprow_group);
    dimVV_loc_col = Numroc(dimVV, NB_COL, mypcol_group, npcol_group);
    NBtot = dimC + dimV;
    dimKKCC = NKSCtot2 * dimC * dimC;
    dimKKCC_loc_row = Numroc(dimKKCC, MB_ROW, myprow_group, nprow_group);
    dimKKCC_loc_col = Numroc(dimKKCC, NB_COL, mypcol_group, npcol_group);
    dimKKVV = NKSCtot2 * dimV * dimV;
    dimKKVV_loc_row = Numroc(dimKKVV, MB_ROW, myprow_group, nprow_group);
    dimKKVV_loc_col = Numroc(dimKKVV, NB_COL, mypcol_group, npcol_group);
    malloc_root      = wvc->malloc_root;
    malloc_end       = wvc->malloc_end;
    share_memory_len = wvc->share_memory_len;
    all_malloc_root  = wvc->all_malloc_root;
    all_mal_nodeclr  = wvc->all_mal_nodeclr;
    
    ngf_loc_n0 = wvc->ngf_loc_n0;
    ngf_loc_0_start = wvc->ngf_loc_0_start;
    all_ngf_loc_n0 = wvc->all_ngf_loc_n0;
    all_ngf_loc_0_start = wvc->all_ngf_loc_0_start;
    tot_loc_ngf = wvc->tot_loc_ngf;
    ngftot_loc = ngf_loc_n0 * ngf[1] * ngf[2];
    gidx = new int*[numQ];
    npw = new int[numQ];
    npw_loc = new int[numQ];
    qptvecs = new double*[numQ];
    for(int iQQ = 0; iQQ < numQ; iQQ++) qptvecs[iQQ] = new double[3];
    if(epsilon < 0) { // GW
        emax = wpc->emax_GW;
        for(int iQQ = 0; iQQ < numQ; iQQ++) {
            npw[iQQ] = wpc->Qcs[iQQ].npwSC;
            npw_loc[iQQ] = wpc->Qcs[iQQ].npwSC_loc;
            gidx[iQQ] = wpc->Qcs[iQQ].Gind;
            for(int s = 0; s < 3; s++) qptvecs[iQQ][s] = wpc->Qcs[iQQ].Qptvec[s];
        }
        for(int s = 0; s < 3; s++) ng[s] = wpc->NG_SCGW[s];
        ngtot = ng[0] * ng[1] * ng[2];
        omega0 = wpc->omega0_GW; ncells = wpc->ncells;
        omega = omega0 * ncells;
    }
    else {
        emax = wvc->emax * 2.0 / 3.0;
        GetngBSE();
        GetgidxBSE();
        omega = wvc->volume * NKSCtot;
    }
    Getqgabsdir();
    totnvij = new int[wvc->numatoms + 1];
    totlmmax = new int[wvc->numatoms + 1];
    totnvij[0] = 0; totlmmax[0] = 0;
    for(int iatom = 0; iatom < wvc->numatoms; iatom++) {
        wvc->atoms[iatom].potc->CalcEkrij(numQ, npw, npw_loc, qgabs, qgtheta, qgphi);
    }
    for(int iatom = 1; iatom <= wvc->numatoms; iatom++) {
        totnvij[iatom] = totnvij[iatom - 1] + wvc->atoms[iatom - 1].potc->tot_nv_ij;
        totlmmax[iatom] = totlmmax[iatom - 1] + wvc->atoms[iatom - 1].potc->lmmax;
    }
    GetfftIntPre(1);
    GetfPiOverGabs2(epsilon < 0 ? 1 : numQ);
    return;
}

excitonclass::~excitonclass() {
    if(isInitial) {
        if(epsilon < 0) {} // GW
        else {
            if(isGetgidxBSE) {
                MPI_Win_free(&window_gidx);
            }
        }
        for(int iQQ = 0; iQQ < numQ; iQQ++) delete[] qptvecs[iQQ];
        delete[] qptvecs;
        delete[] gidx;
        delete[] npw; delete[] npw_loc;
        delete[] totnvij;
        delete[] totlmmax;
    }
    if(isGetfftIntPre) {
        MPI_Win_free(&window_fftIntPre);
        delete[] fftIntPre;
    }
    if(isGetfPiOverGabs2) {
        MPI_Win_free(&window_fPiOverGabs2);
        delete[] fPiOverGabs2;
    }
    if(isGetqgabsdir) {
        MPI_Win_free(&window_qgabs);
        MPI_Win_free(&window_qgtheta);
        MPI_Win_free(&window_qgphi);
        delete[] qgabs; delete[] qgtheta; delete[] qgphi;
    }
}
