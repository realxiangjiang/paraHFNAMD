#include "dsh.h"

void ReadAllEnergyDiffC0123(double **dEc0123, // output
                            const double h, vector<int> &allispns, vector<int> &allikpts,
                            const int nstates, const int *ibndstart, const int nbnds,
                            const int totnspns, const int totnkpts, const int totnbnds) {
/*
    dEc0123[4][(2 x totstru - 2) x (ndim_loc_row x ndim_loc_col)]
    if t_ion >= totstru, E[t_ion] = E[2 x totstru - 2 - t_ion]
    c0' = c0 + c1h + c2h^2 + c3h^3
    c1' = -(c1 + 2c2h + 3c3h^2)
    c2' = c2 + 3c3h
    c3' = -c3
*/
    ifstream inf;
    if(carrier == "electron" || carrier == "hole") inf.open(namddir + "/tmpEnergy/c0123", ios::in|ios::binary);
    else if(carrier == "exciton") inf.open(namddir + "/tmpDiagonal/c0123", ios::in|ios::binary);
    assert(inf.is_open());
    ifstream inf_direct, inf_exchange;
    if(carrier == "exciton" && is_bse_calc) {
        inf_direct.open(namddir + "/tmpDirect/c0123", ios::in|ios::binary);     assert(inf_direct.is_open());
        inf_exchange.open(namddir + "/tmpExchange/c0123", ios::in|ios::binary); assert(inf_exchange.is_open());
    }
    complex<double> cdtmp;
    const int start_idx = (carrier == "exciton" && lrecomb ? 1 : 0);
    const int readsize = nstates - start_idx;
    const double h2 = h  * h;
    const double h3 = h2 * h;
    const int nspns = allispns.size();
    const int nkpts = allikpts.size();
    const int dim = ( carrier == "exciton" ? nstates : nspns * nkpts * nbnds );
    const int ndim_loc_row = Numroc(dim, MB_ROW, myprow_group, nprow_group);
    const int ndim_loc_col = Numroc(dim, NB_COL, mypcol_group, npcol_group);
    const size_t mn_loc = (size_t)ndim_loc_row * ndim_loc_col;
    int ispn, ikpt;
    double *allenergies = NULL; 
    double *dEc0123_full = NULL;
    if(is_sub_root) {
        allenergies = new double[dim]();
        dEc0123_full = new double[(size_t)dim * dim]();
    }
    for(int t_ion = 0; t_ion < totstru; t_ion++) for(int ic = 0; ic < 4; ic++) {
        if(is_sub_root) {
            if(carrier == "electron" || carrier == "hole") {
                for(int is = 0; is < nspns; is++) {
                    ispn = allispns[is];
                    for(int ik = 0; ik < nkpts; ik++) {
                        ikpt = allikpts[ik];
                        inf.seekg(sizeof(double) * ( (size_t)(t_ion * 4 + ic) * totnspns * totnkpts * totnbnds +
                                                     ispn * totnkpts * totnbnds + ikpt * totnbnds + ibndstart[ispn] ),
                                  ios::beg);
                        inf.read((char*)(allenergies + is * nkpts * nbnds + ik * nbnds), sizeof(double) * nbnds);
                    }
                }
            }
            else if(carrier == "exciton") {
                inf.seekg(sizeof(double) * (t_ion * 4 + ic) * readsize, ios::beg);
                inf.read((char*)(allenergies + start_idx), sizeof(double) * readsize);
                if(is_bse_calc) {
                    if(abs(dynchan[0]) > 1e-8) { // direct term
                        for(int ist = 0; ist < readsize; ist++) {
                            inf_direct.seekg(sizeof(complex<double>) * ( (size_t)(t_ion * 4 + ic) * readsize * readsize 
                                                                       + (ist + ist * readsize) ), ios::beg);
                            inf_direct.read((char*)&cdtmp, sizeof(complex<double>));
                            allenergies[start_idx + ist] += real(cdtmp);
                        }
                    }
                    if(abs(dynchan[1]) > 1e-8) { // exchange term
                        for(int ist = 0; ist < readsize; ist++) {
                            inf_exchange.seekg(sizeof(complex<double>) * ( (size_t)(t_ion * 4 + ic) * readsize * readsize 
                                                                         + (ist + ist * readsize) ), ios::beg);
                            inf_exchange.read((char*)&cdtmp, sizeof(complex<double>));
                            allenergies[start_idx + ist] += real(cdtmp);
                        }
                    }
                }
            }
            for(int ist = 0; ist < dim; ist++) for(int jst = 0; jst < ist; jst++) {
                dEc0123_full[(size_t)ist + jst * dim] = allenergies[ist] - allenergies[jst];
                dEc0123_full[(size_t)jst + ist * dim] = allenergies[ist] - allenergies[jst];
            }
            inf.close();
            if(carrier == "exciton" && is_bse_calc) {
                inf_direct.close();
                inf_exchange.close();
            }
        }
        MPI_Barrier(group_comm);
        Blacs_MatrixDScatter(dim, dim, dEc0123_full, dim,
                                       dEc0123[ic] + t_ion * mn_loc, ndim_loc_row);
    } // t_ion, ic
    for(int t_ion = totstru; t_ion < 2 * totstru - 2; t_ion++) {
        int t_mirror = 2 * totstru - 2 - t_ion;
        #pragma omp parallel for
        for(size_t ij = 0; ij < mn_loc; ij++) {
            dEc0123[0][(size_t)t_ion * mn_loc + ij] = dEc0123[0][(size_t)t_mirror * mn_loc + ij]
                                                    + dEc0123[1][(size_t)t_mirror * mn_loc + ij] * h
                                                    + dEc0123[2][(size_t)t_mirror * mn_loc + ij] * h2
                                                    + dEc0123[3][(size_t)t_mirror * mn_loc + ij] * h3;
            dEc0123[1][(size_t)t_ion * mn_loc + ij] = - ( dEc0123[1][(size_t)t_mirror * mn_loc + ij]
                                                        + dEc0123[2][(size_t)t_mirror * mn_loc + ij] * h  * 2.0
                                                        + dEc0123[3][(size_t)t_mirror * mn_loc + ij] * h2 * 3.0 );
            dEc0123[2][(size_t)t_ion * mn_loc + ij] = dEc0123[2][(size_t)t_mirror * mn_loc + ij]
                                                    + dEc0123[3][(size_t)t_mirror * mn_loc + ij] * h * 3.0;
            dEc0123[3][(size_t)t_ion * mn_loc + ij] = - dEc0123[3][(size_t)t_mirror * mn_loc + ij];
        }
    }
    if(is_sub_root) {
        delete[] allenergies;
        delete[] dEc0123_full;
    }

    // each local dEc0123[] matrix can be regarded as 
    // shape of [mn_loc, 2 x (totstru - 2)] with column-major
    for(int ic = 0; ic < 4; ic++)
    Dimatcopy("CblasColMajor", "CblasTrans", mn_loc, 2 * (totstru - 2),
              1.0, dEc0123[ic], mn_loc, 2 * (totstru - 2));
    // after transpose, its shape change to [2 x (totstru - 2), mn_loc]
    // each column successively involves one energy difference for times

    MPI_Barrier(group_comm);
    return;
}

void ReadAllEnergies(double *allenergies, // output
                     vector<int> &allispns, vector<int> &allikpts,
                     const int nstates, const int *ibndstart, const int nbnds,
                     const int totnspns, const int totnkpts, const int totnbnds) {
/* allenergies[(2 x totstru - 2) x dim] */
    const int nspns = allispns.size();
    const int nkpts = allikpts.size();
    const int dim = ( carrier == "exciton" ? nstates : nspns * nkpts * nbnds );
    const int start_idx = (carrier == "exciton" && lrecomb ? 1 : 0);
    const int readsize = nstates - start_idx;
    complex<double> cdtmp;
    int ispn, ikpt;
    ifstream inf;
    for(int t_ion = world_rk; t_ion < totstru * 2 - 2; t_ion += world_sz) {
        const int t_stru = ( t_ion < totstru ? t_ion : 2 * totstru - 2 - t_ion ) + 1;
        if(carrier == "electron" || carrier == "hole") {
            inf.open(namddir + "/tmpEnergy/" + Int2Str(t_stru), ios::in|ios::binary); assert(inf.is_open());
            for(int is = 0; is < nspns; is++) {
                ispn = allispns[is];
                for(int ik = 0; ik < nkpts; ik++) {
                    ikpt = allikpts[ik];
                    inf.seekg(sizeof(double) * ( ispn * totnkpts * totnbnds + ikpt * totnbnds + ibndstart[ispn] ),
                              ios::beg);
                    inf.read((char*)(allenergies + t_ion * dim +
                                     is * nkpts * nbnds + ik * nbnds), sizeof(double) * nbnds);
                } // kpoint
            } // spin
            inf.close();
        } // electron/hole
        else if(carrier == "exciton") {
            inf.open(namddir + "/tmpDiagonal/" + Int2Str(t_stru), ios::in|ios::binary); assert(inf.is_open());
            inf.read((char*)(allenergies + t_ion * dim + start_idx), sizeof(double) * readsize);
            inf.close();
            if(is_bse_calc) {
                if(abs(dynchan[0]) > 1e-8) { // direct term
                    inf.open(namddir + "/tmpDirect/" + Int2Str(t_stru), ios::in|ios::binary);
                    assert(inf.is_open());
                    for(int ist = 0; ist < readsize; ist++) {
                        inf.seekg(sizeof(complex<double>) * (ist + ist * readsize), ios::beg);
                        inf.read((char*)&cdtmp, sizeof(complex<double>));
                        allenergies[t_ion * dim + start_idx + ist] += real(cdtmp);
                    }
                    inf.close();
                }
                if(abs(dynchan[1]) > 1e-8) { // exchange term
                    inf.open(namddir + "/tmpExchange/" + Int2Str(t_stru), ios::in|ios::binary);
                    assert(inf.is_open());
                    for(int ist = 0; ist < readsize; ist++) {
                        inf.seekg(sizeof(complex<double>) * (ist + ist * readsize), ios::beg);
                        inf.read((char*)&cdtmp, sizeof(complex<double>));
                        allenergies[t_ion * dim + start_idx + ist] += real(cdtmp);
                    }
                    inf.close();
                }
            } // is_bse_calc
        } // exciton
    }
    MPI_Barrier(world_comm);

    for(int rt = 0; rt < world_sz; rt++) {
        for(int t_ion = rt; t_ion < 2 * totstru - 2; t_ion += world_sz)
        MPI_Bcast(allenergies + t_ion * dim, dim, MPI_DOUBLE, rt, world_comm);
    }

    if(is_world_root) {
        ofstream otf(namddir + "/allenergies.dat", ios::out); assert(otf.is_open());
        for(int t_ion = 0; t_ion < totstru; t_ion++) {
            for(int ist = 0; ist < dim; ist++) {
                otf << fixed << setprecision(18) << allenergies[t_ion * dim + ist] << "    ";
            }
            if(t_ion < totstru - 1) otf << endl;
        }
        otf.close();
    }

    MPI_Barrier(world_comm);
    return;
}

void CalcDecoRate(double *decorate, // output
                  vector<int> &allispns, vector<int> &allikpts,
                  const int nstates, const int *ibndstart, const int dimC, const int dimV,
                  const int totnspns, const int totnkpts, const int totnbnds) {
    const int nbnds = dimC + dimV;
    const int dim = ( carrier == "exciton" ? nstates : numspns * numkpts * nbnds );
    const int start_idx = (carrier == "exciton" && lrecomb ? 1 : 0);
    const int nbcv = (carrier == "exciton" ? dimC * dimV : nbnds);
    double *allenergies = new double[(2 * totstru - 2) * dim]();
    ReadAllEnergies(allenergies, allispns, allikpts, nstates, ibndstart, nbnds,
                    totnspns, totnkpts, totnbnds);
    double *dEij = new double[2 * totstru - 2];
    int ii, jj;
    int is, ik, ibcv, js, jk, jbcv;
    double *decorate_full = new double[(size_t)dim * dim]();
    for(size_t ij = world_rk; ij < (size_t)dim * dim; ij += world_sz) {
        ii = ij % dim; jj = ij / dim;
        if(ii == jj) continue;
        if(!lrecomb || (ii > 0 && jj > 0)) {
            IdxNat1toNat3(ii - start_idx, is, ik, ibcv, numspns, numkpts, nbcv);
            IdxNat1toNat3(jj - start_idx, js, jk, jbcv, numspns, numkpts, nbcv);
            if(is != js) continue;
            if(ik != jk) {
                if( carrier == "electron" || carrier == "hole" ) continue;
                if( !is_bse_calc ) continue;
                if( abs(dynchan[0]) < 1e-8 && abs(dynchan[1]) < 1e-8 ) continue;
            }
        }
        #pragma omp parallel for
        for(int t_ion = 0; t_ion < 2 * totstru - 2; t_ion++)
        dEij[t_ion] = allenergies[t_ion * dim + ii] - allenergies[t_ion * dim + jj];
        int NN = totstru; //2 * totstru - 2; // totstru or 2 x totstru - 2, both ok but not best
        DselfCrossCorrelation(NN, dEij);
        CumulativeIntegralTwice(iontime * iontime / hbar / hbar, NN, dEij);
        decorate_full[ij] = 1.0 / GassuianFitting1(NN, 0.0, iontime, dEij);
    }
    MPI_Barrier(world_comm);
    for(int rt = 0; rt < world_sz; rt++) {
        for(size_t ij = rt; ij < (size_t)dim * dim; ij += world_sz)
        MPI_Bcast(decorate_full + ij, 1, MPI_DOUBLE, rt, world_comm);
    }
    MPI_Barrier(world_comm);
    if(is_world_root) {
        ofstream otf(namddir + "/decorate.dat", ios::out);
        if(!otf.is_open()) {
            cout << namddir + "/decorate.dat" << " can't open when storing decoherence rate" << endl; exit(1);
        }
        for(int i = 0; i < dim; i++) {
            for(int j = 0; j < dim; j++) {
                otf << setprecision(16) << setiosflags(ios::scientific) << setiosflags(ios::uppercase)
                    << decorate_full[i + j * dim] << ' ';
            }
            otf << endl;
        }
        otf.close();
    }
    MPI_Barrier(world_comm);
    const int ndim_loc_row = Numroc(dim, MB_ROW, myprow_group, nprow_group);
    Blacs_MatrixDScatter(dim, dim, decorate_full, dim,
                                   decorate, ndim_loc_row);
    delete[] decorate_full;
    delete[] allenergies;
    delete[] dEij;
    MPI_Barrier(world_comm);
    return;
}

void SetIniCurStates(const double *population, const int nstates,
                     const int ntrajs_loc_col, int *currentstates) {
    const int nstates_loc = Numroc(nstates, MB_ROW, myprow_group, nprow_group);
    double *popu_full = NULL;
    if(myprow_group == 0) {
        popu_full = new double[nstates + 1]();
    }
    Blacs_MatrixDGather(nstates, 1, population, nstates_loc, popu_full + 1, nstates,
                        nstates, 1, 0, 0, 0, 0,
                        onecol_root_prow, onecol_root_pcol, ctxt_only_onecol_root,
                        myprow_onecol, mypcol_onecol, nprow_onecol, npcol_onecol, ctxt_onecol);
    if(myprow_group == 0) {
        for(int ist = 1; ist < nstates + 1; ist++) popu_full[ist] += popu_full[ist - 1]; // population to sum(population)
        #pragma omp parallel for
        for(int itraj = 0; itraj < ntrajs_loc_col; itraj++) {
            double r = random01(generator);
            for(int ist = 0; ist < nstates; ist++) {
                if(popu_full[ist] < r && r < popu_full[ist + 1]) { 
                    currentstates[itraj] = ist;
                    break;
                }
            } // ist
        } // itraj
    }
    MPI_Barrier(group_comm);
    MPI_Bcast(currentstates, ntrajs_loc_col, MPI_INT, col_root, col_comm);
    if(myprow_group == 0) delete[] popu_full;
    MPI_Barrier(group_comm);
    return;
}

void WritePopuByCurStates(ofstream &otf, const int *currentstates, 
                          const int nstates, const int ntrajs_loc_col) {
    if(myprow_group == 0) {
        int *popu = new int[nstates]();
        for(int itraj = 0; itraj < ntrajs_loc_col; itraj++) popu[ currentstates[itraj] ]++;
        Blacs_MatrixISum(ctxt_group, "ROW", "I", nstates, 1, popu, nstates, 0, 0);
        if(mypcol_group == 0) {
            otf.write((char*)popu, sizeof(int) * nstates);
            //for(int i = 0; i < nstates; i++) cout << popu[i] << ' '; cout << endl;
        }
        delete[] popu;
    }
    MPI_Barrier(group_comm);
    return;
}

void CalcDecoTime(const int nstates, const int ntrajs, 
                  const int ndim_loc_row, const int ntrajs_loc_col,
                  const double *decorate, const complex<double> *coeff,
                  double *decotime) {
    double *popu = new double[max((size_t)ndim_loc_row * ntrajs_loc_col, (size_t)1)];
    #pragma omp parallel for
    for(size_t ij = 0; ij < (size_t)ndim_loc_row * ntrajs_loc_col; ij++) 
    popu[ij] = abs(coeff[ij]) * abs(coeff[ij]);
    MPI_Barrier(group_comm);
    Pdgemm("N", "N", nstates, ntrajs, nstates,
           1.0, decorate, ndim_loc_row, popu, ndim_loc_row,
           0.0, decotime, ndim_loc_row);
    #pragma omp parallel for
    for(size_t ij = 0; ij < (size_t)ndim_loc_row * ntrajs_loc_col; ij++) 
    decotime[ij] = ( (decotime[ij] < 1.0e-100 || popu[ij] < 1.0e-100) ? namdtim * 100 : 1.0 / decotime[ij] );

    MPI_Barrier(group_comm);
    return;
}

void WhichToDeco(double *decomoments, int *whiches,
                 const int nstates, const int ndim_loc_row, const int ntrajs_loc_col, 
                 const double *decotime) {
    double *decotime_full = NULL;
    if(is_col_root) decotime_full = new double[max((size_t)nstates * ntrajs_loc_col, (size_t)1)]();
    Blacs_MatrixDGather(nstates, ntrajs_loc_col, decotime, ndim_loc_row, decotime_full, nstates,
                        nstates, ntrajs_loc_col, 0, 0, 0, 0,
                        onecol_root_prow, onecol_root_pcol, ctxt_only_onecol_root,
                        myprow_onecol, mypcol_onecol, nprow_onecol, npcol_onecol, ctxt_onecol);
    MPI_Barrier(group_comm);
    
    if(myprow_group == 0) {
        for(int itraj = 0; itraj < ntrajs_loc_col; itraj++) {
            // firstly, collect all decoherent candidates
            vector<int> candidate_which;
            for(int ist = 0; ist < nstates; ist++) {
                if(     decomoments[(size_t)ist + itraj * nstates] 
                    > decotime_full[(size_t)ist + itraj * nstates]) candidate_which.push_back(ist);
            }
            // secondly, randomly determine the "which"
            if(candidate_which.size() > 0) {
                shuffle(candidate_which.begin(), candidate_which.end(), generator);
                whiches[itraj] = candidate_which.front();
                decomoments[ whiches[itraj] + itraj * nstates ] = 0.0;
            }
            else whiches[itraj] = -1;
        }
    }
    MPI_Barrier(group_comm);
    
    MPI_Bcast(whiches, ntrajs_loc_col, MPI_INT, col_root, col_comm);
     
    if(myprow_group == 0) { // update decomoments
        #pragma omp parallel for
        for(size_t ii = 0; ii < (size_t)nstates * ntrajs_loc_col; ii++) decomoments[ii] += iontime;
    }
    
    MPI_Barrier(group_comm);
    return;
}

void Projector(const int *whiches, int *currentstates, const double temp,
               const int nstates, const int ndim_loc_row, const int ntrajs_loc_col,
               complex<double> *coeff, complex<double> **c_onsite) {
    const double odt  = iontime;
    const double odt2 = odt * odt;
    const double odt3 = odt2 * odt;
    int iprow, jpcol, irow_loc, jcol_loc, ij_loc;
    double *allenergies = new double[nstates];
    for(int ist = 0; ist < nstates; ist++) {
        BlacsIdxglb2loc(ist, iprow, irow_loc, 0, nstates, MB_ROW, nprow_group);
        BlacsIdxglb2loc(ist, jpcol, jcol_loc, 0, nstates, NB_COL, npcol_group);
        if(myprow_group == iprow && mypcol_group == jpcol) {
            ij_loc = irow_loc + jcol_loc * ndim_loc_row;
            allenergies[ist] = real( (c_onsite[0][ij_loc] + c_onsite[1][ij_loc] * odt 
                                                          + c_onsite[2][ij_loc] * odt2 + c_onsite[3][ij_loc] * odt3)
                                   * (iu_d * hbar) );
        }
        MPI_Barrier(group_comm);
        MPI_Bcast(allenergies + ist, 1, MPI_DOUBLE, iprow * npcol_group + jpcol, group_comm);
    }
    MPI_Barrier(group_comm);
    
    int iprow_which, which_loc;
    double popuBoltzWhich, dE;
    int is_proj_in;
    for(int itraj = 0; itraj < ntrajs_loc_col; itraj++) {
        if(whiches[itraj] < 0) continue;
        BlacsIdxglb2loc(whiches[itraj], iprow_which, which_loc, 0, nstates, MB_ROW, nprow_group);
        if(myprow_group == iprow_which) {
            popuBoltzWhich = abs(coeff[which_loc + itraj * ndim_loc_row]);
            popuBoltzWhich *= popuBoltzWhich; 
            dE = allenergies[whiches[itraj]] - allenergies[currentstates[itraj]];
            if(dE > 0.0) popuBoltzWhich *= exp( - dE / (kb * temp) );
            is_proj_in = (random01(generator) < popuBoltzWhich ? 1 : 0);
            if(is_proj_in) { // project in
                fill_n(coeff + itraj * ndim_loc_row, ndim_loc_row, complex<double>(0.0, 0.0));
                coeff[which_loc + itraj * ndim_loc_row] = 1.0; // currentstate_loc = which_loc
            }
            else { // project out
                coeff[which_loc + itraj * ndim_loc_row] = 0.0;
            }
        }
        MPI_Barrier(col_comm);
        MPI_Bcast(&is_proj_in, 1, MPI_INT, iprow_which, col_comm);
        if(is_proj_in) { // project in
            currentstates[itraj] = whiches[itraj];
            if(myprow_group != iprow_which) fill_n(coeff + itraj * ndim_loc_row, ndim_loc_row, complex<double>(0.0, 0.0));
        }
        else { // project out
            double norm2 = Pdznrm2(nstates, coeff + itraj * ndim_loc_row, ndim_loc_row, 1,
                                   0, 0, nstates, 1,
                                   0, 0, MB_ROW, NB_COL, ctxt_onecol);
            ZDscal(ndim_loc_row, 1.0 / norm2, coeff + itraj * ndim_loc_row);
        }
        MPI_Barrier(col_comm);
    }

    MPI_Barrier(group_comm);
    return;
}

void DecoCorrectCoeff(const int nstates, const int ndim_loc_row, const int ntrajs_loc_col,
                      const int *currentstates, const double *decorate_col,
                      complex<double> *coeff) {
    int curst, curst_loc, iprow;
    double colnrm, curstnrm;
    for(int itraj = 0; itraj < ntrajs_loc_col; itraj++) {
        curst = currentstates[itraj];
        BlacsIdxglb2loc(curst, iprow, curst_loc, 0, nstates, MB_ROW, nprow_group);
        if(myprow_group != iprow) {
            for(int ist = 0; ist < ndim_loc_row; ist++)
            coeff[ist + itraj * ndim_loc_row] *= exp(- iontime * decorate_col[ist + curst * ndim_loc_row]);
                                                                   // NO "_loc" for curst
        }
        else {
            for(int ist = 0; ist < ndim_loc_row; ist++) if(ist != curst_loc)
            coeff[ist + itraj * ndim_loc_row] *= exp(- iontime * decorate_col[ist + curst * ndim_loc_row]);
                                                                   // NO "_loc" for curst
        }
        MPI_Barrier(col_comm);
        colnrm = Pdznrm2(nstates, coeff + itraj * ndim_loc_row, ndim_loc_row, 1,
                         0, 0, nstates, 1,
                         0, 0, MB_ROW, NB_COL, ctxt_onecol);
        if(myprow_group == iprow) {
            curstnrm = abs(coeff[curst_loc + itraj * ndim_loc_row]);
            curstnrm *= curstnrm;
            colnrm = 1.0 - (colnrm * colnrm - curstnrm);
            // now colnrm = 1 - sum_{i != curst} |c_i|^2
            coeff[curst_loc + itraj * ndim_loc_row] *= sqrt(colnrm / curstnrm);
        }
        MPI_Barrier(col_comm);
    }
    return;
}

void UpdateCurrentStates(const int nstates, const int ntrajs_loc_col,
                         int *currentstates,
                         complex<double> **c_onsite, complex<double> **c_midsite,
                         const complex<double> *coeff, const double temp) {
    const double odt = iontime;       const double mdt = iontime / 2.0;
    const double odt2 = odt * odt;    const double mdt2 = mdt * mdt;
    const double odt3 = odt2 * odt;   const double mdt3 = mdt2 * mdt;
    const int ndim_loc_row = Numroc(nstates, MB_ROW, myprow_group, nprow_group);
    const int ndim_loc_col = Numroc(nstates, NB_COL, mypcol_group, npcol_group);
    const size_t mn_loc = (size_t)ndim_loc_row * ndim_loc_col;
    complex<double> *Hkj = new complex<double>[max(mn_loc, (size_t)1)];
    complex<double> *Hkj_col = new complex<double>[max((size_t)ndim_loc_row * nstates, (size_t)1)];
    #pragma omp parallel for
    for(size_t kj = 0; kj < mn_loc; kj++) {
        Hkj[kj] = c_onsite[0][kj] + c_onsite[1][kj] * odt 
                                  + c_onsite[2][kj] * odt2 + c_onsite[3][kj] * odt3
                + c_midsite[0][kj] + c_midsite[1][kj] * mdt
                                   + c_midsite[2][kj] * mdt2 + c_midsite[3][kj] * mdt3;
    }
    MPI_Barrier(group_comm);
    
    // Hkj to Hkj_col
    int iprow, jpcol;
    int kst_loc;
    for(int kst = 0; kst < nstates; kst++) {
        BlacsIdxglb2loc(kst, jpcol, kst_loc, 0, nstates, NB_COL, npcol_group);
        if(mypcol_group == jpcol) {
            Zcopy(ndim_loc_row, Hkj + kst_loc * ndim_loc_row, 1, 
                                Hkj_col + kst * ndim_loc_row, 1);
            Zgebs2d(ctxt_group, "ROW", "I", ndim_loc_row, 1, Hkj_col + kst * ndim_loc_row, ndim_loc_row);
        }
        else {
            Zgebr2d(ctxt_group, "ROW", "I", ndim_loc_row, 1, Hkj_col + kst * ndim_loc_row, ndim_loc_row,
                    myprow_group, jpcol);
        }
        Blacs_barrier(ctxt_group, "R");
    }
    MPI_Barrier(group_comm);
    
    // set current allenergies
    double *allenergies = new double[nstates];
    int irow_loc, jcol_loc;
    size_t ij_loc;
    for(int ist = 0; ist < nstates; ist++) {
        BlacsIdxglb2loc(ist, iprow, irow_loc, 0, nstates, MB_ROW, nprow_group);
        BlacsIdxglb2loc(ist, jpcol, jcol_loc, 0, nstates, NB_COL, npcol_group);
        if(myprow_group == iprow && mypcol_group == jpcol) {
            ij_loc = (size_t)irow_loc + jcol_loc * ndim_loc_row;
            allenergies[ist] = real( (c_onsite[0][ij_loc] + c_onsite[1][ij_loc] * odt 
                                                          + c_onsite[2][ij_loc] * odt2 + c_onsite[3][ij_loc] * odt3)
                                   * (iu_d * hbar) );
        }
        MPI_Barrier(group_comm);
        MPI_Bcast(allenergies + ist, 1, MPI_DOUBLE, iprow * npcol_group + jpcol, group_comm);
    }
    MPI_Barrier(group_comm);

    // update current state by probability
    int curst, curst_loc, jst_glb;
    complex<double> ck;
    double dEjk;
    double *probability = new double[max(ndim_loc_row, 1)];
    double *probability_col = NULL;
    double colsum;
    if(myprow_group == 0) probability_col = new double[nstates + 1]();
    for(int itraj = 0; itraj < ntrajs_loc_col; itraj++) {
        curst = currentstates[itraj];
        BlacsIdxglb2loc(curst, iprow, curst_loc, 0, nstates, MB_ROW, nprow_group);
        if(myprow_group == iprow) ck = coeff[curst_loc + itraj * ndim_loc_row];
        MPI_Bcast(&ck, 1, MPI_CXX_DOUBLE_COMPLEX, iprow, col_comm);
        if(abs(ck) < 1e-30) {
            for(int jst = 0; jst < ndim_loc_row; jst++) {
                probability[jst] = abs(coeff[jst + itraj * ndim_loc_row]);
                probability[jst] *= probability[jst];
            }
        }
        else {
            for(int jst = 0; jst < ndim_loc_row; jst++)
            probability[jst] = 2 * iontime * 
                               real(coeff[jst + itraj * ndim_loc_row] * conj(Hkj_col[jst + curst * ndim_loc_row]) / ck);
        }
        for(int jst = 0; jst < ndim_loc_row; jst++) {
            jst_glb = BlacsIdxloc2glb(jst, nstates, MB_ROW, myprow_group, nprow_group);
            dEjk = allenergies[jst_glb] - allenergies[curst];
            if(dEjk > 0) probability[jst] *= exp( - dEjk / (kb * temp) );
        }
        MPI_Barrier(col_comm);
        Blacs_MatrixDGather(nstates, 1, probability, ndim_loc_row,
                                        probability_col + 1, nstates,
                            nstates, 1, 0, 0, 0, 0,
                            onecol_root_prow, onecol_root_pcol, ctxt_only_onecol_root,
                            myprow_onecol, mypcol_onecol, nprow_onecol, npcol_onecol, ctxt_onecol);
        if(myprow_group == 0) {
            probability_col[1 + curst] = 0.0;
            colsum = Dasum(nstates, probability_col + 1);
            if(abs(ck) > 1e-30) {
                if(colsum < 1.0) probability_col[1 + curst] = 1.0 - colsum;
            }
            if( !(abs(ck) > 1e-30 && colsum < 1.0) ) Dscal(nstates, 1.0 / colsum, probability_col + 1);
            for(int jst = 1; jst < nstates + 1; jst++)
            probability_col[jst] += probability_col[jst - 1]; // population to sum(population)
            double r = random01(generator);
            for(int jst = 0; jst < nstates; jst++) {
                if(probability_col[jst] < r && r < probability_col[jst + 1]) { 
                    currentstates[itraj] = jst;
                    break;
                }
            }
        }
        MPI_Barrier(col_comm);
    }
    MPI_Barrier(group_comm);
    MPI_Bcast(currentstates, ntrajs_loc_col, MPI_INT, col_root, col_comm);

    delete[] Hkj; delete[] Hkj_col; delete[] allenergies;
    delete[] probability;
    if(myprow_group == 0) delete[] probability_col;
    MPI_Barrier(group_comm);
    return;
}

void MergeAllPopu(const int nstates, const int ntraj_bcks, const int begtime) {
    ifstream inf;
    int *ipopu = new int[(size_t)namdtim * nstates]();
    double *meanpopu = new double[(size_t)namdtim * nstates]();
    string infname;
    for(int itraj_bck = 0; itraj_bck < ntraj_bcks; itraj_bck++) {
        infname = resdir + "/shprop_" + Int2Str(begtime) + "_" + to_string(itraj_bck);
        inf.open(infname.c_str(), ios::out|ios::binary);
        assert(inf.is_open());
        inf.read((char*)ipopu, sizeof(int) * namdtim * nstates);
        remove(infname.c_str());
        #pragma omp parallel for
        for(size_t ii = 0; ii < (size_t)namdtim * nstates; ii++) meanpopu[ii] += ipopu[ii];
        inf.close();
    }
    Dscal((size_t)namdtim * nstates, 1.0 / ntrajec, meanpopu);
    ofstream otf(resdir + "/shprop_"  + Int2Str(begtime), ios::out); assert(otf.is_open());
    for(int t_ion = 0; t_ion < namdtim; t_ion++) {
        for(int ist = 0; ist < nstates; ist++)
        otf << setiosflags(ios::fixed) << setprecision(18) << meanpopu[(size_t)t_ion * nstates + ist] << ' ';
        if(t_ion < namdtim - 1) otf << endl;
    }
    otf.close();
    
    delete[] ipopu; delete[] meanpopu;
    return;
}
