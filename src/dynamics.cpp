#include "dynamics.h"

TIMECOST InterTimeAction(waveclass &wvc1, waveclass &wvc2, 
                         const double dt, const int num) {
    double tstart, tend;
    tstart = omp_get_wtime();
    
    complex<double> *nac = NULL;
    complex<double> *condnac = NULL;
    complex<double> *valenac = NULL;
    complex<double> *c2vnac = NULL;
    complex<double> *res_full = NULL;
    int nst, nst_loc_r, nst_loc_c;
    if(carrier == "electron" || carrier == "hole") { 
        nst = wvc1.nbnds;
        nst_loc_r = Numroc(nst, MB_ROW, myprow_group, nprow_group);
        nst_loc_c = Numroc(nst, NB_COL, mypcol_group, npcol_group);
        nac = new complex<double>[max(wvc1.nspns * nst_loc_r * nst_loc_c, 1)]();
        CalcNAC(dt, wvc1, wvc2, nst, 0, nst, 0, nac,
                (namddir + "/tmpNAC/" + Int2Str(num)).c_str());
    }
    else if(carrier == "exciton") {
        nst = wvc1.dimC;
        nst_loc_r = Numroc(nst, MB_ROW, myprow_group, nprow_group);
        nst_loc_c = Numroc(nst, NB_COL, mypcol_group, npcol_group);
        condnac = new complex<double>[max(wvc1.nspns * wvc1.nkpts * nst_loc_r * nst_loc_c, 1)]();
        CalcNAC(dt, wvc1, wvc2, nst, 0, nst, 0, condnac,
                (namddir + "/tmpCBNAC/" + Int2Str(num)).c_str());
        
        nst = wvc1.dimV;
        nst_loc_r = Numroc(nst, MB_ROW, myprow_group, nprow_group);
        nst_loc_c = Numroc(nst, NB_COL, mypcol_group, npcol_group);
        valenac = new complex<double>[max(wvc1.nspns * wvc1.nkpts * nst_loc_r * nst_loc_c, 1)]();
        CalcNAC(dt, wvc1, wvc2, nst, wvc1.dimC, nst, wvc1.dimC, valenac,
                (namddir + "/tmpVBNAC/" + Int2Str(num)).c_str());

        int nst_row = wvc1.dimC, nst_col = wvc1.dimV;
        nst_loc_r = Numroc(nst_row, MB_ROW, myprow_group, nprow_group);
        nst_loc_c = Numroc(nst_col, NB_COL, mypcol_group, npcol_group);
        c2vnac = new complex<double>[max(wvc1.nspns * wvc1.nkpts * nst_loc_r * nst_loc_c, 1)]();
        CalcNAC(dt, wvc1, wvc2, nst_row, 0, nst_col, wvc1.dimC, c2vnac,
                (namddir + "/tmpC2VNAC/" + Int2Str(num)).c_str(), false); // "false" means needn't phase correction
    }
    
    if(false && carrier == "exciton") {
        const int ispn = 0;
        const int ikpt = 5;
        complex<double> *res_full = NULL;
        nst = wvc1.dimC;
        if(is_sub_root) res_full = new complex<double>[nst * nst];
        nst_loc_r = Numroc(nst, MB_ROW, myprow_group, nprow_group);
        nst_loc_c = Numroc(nst, NB_COL, mypcol_group, npcol_group);
        Blacs_MatrixZGather(nst, nst, condnac + (ispn * wvc1.nkpts + ikpt) * nst_loc_r * nst_loc_c, nst_loc_r,
                                      res_full, nst);
        if(is_sub_root) {
            cout << "conduction band NAC:" << endl;
            for(int ii = 0; ii < nst; ii++) {
                for(int jj = 0; jj < nst; jj++) cout << res_full[ii + jj * nst];
                cout << endl;
            }
            delete[] res_full;
        }
        
        nst = wvc1.dimV;
        if(is_sub_root) res_full = new complex<double>[nst * nst];
        nst_loc_r = Numroc(nst, MB_ROW, myprow_group, nprow_group);
        nst_loc_c = Numroc(nst, NB_COL, mypcol_group, npcol_group);
        Blacs_MatrixZGather(nst, nst, valenac + (ispn * wvc1.nkpts + ikpt) * nst_loc_r * nst_loc_c, nst_loc_r,
                                      res_full, nst);
        if(is_sub_root) {
            cout << "valence band NAC:" << endl;
            for(int ii = 0; ii < nst; ii++) {
                for(int jj = 0; jj < nst; jj++) cout << res_full[ii + jj * nst];
                cout << endl;
            }
            delete[] res_full;
        }
        
        int nst_row = wvc1.dimC, nst_col = wvc1.dimV;
        if(is_sub_root) res_full = new complex<double>[nst_row * nst_col];
        nst_loc_r = Numroc(nst_row, MB_ROW, myprow_group, nprow_group);
        nst_loc_c = Numroc(nst_col, NB_COL, mypcol_group, npcol_group);
        Blacs_MatrixZGather(nst_row, nst_col, c2vnac + (ispn * wvc1.nkpts + ikpt) * nst_loc_r * nst_loc_c, nst_loc_r,
                                              res_full, nst_row);
        if(is_sub_root) {
            cout << "conduction to valence band NAC:" << endl;
            for(int ii = 0; ii < nst_row; ii++) {
                for(int jj = 0; jj < nst_col; jj++) cout << res_full[ii + jj * nst_row];
                cout << endl;
            }
            delete[] res_full;
        }
    }
    
    if(carrier == "electron" || carrier == "hole") delete[] nac;
    else if(carrier == "exciton") { delete[] condnac; delete[] valenac; }
    
    MPI_Barrier(group_comm);
    tend = omp_get_wtime();
    return tend - tstart;
}

void PhaseCorrection(const int stru_beg, const int tot_stru, waveclass *wvc) {
/*
    nstates = nspns * nkpts * (dimC + dimV)
    modify the phase of mpi_split_clr > 0
    phase[stru_beg + tot_stru] *= phase[stru_beg]
*/
    const int nstates = wvc->nstates;
    fstream phasefile;
    complex<double> *phase_beg = new complex<double>[nstates];
    complex<double> *phase_end = new complex<double>[nstates];
    string filename;
    for(int irk = 0; irk < world_sz; irk++) {
        if(world_rk == irk && is_sub_root && mpi_split_clr) { // sequentially loop and only sub_root works
            // read begin phase
            filename = namddir + "/tmpPhase/" + Int2Str(stru_beg);
            phasefile.open(filename.c_str(), ios::in|ios::binary);
            if(!phasefile.is_open()) { cerr << filename << " can't open to read when correcting phase" << endl; exit(1); }
            phasefile.read((char*)phase_beg, sizeof(complex<double>) * nstates);
            phasefile.close();
            
            // read end phase
            filename = namddir + "/tmpPhase/" + Int2Str(stru_beg + tot_stru);
            phasefile.open(filename.c_str(), ios::in|ios::out|ios::binary);
            if(!phasefile.is_open()) { cerr << filename << " can't open when correcting phase" << endl; exit(1); }
            phasefile.read((char*)phase_end, sizeof(complex<double>) * nstates);

            // change end phase
            #pragma omp parallel for
            for(int ist = 0; ist < nstates; ist++) phase_end[ist] *= phase_beg[ist];
            
            // rewrite
            phasefile.seekp(0, ios::beg);
            phasefile.write((char*)phase_end, sizeof(complex<double>) * nstates);
            phasefile.close();
        }
        MPI_Barrier(world_comm);
    }
    delete[] phase_beg; delete[] phase_end;
    if(is_world_root) {
        ofstream otf((namddir + "/.info.tmp").c_str(), ios::out|ios::binary);
        if(!otf.is_open()) { cerr << namddir + "/.info.tmp" << " can't open to write" << endl; exit(1); }
        otf.write((char*)(&wvc->nspns), sizeof(int));
        otf.write((char*)(&wvc->nkpts), sizeof(int));
        otf.write((char*)(&wvc->dimC), sizeof(int));
        otf.write((char*)(&wvc->dimV), sizeof(int));
        // record spns
        otf.write((char*)(&numspns), sizeof(int));
        for(int is = 0; is < numspns; is++) otf.write((char*)&(Spins[is]), sizeof(int));
        // record kpts
        otf.write((char*)(&numkpts), sizeof(int));
        for(int ik = 0; ik < numkpts; ik++) otf.write((char*)&(Kpoints[ik]), sizeof(int));
        // record bnds
        int bnd;
        if(carrier == "electron" || carrier == "hole") {
            otf.write((char*)(&wvc->nbnds), sizeof(int));
            for(int is = 0; is < numspns; is++) for(int ib = 0; ib < wvc->nbnds; ib++) {
                bnd = wvc->bands[is][ib] + 1;
                otf.write((char*)&bnd, sizeof(int));
            }
        }
        else if(carrier == "exciton") {
            otf.write((char*)(&wvc->dimC), sizeof(int));
            otf.write((char*)(&wvc->dimV), sizeof(int));
            for(int is = 0; is < numspns; is++) {
                for(int ic = 0; ic < wvc->dimC; ic++) {
                    bnd = wvc->bands[is][ic] + 1;
                    otf.write((char*)&bnd, sizeof(int));
                }
                for(int iv = 0; iv < wvc->dimV; iv++) {
                    bnd = wvc->bands[is][wvc->dimC + iv] + 1;
                    otf.write((char*)&bnd, sizeof(int));
                }
            }
        }
        otf.close();
    }
    MPI_Barrier(world_comm);
    
    return;
}

void MatrixPhaseModify(const complex<double> *begphase, const char *filename, waveclass *wvc,
                       const bool modifyleft, const bool modifyright) {
    fstream matfile(filename, ios::in|ios::out|ios::binary);
    if(!matfile.is_open()) { Cerr << filename << " can't open to change" << endl; exit(1); }
    const int dimC = wvc->dimC;
    const int dimV = wvc->dimV;
    const int nbnds = dimC + dimV;
    const int nkpts = wvc->nkpts;
    const int nspns = wvc->nspns;
    const int dim = nkpts * dimC * dimV;
    const int nskcv = numspns * nkpts * dimC * dimV;
    complex<double> *mat = new complex<double>[(size_t)nskcv * nskcv];
    matfile.read((char*)mat, sizeof(complex<double>) * nskcv * nskcv); // column-major
    int ik, ic, iv;
    if(modifyleft) for(int isR = 0; isR < numspns; isR++) {
        #pragma omp parallel for private(ik, ic, iv)
        for(int irow = 0; irow < dim; irow++) {
            IdxNat1toNat3(irow, ik, ic, iv, nkpts, dimC, dimV);
            complex<double> iphase = begphase[IdxNat3toNat1(min(isR, totdiffspns - 1), ik, dimC + iv,
                                                            nspns, nkpts, nbnds)]
                              * conj(begphase[IdxNat3toNat1(min(isR, totdiffspns - 1), ik, ic,
                                                            nspns, nkpts, nbnds)]);
            Zscal(nskcv, iphase, mat + (irow + isR * dim), nskcv);
        }
    }

    if(modifyright) for(int isC = 0; isC < numspns; isC++) {
        #pragma omp parallel for private(ik, ic, iv)
        for(int jcol = 0; jcol < dim; jcol++) {
            IdxNat1toNat3(jcol, ik, ic, iv, nkpts, dimC, dimV);
            complex<double> iphase = begphase[IdxNat3toNat1(min(isC, totdiffspns - 1), ik, ic,
                                                            nspns, nkpts, nbnds)]
                              * conj(begphase[IdxNat3toNat1(min(isC, totdiffspns - 1), ik, dimC + iv,
                                                            nspns, nkpts, nbnds)]);
            Zscal(nskcv, iphase, mat + ((jcol + isC * dim) * nskcv), 1);
        }
    }
    matfile.seekp(0, ios::beg);
    matfile.write((char*)mat, sizeof(complex<double>) * nskcv * nskcv);
    
    delete[] mat;
    matfile.close();
    return;
}

void DynamicsMatrixConstruct() {
    COUT << "Dynamics Matrix Construct" << endl;
    waveclass wvc1((runhome + '/' + Int2Str(1) + "/WAVECAR").c_str(), ispinor);
    CheckBasisSets(wvc1); // check if the input basis sets are legal
    MPI_Barrier(world_comm);
    LoadStates2WVC(wvc1, NOT_EXCLUDE_BANDS);
    if( (!CreatNAMDdir(wvc1)) && (!is_sub_calc) ) return;

    // paw initial
    pawpotclass *pawpots = NULL;
    int numelem;
    if(dftcode == "vasp" && is_paw_calc) {
        numelem = ReadAllpawpot(pawpots, (runhome + '/' + Int2Str(1) + "/POTCAR").c_str());
    }
    
    // waveclass 2
    waveclass wvc2((runhome + '/' + Int2Str(1) + "/WAVECAR").c_str(), ispinor);
    LoadStates2WVC(wvc2, NOT_EXCLUDE_BANDS);
    
    if(pawpsae == "ae") {
        wvc1.IniAtoms((runhome + '/' + Int2Str(1) + "/CONTCAR").c_str(), pawpots);
        wvc2.IniAtoms((runhome + '/' + Int2Str(1) + "/CONTCAR").c_str(), pawpots);
    }

    waveclass wvc_aux((runhome + '/' + Int2Str(1) + "/WAVECAR").c_str(), NOT_SPINOR); // for spinor construct if is_make_spinor is true
    socclass soccls(&wvc_aux);
    if(is_make_spinor) {
        wvc_aux.StatesRead(); // load all states index for spinor calculation
        wvc_aux.IniAtoms((runhome + '/' + Int2Str(1) + "/CONTCAR").c_str(), pawpots);
    }

    if(dftcode == "vasp") {
        wvc1.ReadKptvecNpw((runhome + '/' + Int2Str(1) + "/WAVECAR").c_str(), vaspgam, vaspncl);
        wvc1.Getgidx(vaspGAM, vaspver);
        wvc1.IniEigens(IS_MALCOEFF);
        wvc2.ReadKptvecNpw((runhome + '/' + Int2Str(1) + "/WAVECAR").c_str(), vaspgam, vaspncl);
        wvc2.Getgidx(vaspGAM, vaspver);
        wvc2.IniEigens(IS_MALCOEFF);
        if(is_make_spinor) {
            wvc_aux.ReadKptvecNpw((runhome + '/' + Int2Str(1) + "/WAVECAR").c_str(), vaspgam, NOT_VASPNCL);
            wvc_aux.Getgidx(vaspgam, vaspver);
            wvc_aux.IniEigens(NOT_MALCOEFF);
        }
    }

    // paw process after gidx etc. setting
    if(is_paw_calc) {
        if(is_make_spinor) CalcQprojKptsForAllElements(wvc_aux, pawpots, numelem);
        else               CalcQprojKptsForAllElements(wvc1,    pawpots, numelem);
    }
    if(is_make_spinor) soccls.Initial();

    // gw, exciton
    wpotclass wpc(&wvc1);
    excitonclass extc(&wvc1, &wpc);
    if(carrier == "exciton" && is_bse_calc) { 
        if(epsilon < 0) wpc.Initial();
        extc.Initial();
    }
    if(carrier == "exciton" && is_world_root) WriteOutput(4);
    MPI_Barrier(world_comm);

    int loc_strubeg, loc_totstru;
    DivideTot2Loc(totstru, mpi_split_clr, mpi_split_num, loc_strubeg, loc_totstru);
    loc_strubeg += strubeg; // starting from strubeg

    COUT << "Wavefunction classes initialized, begin loop for each structure." << endl;
    MPI_Barrier(group_comm); // wait all waveclass seting well within one color
    waveclass *wvcPtrC = &wvc1, *wvcPtrN = &wvc2;   // pointer for current and next
    WVCBasicProcess(wvcPtrN, &soccls, loc_strubeg); // initial basic process for the first structure
    double tstart, tend, dur1, dur2, dur3;
    for(int tt = loc_strubeg; tt < loc_strubeg + loc_totstru; tt++) { // loop for continuous time, real time = tt
        tstart = omp_get_wtime();
        PointerSwap(wvcPtrC, wvcPtrN); // swap and wvcPtrC stores basic parts of informations 
        dur2 = WVCAdvanProcess(wvcPtrC, &extc, tt);      // for current WVC
        dur1 = WVCBasicProcess(wvcPtrN, &soccls, tt + 1, // for next    WVC
                               tt == (loc_strubeg + loc_totstru - 1));
        dur3 = InterTimeAction(*wvcPtrC, *wvcPtrN, iontime, tt);
        
        tend = omp_get_wtime();
        Cout << Int2Str(tt) << "---" << Int2Str(tt + 1) << flush; 
        Cout << ": wavefunction read " << setw(4) << setfill(' ') << (int)dur1 << " s,    " << flush; 
        Cout << "Time-Local BSE etc. " << setw(7) << setfill(' ') << (int)dur2 << " s,    " << flush; 
        Cout << "Cross-Time NAC etc. " << setw(4) << setfill(' ') << (int)dur3 << " s,    " << flush;
        Cout << "Total " <<  setw(7) << setfill(' ') << (int)(tend - tstart) << " s." << endl;
        cout.copyfmt(iosDefaultState);
    }
    MPI_Barrier(world_comm);
    
    if(is_bse_calc) COUT << "Exciton files prepared, loop for change wavefunction phases >>>>>>";
    tstart = omp_get_wtime();
    PhaseCorrection(loc_strubeg, loc_totstru, &wvc1);
    MPI_Barrier(world_comm);
    if(is_bse_calc) {
        if(mpi_split_clr > 0) {
            ifstream phasefile((namddir + "/tmpPhase/" + Int2Str(loc_strubeg)).c_str(), ios::in|ios::binary);
            if(!phasefile.is_open()) { 
                Cerr << namddir + "/tmpPhase/" + Int2Str(loc_strubeg) << " doesn't open for reading" << endl; 
                exit(1);
            }
            complex<double> *begphase = new complex<double>[wvcPtrN->nstates];
            phasefile.read((char*)begphase, sizeof(complex<double>) * wvcPtrN->nstates);
            phasefile.close();
            
            for(int tt = loc_strubeg + sub_rank; tt < loc_strubeg + loc_totstru; tt += sub_size) {
                MatrixPhaseModify(begphase, (namddir + "/tmpDirect/" + Int2Str(tt)).c_str(), wvcPtrN);
                MatrixPhaseModify(begphase, (namddir + "/tmpExchange/" + Int2Str(tt)).c_str(), wvcPtrN);
                MatrixPhaseModify(begphase, (runhome + '/' + Int2Str(tt) + "/bsevec").c_str(), wvcPtrN, true, false);
            }
            MPI_Barrier(group_comm); //set group_comm to prevent rash delete
            delete[] begphase;
        }
        MPI_Barrier(world_comm);
    }
    tend = omp_get_wtime();
    if(is_bse_calc) COUT << " DONE, used " << setw(6) << setfill(' ') << (int)(tend - tstart) << " s." << endl; cout.copyfmt(iosDefaultState);
    MPI_Barrier(world_comm);

    if(dftcode == "vasp" && is_paw_calc) {
        delete[] pawpots;
    }
    return;
}

void OnlyBSECalc() {
    COUT << "Only BSE Calculation" << endl;
    waveclass wvc((runhome + '/' + Int2Str(1) + "/WAVECAR").c_str(), ispinor);
    CheckBasisSets(wvc); // check if the input basis sets are legal
    MPI_Barrier(world_comm);
    LoadStates2WVC(wvc, NOT_EXCLUDE_BANDS);
    if( (!CreatNAMDdir(wvc)) && (!is_sub_calc) ) return;
    
    // paw initial
    pawpotclass *pawpots = NULL;
    int numelem;
    if(dftcode == "vasp" && is_paw_calc) {
        numelem = ReadAllpawpot(pawpots, (runhome + '/' + Int2Str(1) + "/POTCAR").c_str());
    }
    if(pawpsae == "ae") wvc.IniAtoms((runhome + '/' + Int2Str(1) + "/CONTCAR").c_str(), pawpots);
    
    waveclass wvc_aux((runhome + '/' + Int2Str(1) + "/WAVECAR").c_str(), NOT_SPINOR); // for spinor construct if is_make_spinor is true
    socclass soccls(&wvc_aux);
    if(is_make_spinor) {
        wvc_aux.StatesRead(); // load all states index for spinor calculation
        wvc_aux.IniAtoms((runhome + '/' + Int2Str(1) + "/CONTCAR").c_str(), pawpots);
    }

    if(dftcode == "vasp") {
        wvc.ReadKptvecNpw((runhome + '/' + Int2Str(1) + "/WAVECAR").c_str(), vaspgam, vaspncl);
        wvc.Getgidx(vaspGAM, vaspver);
        wvc.IniEigens(IS_MALCOEFF);
        if(is_make_spinor) {
            wvc_aux.ReadKptvecNpw((runhome + '/' + Int2Str(1) + "/WAVECAR").c_str(), vaspgam, NOT_VASPNCL);
            wvc_aux.Getgidx(vaspgam, vaspver);
            wvc_aux.IniEigens(NOT_MALCOEFF);
        }
    }
    
    // paw process after gidx etc. setting
    if(is_paw_calc) {
        if(is_make_spinor) CalcQprojKptsForAllElements(wvc_aux, pawpots, numelem);
        else               CalcQprojKptsForAllElements(wvc,     pawpots, numelem);
    }
    if(is_make_spinor) soccls.Initial();

    // gw, exciton
    wpotclass wpc(&wvc);
    excitonclass extc(&wvc, &wpc);
    if(epsilon < 0) wpc.Initial();
    extc.Initial();
    if(is_world_root) WriteOutput(4);
    MPI_Barrier(world_comm);
    
    int loc_strubeg, loc_totstru;
    DivideTot2Loc(totstru, mpi_split_clr, mpi_split_num, loc_strubeg, loc_totstru);
    loc_strubeg += strubeg; // starting from strubeg

    COUT << "Wavefunction classes initialized, begin BSE calculations for each structure." << endl;
    MPI_Barrier(group_comm);  // wait all waveclass seting well within one color
    waveclass *wvcPtr = &wvc; // pointer for wvc
    double tstart, tend, dur1, dur2;
    for(int tt = loc_strubeg; tt < loc_strubeg + loc_totstru; tt++) { // loop for continuous time, real time = tt
        tstart = omp_get_wtime();
        dur1 = WVCBasicProcess(wvcPtr, &soccls, tt);
        dur2 = WVCAdvanProcess(wvcPtr, &extc,   tt);
        tend = omp_get_wtime();
        Cout << Int2Str(tt) << flush; 
        Cout << ": wavefunction read " << setw(4) << setfill(' ') << (int)dur1 << " s,    " << flush; 
        Cout << "BSE calculation " << setw(7) << setfill(' ') << (int)dur2 << " s,    " << flush; 
        Cout << "Total " <<  setw(7) << setfill(' ') << (int)(tend - tstart) << " s." << endl;
        cout.copyfmt(iosDefaultState);
    }

    if(dftcode == "vasp" && is_paw_calc) {
        delete[] pawpots;
    }
    MPI_Barrier(world_comm);
    return;
}

void OnlySpinorCalc() {
    COUT << "Only Spinor Wavefunction Calculation" << endl;
    
    // paw initial
    pawpotclass *pawpots = NULL;
    int numelem;
    if(dftcode == "vasp") {
        numelem = ReadAllpawpot(pawpots, (runhome + '/' + Int2Str(1) + "/POTCAR").c_str());
    }
    
    waveclass wvc_aux((runhome + '/' + Int2Str(1) + "/WAVECAR").c_str(), NOT_SPINOR); // for spinor construct if is_make_spinor is true
    socclass soccls(&wvc_aux);
    wvc_aux.StatesRead(); // load all states index for spinor calculation
    wvc_aux.IniAtoms((runhome + '/' + Int2Str(1) + "/CONTCAR").c_str(), pawpots);

    if(dftcode == "vasp") {
        wvc_aux.ReadKptvecNpw((runhome + '/' + Int2Str(1) + "/WAVECAR").c_str(), vaspgam, NOT_VASPNCL);
        wvc_aux.Getgidx(vaspgam, vaspver);
        wvc_aux.IniEigens(NOT_MALCOEFF);
    }
    
    // paw process after gidx etc. setting
    CalcQprojKptsForAllElements(wvc_aux, pawpots, numelem);
    soccls.Initial();

    int loc_strubeg, loc_totstru;
    DivideTot2Loc(totstru, mpi_split_clr, mpi_split_num, loc_strubeg, loc_totstru);
    loc_strubeg += strubeg; // starting from strubeg

    COUT << "Wavefunction classes initialized, begin spinor calculations for each structure." << endl;
    MPI_Barrier(group_comm);  // wait all waveclass seting well within one color
    double dur;
    waveclass *wvcPtr = &wvc_aux; // pointer for wvc_aux
    for(int tt = loc_strubeg; tt < loc_strubeg + loc_totstru; tt++) { // loop for continuous time, real time = tt
        dur = WVCBasicProcess(wvcPtr, &soccls, tt);
        cout.copyfmt(iosDefaultState);
    }

    if(dftcode == "vasp" && is_paw_calc) {
        delete[] pawpots;
    }
    MPI_Barrier(world_comm);
    return;
}

void CheckBasisSpace(vector<int> *&allbands, int &nstates, int &nspns, int &nkpts, int &dimC, int &dimV,
                     vector<int> &allispns, vector<int> &allikpts, int *&ibndstart,
                     int &totnspns, int &totnkpts, int &totnbnds) {
    int rdtotdiffspns, rdtotnkpts, readdimC, readdimV;
    ReadInfoTmp(rdtotdiffspns, rdtotnkpts, readdimC, readdimV);
    totnspns = rdtotnkpts;
    if(carrier == "electron" || carrier == "hole") {
        vector<int> readspns, readkpts, readbnds;
        vector<int> bnds_onespin;
        ReadInfoTmp1(readspns, readkpts, readbnds);
        const int rnbnds = readbnds.size() / readspns.size();
        
        
        for(int is = 0; is < numspns; is++) 
        assert( FindIndex(readspns, Spins[is]) != -1 );
        for(int ik = 0; ik < numkpts; ik++) {
            assert( FindIndex(readkpts, Kpoints[ik]) != -1 );
            allikpts.push_back( FindIndex(readkpts, Kpoints[ik]) );
        }
        
        const int exlnum = exclude[0];
        for(int is = 0; is < numspns; is++) {
            allbands[is].clear();
            bnds_onespin.clear();
            copy(readbnds.begin() + is * rnbnds , readbnds.begin() + (is + 1) * rnbnds, back_inserter(bnds_onespin));
            for(int ib = bandmin[is]; ib <= bandmax[is]; ib++) {
                if(find(exclude.begin() + 1 +  is      * exlnum, 
                        exclude.begin() + 1 + (is + 1) * exlnum, ib)
                   ==   exclude.begin() + 1 + (is + 1) * exlnum) allbands[is].push_back(ib); 
                assert( FindIndex(bnds_onespin, allbands[is].back()) != -1 );
            }
            allispns.push_back( min(Spins[is], rdtotdiffspns - 1) );
            ibndstart[is] = FindIndex(bnds_onespin, allbands[is][0]);
        }

        if(numspns == 2) assert( allbands[0].size() ==  allbands[1].size() );
        
        if(carrier == "electron") { dimC = allbands[0].size(); dimV = 0; }
        else                      { dimC = 0; dimV = allbands[0].size(); }
        nstates = numspns * numkpts * (dimC + dimV);

        totnkpts = readkpts.size(); assert(totnkpts == rdtotnkpts);
        totnbnds = readbnds.size(); assert(totnbnds == (readdimC + readdimV));
    }
    else if(carrier == "exciton") {
        vector<int> readspns, readkpts, readcbds, readvbds;
        ReadInfoTmp2(readspns, readkpts, readcbds, readvbds);
        
        for(int is = 0; is < numspns; is++) 
        assert( FindIndex(readspns, Spins[is]) != -1 );

        const int readnspns = readspns.size(); assert(readnspns == numspns);
        const int readnkpts = readkpts.size(); assert(readnkpts == numkpts);
        dimC = readcbds.size() / readnspns;
        dimV = readvbds.size() / readnspns;
        nstates = numspns * numkpts * dimC * dimV + ( lrecomb ? 1 : 0 ); 
        
        for(int is = 0; is < numspns; is++) {
            allbands[is].clear();
            copy(readcbds.begin() + is * dimC, readcbds.begin() + (is + 1) * dimC, back_inserter(allbands[is]));
            copy(readvbds.begin() + is * dimV, readvbds.begin() + (is + 1) * dimV, back_inserter(allbands[is]));
        }

        assert(rdtotdiffspns == totdiffspns);
    }
    nspns = numspns; nkpts = numkpts;
   
    MPI_Barrier(world_comm);
    return;
}

void RunDynamics() {
    //// determine state space
    int nstates, nspns, nkpts, dimC, dimV;
    vector<int> *allbands = new vector<int>[2]();
    vector<int> allispns, allikpts;
    int *ibndstart = new int[2]();
    int totnspns, totnkpts, totnbnds;
    CheckBasisSpace(allbands, nstates, nspns, nkpts, dimC, dimV,
                    allispns, allikpts, ibndstart,
                    totnspns, totnkpts, totnbnds);
    
    //// determing ntrajs by fssh or dish/dcsh
    const int ntrajs = ( taskmod == "fssh" ? 1 : bckntrajs );

    //// set constructs and malloc memory
    const int ndim_loc_row = Numroc(nstates, MB_ROW, myprow_group, nprow_group);
    const int ndim_loc_col = Numroc(nstates, NB_COL, mypcol_group, npcol_group);
    const int nvec_loc_col = Numroc(ntrajs,  NB_COL, mypcol_group, npcol_group); // 1 or 0 for fssh
    const int ntrajs_loc_col = nvec_loc_col;
    const size_t mn_loc = (size_t)ndim_loc_row * ndim_loc_col;
    const double h = iontime / neleint;
    complex<double> *coeff = new complex<double>[max(ndim_loc_row * nvec_loc_col, 1)]();
    complex<double> *fullcoeff = NULL;
    double *population = new double[max(ndim_loc_row * nvec_loc_col, 1)]();
    double *fullpopu = NULL;
    if(is_sub_root) { fullcoeff = new complex<double>[nstates]; fullpopu = new double[nstates]; } MPI_Barrier(world_comm);
    complex<double> **c_onsite  = new complex<double>*[4];
    complex<double> **c_midsite = new complex<double>*[4];
    for(int i = 0; i < 4; i++) {
        c_onsite[i]  = new complex<double>[max(mn_loc, (size_t)1)]();
        c_midsite[i] = new complex<double>[max(mn_loc, (size_t)1)]();
    }
    complex<double> *a0 = new complex<double>[max(mn_loc, (size_t)1)]();
    complex<double> *a1 = new complex<double>[max(mn_loc, (size_t)1)]();
    complex<double> *a2 = new complex<double>[max(mn_loc, (size_t)1)]();
    complex<double> *a3 = new complex<double>[max(mn_loc, (size_t)1)]();
    complex<double> *om_oih = new complex<double>[max(mn_loc, (size_t)1)]();
    double *probmat = new double[max(mn_loc, (size_t)1)]();
    complex<double> *vmat = new complex<double>[max(mn_loc, (size_t)1)]();
    /*vector<int> locbeg, glbbeg, bcklen;
    BlacsBegidxBlocklen(myprow_group, nprow_group, nstates, MB_ROW, locbeg, glbbeg, bcklen);*/

    /// dish/dcsh: determine decorate
    double *decorate = NULL;
    double *decorate_col = NULL;
    if( taskmod == "dish" || taskmod == "dcsh" ) {
        decorate = new double[max(mn_loc, (size_t)1)]();
        CalcDecoRate(decorate,
                     allispns, allikpts, nstates, ibndstart, dimC, dimV,
                     totnspns, totnkpts, totnbnds);
        if(taskmod == "dcsh") {
            decorate_col = new double[max(ndim_loc_row * nstates, 1)]();
            int jj_loc, jpcol;
            for(int jj = 0; jj < nstates; jj++) {
                BlacsIdxglb2loc(jj, jpcol, jj_loc, 0, nstates, NB_COL, npcol_group);
                if(mypcol_group == jpcol) {
                    Dcopy(ndim_loc_row, decorate + jj_loc * ndim_loc_row, 1, 
                                        decorate_col + jj * ndim_loc_row, 1);
                    Dgebs2d(ctxt_group, "ROW", "I", ndim_loc_row, 1, decorate_col + jj * ndim_loc_row, ndim_loc_row);
                }
                else {
                    Dgebr2d(ctxt_group, "ROW", "I", ndim_loc_row, 1, decorate_col + jj * ndim_loc_row, ndim_loc_row,
                            myprow_group, jpcol);
                }
                Blacs_barrier(ctxt_group, "R");
            }
        }
        MPI_Barrier(world_comm);
    }

    //// loop for sample
    if(is_world_root) CheckIniconFile(allbands); MPI_Barrier(world_comm);
    double tstart, tend;
    COUT << endl << "Start to calculate dynamics for " << nsample << " sample(s):" << endl; MPI_Barrier(world_comm);
    int begtime;
    ofstream outFileC, outFileP;
    for(int ismp = mpi_split_clr; ismp < nsample; ismp += mpi_split_num) {
        tstart = omp_get_wtime();
        if(taskmod == "fssh") {
            SetIniCoeff(coeff, population, nstates, ismp, begtime, allbands); // begtime will read here
            if(is_sub_root) {
                outFileC.open(resdir + "/psict_"  + Int2Str(begtime), ios::out); assert(outFileC.is_open());
                outFileP.open(resdir + "/shprop_" + Int2Str(begtime), ios::out); assert(outFileP.is_open());
            }
            MPI_Barrier(group_comm);
            WritePzvec(outFileC, nstates, ndim_loc_row, coeff,      fullcoeff);
            WritePdvec(outFileP, nstates, ndim_loc_row, population, fullpopu);
            ReadMidsiteC(begtime - 1, nstates, nspns, nkpts, dimC, dimV,
                         allispns, allikpts, ibndstart, totnspns, totnkpts, totnbnds, 
                         c_midsite);
            for(int t_ion = begtime; t_ion < begtime + namdtim - 1; t_ion++) {
                CoeffUpdate(h, t_ion, nstates, nspns, nkpts, dimC, dimV,
                            allispns, allikpts, ibndstart, totnspns, totnkpts, totnbnds, 
                            c_onsite, c_midsite, coeff, 1, a0, a1, a2, a3, om_oih);
                PopuUpdateFSSH(nstates, coeff, population, 
                               c_onsite, c_midsite, probmat, vmat, dyntemp);
                WritePzvec(outFileC, nstates, ndim_loc_row, coeff, fullcoeff, t_ion < begtime + namdtim - 2);
                WritePdvec(outFileP, nstates, ndim_loc_row, population, fullpopu, t_ion < begtime + namdtim - 2);
            }
            if(is_sub_root) { outFileC.close(); outFileP.close(); }
        }
        else if(taskmod == "dish" || taskmod == "dcsh") {
            int *currentstates  = new int[max(ntrajs_loc_col, 1)]();
            int *whiches        = new int[max(ntrajs_loc_col, 1)]();
            double *decomoments = NULL;
            if(myprow_group == 0) decomoments = new double[max((size_t)nstates * ntrajs_loc_col, (size_t)1)]();
            double *decotime = new double[max((size_t)ndim_loc_row * ntrajs_loc_col, (size_t)1)]();

            for(int itraj_bck = 0; itraj_bck < ntrajec / ntrajs; itraj_bck++) {
                SetIniCoeff(coeff, population, nstates, ismp, begtime, allbands, ntrajs); // begtime will read here
                if(is_sub_root) {
                    outFileP.open(resdir + "/shprop_" + Int2Str(begtime) + "_" + to_string(itraj_bck),
                                  ios::out|ios::binary);
                    assert(outFileP.is_open());
                }
                SetIniCurStates(population, nstates, ntrajs_loc_col, currentstates);
                WritePopuByCurStates(outFileP, currentstates, nstates, ntrajs_loc_col);
                ReadMidsiteC(begtime - 1, nstates, nspns, nkpts, dimC, dimV,
                             allispns, allikpts, ibndstart, totnspns, totnkpts, totnbnds, 
                             c_midsite);
                for(int t_ion = begtime; t_ion < begtime + namdtim - 1; t_ion++) {
                    CoeffUpdate(h, t_ion, nstates, nspns, nkpts, dimC, dimV,
                                allispns, allikpts, ibndstart, totnspns, totnkpts, totnbnds, 
                                c_onsite, c_midsite, coeff, ntrajs, a0, a1, a2, a3, om_oih);
                    if(taskmod == "dish") {
                        CalcDecoTime(nstates, ntrajs, ndim_loc_row, ntrajs_loc_col,
                                     decorate, coeff, decotime);
                        WhichToDeco(decomoments , whiches, nstates, ndim_loc_row, ntrajs_loc_col, decotime);
                        Projector(whiches, currentstates, dyntemp, nstates, ndim_loc_row, ntrajs_loc_col,
                                  coeff, c_onsite);
                    }
                    else if(taskmod == "dcsh") {
                        DecoCorrectCoeff(nstates, ndim_loc_row, ntrajs_loc_col, currentstates, decorate_col, coeff);
                        UpdateCurrentStates(nstates, ntrajs_loc_col, currentstates,
                                            c_onsite, c_midsite, coeff, dyntemp);
                    }
                    WritePopuByCurStates(outFileP, currentstates, nstates, ntrajs_loc_col);
                }
                if(is_sub_root) outFileP.close();
            }
            if(is_sub_root) MergeAllPopu(nstates, ntrajec / ntrajs, begtime);
            MPI_Barrier(group_comm);

            delete[] currentstates;
            delete[] whiches;
            if(myprow_group == 0) delete[] decomoments;
            delete[] decotime;
        }
        MPI_Barrier(group_comm);
        tend = omp_get_wtime();
        Cout << taskmod << " dynamics beginning from time = " 
             << std::right << setw(5) << setfill(' ') << begtime << "  finished, used"
             << std::right << setw(7) << setfill(' ') << (int)(tend - tstart) << " s." << endl;
        cout.copyfmt(iosDefaultState);
        MPI_Barrier(group_comm);
    }
    MPI_Barrier(world_comm);
    
    //// free memory
    delete[] coeff; delete[] population;
    if(is_sub_root) { delete[] fullcoeff; delete fullpopu; }
    for(int i = 0; i < 4; i++) { delete[] c_onsite[i]; delete[] c_midsite[i]; }
    delete[] c_onsite; delete[] c_midsite;
    delete[] a0; delete[] a1; delete[] a2; delete[] a3; delete[] om_oih;
    delete[] probmat; delete[] vmat;
    delete[] allbands;
    delete[] ibndstart;
    if( taskmod == "dish" || taskmod == "dcsh" ) delete[] decorate;
    if( taskmod == "dish" ) delete[] decorate_col;
    
    MPI_Barrier(world_comm);
    return;
}
