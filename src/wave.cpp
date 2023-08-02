#include "wave.h"

void LoadStates2WVC(waveclass &wvc, const int is_exclude_bands) {
    if(carrier == "electron" || carrier == "hole") {
        if(is_exclude_bands) wvc.StatesRead(Spins, Kpoints, bandtop, bandbot, exclude);
        else {
            vector<int> not_exclude_tmp{0};
            wvc.StatesRead(Spins, Kpoints, bandbot, bandtop, not_exclude_tmp);
        }
    }
    else if(carrier == "exciton") wvc.StatesRead(Spins, Kpoints, condmin, condmax, valemin, valemax, exclude);

    return;
}

TIMECOST WVCBasicProcess(waveclass *&wvc, socclass *soccls, const int num, bool isTmp) {
/*
    soccls is auxiliary for spinor construct
*/
    double tstart, tend;
    tstart = omp_get_wtime();
    // Prepare: Comfirm the "wavecar" file
    const int no_num_dir = access((runhome + '/' + Int2Str(num)).c_str(), F_OK);
    int inum = num; // relocation to the first one if over strutures boundary
    if(num > laststru || no_num_dir) { inum = 1; isTmp = true; } 
    string wavecar; // vasp WAVECAR file
    string normcar; // home-made normcar file
    if(is_paw_calc)  normcar = runhome + '/' + Int2Str(inum) + "/normcar";
    bool is_delete_wavecar = false;
    if(is_make_spinor) {
        wavecar = runhome + '/' + Int2Str(inum) + "/wavecar"; // home-made spinor "wavecar"s are lower-case 
        if(isTmp && soccls->MakeSpinor((runhome + '/' + Int2Str(inum) + '/').c_str(), "_tmp")) { // MakeSpinor return ture if do make a spinor within this function
            wavecar += "_tmp"; 
            is_delete_wavecar = true;
        }
        else soccls->MakeSpinor((runhome + '/' + Int2Str(inum) + '/').c_str());
    }
    else {
        wavecar = runhome + '/' + Int2Str(inum) + "/WAVECAR";
    }
    if(taskmod == "spinor") { // Only make spinor
        tend = omp_get_wtime();
        return tend - tstart; 
    }

    // action
    wvc->ReadCoeff(wavecar.c_str(), vaspGAM, (carrier == "exciton" && is_bse_calc) ? NEED_REALC : NO_NEED_REALC);
    if(is_paw_calc) {
        if( wvc->ReadProjPhi(normcar.c_str()) ) { // if true, read successful, don't need construct again
        }
        else {
            wvc->AtomsRefresh((runhome + '/' + Int2Str(inum) + "/CONTCAR").c_str());
            wvc->CalcProjPhi(normcar.c_str(), !isTmp);
        }
    }

    if(is_sub_root && is_delete_wavecar) remove(wavecar.c_str());
    MPI_Barrier(group_comm);
    tend = omp_get_wtime();
    return tend - tstart; 
}

void WriteEnergyDiff(waveclass *wvc, const int dirnum, const double gapadd) {
    const int dimC = wvc->dimC;
    const int dimV = wvc->dimV;
    const int NBtot = dimC + dimV;
    const int NKSCtot = wvc->nkpts;
    const int dim = NKSCtot * dimC * dimV;
    int ss, kk, cb, vb;
    double *energydiff = new double[numspns * dim];
    #pragma omp parallel for private(ss, kk, cb, vb)
    for(int ii = 0; ii < numspns * dim; ii++) {
        ss = ii / dim;
        IdxNat1toNat3(ii % dim, kk, cb, vb, NKSCtot, dimC, dimV);
        energydiff[ii]
        = wvc->eigens[min(totdiffspns - 1, ss) * NKSCtot * NBtot + kk * NBtot +        cb].energy
        - wvc->eigens[min(totdiffspns - 1, ss) * NKSCtot * NBtot + kk * NBtot + dimC + vb].energy + gapadd;
    }
    
    ofstream diagout((namddir + "/tmpDiagonal/" + Int2Str(dirnum)).c_str(), ios::out|ios::binary);
    if(!diagout.is_open())  { cerr << "ERROR: " << namddir + "/tmpDiagonal/" + Int2Str(dirnum) << " can't open" << endl; exit(1); }
    diagout.write((char*)energydiff, sizeof(double) * numspns * dim);
    diagout.close();
    delete[] energydiff;

    return;
}

TIMECOST WVCAdvanProcess(waveclass *&wvc, excitonclass *extc, const int num) {
    double tstart, tend;
    tstart = omp_get_wtime();

    if(is_sub_root) {
        wvc->WriteBandEnergies((namddir + "/tmpEnergy/" + Int2Str(num)).c_str());
        wvc->WriteExtraPhases((namddir + "/tmpPhase/" + Int2Str(num + 1)).c_str());
                                                      // should "num + 1" means calculating with previous wavefunction
    }
    MPI_Barrier(group_comm);

    if(carrier == "exciton") {
        if(is_bse_calc && !bsedone) {
            extc->wvc = wvc;
            wvc->CalcRealc(); // obtain the real-space coefficients, need delete "realc" space later
            extc->ExcitonMatrix(num);
        }
        else if(is_bse_calc && bsedone) {
            if(is_sub_root) {
                ifstream src1( (           "tmpDirect/" + Int2Str(num)).c_str(), ios::in|ios::binary );
                ofstream dst1( (namddir + "/tmpDirect/" + Int2Str(num)).c_str(), ios::out|ios::binary );
                assert(src1.is_open()); assert(dst1.is_open());
                dst1 << src1.rdbuf();
                src1.close(); dst1.close();
                
                ifstream src2( (           "tmpExchange/" + Int2Str(num)).c_str(), ios::in|ios::binary );
                ofstream dst2( (namddir + "/tmpExchange/" + Int2Str(num)).c_str(), ios::out|ios::binary );
                assert(src2.is_open()); assert(dst2.is_open());
                dst2 << src2.rdbuf();
                src2.close(); dst2.close();

            }
            MPI_Barrier(group_comm);
        }
        else {
            if(is_sub_root) WriteEnergyDiff(wvc, num, gapdiff);
            MPI_Barrier(group_comm);
        }
    }

    tend = omp_get_wtime();
    return tend - tstart;
}

void CalcQprojKptsForAllElements(waveclass &wvc, pawpotclass *&pawpots, const int numelem) {
    //for(int ielem = 0; ielem < numelem; ielem++) pawpots[ielem].CalcQprojKpts_full(wvc.totnkpts, wvc.kpoints, wvc.npw, wvc.gkabs, wvc.gktheta, wvc.gphi, world_rk, world_sz, world_comm);
    for(int ielem = 0; ielem < numelem; ielem++) pawpots[ielem].CalcQprojKpts(wvc.totnkpts, wvc.kpoints, wvc.npw, wvc.gkabs, wvc.gktheta, wvc.gkphi);
    MPI_Barrier(world_comm);
    return;
}
