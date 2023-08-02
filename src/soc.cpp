#include "soc.h"

int socclass::CheckSpinorWavecar() {
    int wavecar_ok = 1;
    if(is_sub_root) {
        while(wavecar_ok) {
            ifstream wavecar((dirpath + "wavecar").c_str(), ios::in|ios::binary);
            if(!wavecar.is_open()) { wavecar_ok = 0; break; }
            double rdum;
            wavecar.read((char*)&rdum, sizeof(double));
            if( (size_t)rdum != 2 * maxnpw * sizeof(complex<float>) ) { wavecar_ok = 0; wavecar.close(); break; }
            wavecar.seekg(0, ios::end);
            if( (size_t)(2 + wvc->nkpts * (1 + 2 * wvc->nbnds)) *
                (size_t)rdum != wavecar.tellg() ) wavecar_ok = 0;
            
            wavecar.close();
            break;
        }
    }
    MPI_Bcast(&wavecar_ok, 1, MPI_INT, sub_root, group_comm);
    return wavecar_ok;
}

void socclass::GetProjPhi() { 
    wvc->ReadCoeff((dirpath + "WAVECAR").c_str(), NOT_VASPGAM, NO_NEED_REALC);
    wvc->AtomsRefresh((dirpath + "CONTCAR").c_str());
    wvc->CalcProjPhi((dirpath + "NORMCAR").c_str(), false);

    return;
}

void socclass::SetupLS() {
    isSetupLS = true;
/*  ref: zqj github
    
    Calculate < Y_lm; sigma | L.S | Y_l'm'; sigma'>
    in which, Y_lm is the REAL spherical harmonics.
    Let Y_l^m = |l,m> is the COMPLEX spherical harmonics, then
            
               /  i/sqrt(2) (|l,m> - (-1)^m|l,-m>)   m < 0
              | 
       Y_lm = |              |l,0>                   m = 0
              |        
               \  1/sqrt(2) (|l,-m> + (-1)^m|l,m>)   m > 0
    
    that is, there is a transform matrix U_C2R: ii:0~2l <==> m:-l~+l
    m = ii - l
    if(m < 0):
    U_C2R[ii,    ii] =  i / sqrt(2)
    U_C2R[2l-ii, ii] = -i * (-1)^m / sqrt(2)
    if(m = 0) U_C2R[ii,ii] = 1.0
    if(m > 0):
    U_C2R[2l-ii, ii] = 1 / sqrt(2)
    U_C2R[ii,    ii] = (-1)^m / sqrt(2)

    L.S = (L_{+}S_{-} + L_{-}S_{+}) / 2 + L_zS_z
    where 
                   /0    1\             /0    0\        hbar /1    0\
       S_{+} = hbar|      |   S_{-}=hbar|      |   S_z=------|      |
                   \0    0/             \1    0/          2  \0   -1/
       
       L_{+}|l,m> = sqrt[(l-m)(l+m+1)]hbar|l,m+1>
       L_{-}|l,m> = sqrt[(l+m)(l-m+1)]hbar|l,m-1>
    then
    <l,m|L_{+}|l',m'> = sqrt[(l-m')(l+m'+1)]hbar * delta_{m, m'+1}delta_{l,l'}
    <l,m|L_{-}|l',m'> = sqrt[(l+m')(l-m'+1)]hbar * delta_{m, m'-1}delta_{l,l'}
    <l,m|L_{z}|l',m'> = m'hbar * delta_{m,m'}delta_{l,l'}
*/
    for(int l = 1; l <= 4; l++) {
        int tlp1 = 2 * l + 1;
        Lplus_mmp[l - 1]  = new complex<double>[tlp1 * tlp1]();
        Lminus_mmp[l - 1] = new complex<double>[tlp1 * tlp1]();
        Lz_mmp[l - 1]     = new complex<double>[tlp1 * tlp1]();
        U_C2R[l - 1]      = new complex<double>[tlp1 * tlp1]();
        LS_mmp[l - 1]     = new complex<double>[4 * tlp1 * tlp1]();
        #pragma omp parallel for
        for(int ii = 0; ii < tlp1; ii++) { // loop for cloumns
            int mp = ii - l;
            if(mp + 1 <=  l)  Lplus_mmp[l - 1][(ii + 1) * tlp1 + ii] = sqrt((l - mp) * (l + mp + 1));
            if(mp - 1 >= -l) Lminus_mmp[l - 1][(ii - 1) * tlp1 + ii] = sqrt((l + mp) * (l - mp + 1));
            Lz_mmp[l - 1][ii * tlp1 + ii] = mp;
            if(mp < 0) {
                U_C2R[l - 1][         ii  * tlp1 + ii] =  iu_d / sqrt(2.0);
                U_C2R[l - 1][(2 * l - ii) * tlp1 + ii] = -iu_d / sqrt(2.0) * pow(-1, mp);
            }
            else if(mp == 0) U_C2R[l - 1][ii  * tlp1 + ii] =  1.0;
            else {
                U_C2R[l - 1][(2 * l - ii) * tlp1 + ii] = 1.0 / sqrt(2.0);
                U_C2R[l - 1][         ii  * tlp1 + ii] = 1.0 / sqrt(2.0) * pow(-1, mp);
            }
        }

        // <up|SO|up>, L_{z}S_{z} works, <up|S_{z}|up> = 1/2
        Zgemm("CblasRowMajor", "CblasNoTrans", "CblasNoTrans", tlp1, tlp1, tlp1,
              1.0, Lz_mmp[l - 1], tlp1, U_C2R[l - 1], tlp1, 0.0, LS_mmp[l - 1], tlp1);
        Zgemm_onsite_R("CblasRowMajor", "CblasConjTrans", "CblasNoTrans", tlp1, tlp1, tlp1,
                       0.5, U_C2R[l - 1], tlp1, 0.0, LS_mmp[l - 1], tlp1, tlp1);
        // <up|SO|dn>, L_{-}S_{+}/2 works
        Zgemm("CblasRowMajor", "CblasNoTrans", "CblasNoTrans", tlp1, tlp1, tlp1,
              1.0, Lminus_mmp[l - 1], tlp1, U_C2R[l - 1], tlp1, 0.0, LS_mmp[l - 1] + tlp1 * tlp1, tlp1);
        Zgemm_onsite_R("CblasRowMajor", "CblasConjTrans", "CblasNoTrans", tlp1, tlp1, tlp1,
                       0.5, U_C2R[l - 1], tlp1, 0.0, LS_mmp[l - 1] + tlp1 * tlp1, tlp1, tlp1);
        // <dn|SO|up>, L_{+}S_{-}/2 works
        Zgemm("CblasRowMajor", "CblasNoTrans", "CblasNoTrans", tlp1, tlp1, tlp1,
              1.0, Lplus_mmp[l - 1], tlp1, U_C2R[l - 1], tlp1, 0.0, LS_mmp[l - 1] + 2 * tlp1 * tlp1, tlp1);
        Zgemm_onsite_R("CblasRowMajor", "CblasConjTrans", "CblasNoTrans", tlp1, tlp1, tlp1,
                       0.5, U_C2R[l - 1], tlp1, 0.0, LS_mmp[l - 1] + 2 * tlp1 * tlp1, tlp1, tlp1);
        // <dn|SO|up>, L_{z}S_{z} works, <dn|S_{z}|dn> = -1/2
        Zgemm("CblasRowMajor", "CblasNoTrans", "CblasNoTrans", tlp1, tlp1, tlp1,
              1.0, Lz_mmp[l - 1], tlp1, U_C2R[l - 1], tlp1, 0.0, LS_mmp[l - 1] + 3 * tlp1 * tlp1, tlp1);
        Zgemm_onsite_R("CblasRowMajor", "CblasConjTrans", "CblasNoTrans", tlp1, tlp1, tlp1,
                       -0.5, U_C2R[l - 1], tlp1, 0.0, LS_mmp[l - 1] + 3 * tlp1 * tlp1, tlp1, tlp1);
        
        /*cout << "U_C2R[" << l << "]" << endl;
        for(int ii = 0; ii < tlp1; ii++) {
            for(int jj = 0; jj < tlp1; jj++) cout << U_C2R[l - 1][ii * tlp1 + jj] << ' ';
            cout << endl;
        }
        cout << "L_{+}[" << l << "]" << endl;
        for(int ii = 0; ii < tlp1; ii++) {
            for(int jj = 0; jj < tlp1; jj++) cout << Lplus_mmp[l - 1][ii * tlp1 + jj] << ' ';
            cout << endl;
        }
        cout << "L_{-}[" << l << "]" << endl;
        for(int ii = 0; ii < tlp1; ii++) {
            for(int jj = 0; jj < tlp1; jj++) cout << Lminus_mmp[l - 1][ii * tlp1 + jj] << ' ';
            cout << endl;
        }
        cout << "L_{z}[" << l << "]" << endl;
        for(int ii = 0; ii < tlp1; ii++) {
            for(int jj = 0; jj < tlp1; jj++) cout << Lz_mmp[l - 1][ii * tlp1 + jj] << ' ';
            cout << endl;
        }
        for(int iss = 0; iss < 4; iss++) {
            cout << '<' << iss / 2 << "|LS|" << iss % 2 << ">[" << l << "]" << endl;
            for(int ii = 0; ii < tlp1; ii++) {
                for(int jj = 0; jj < tlp1; jj++) cout << LS_mmp[l - 1][iss * tlp1 * tlp1 + ii * tlp1 + jj] << ' ';
                cout << endl;
            }
        }*/

    }
    return;
}

void socclass::Setup_hsoc_base(const int myprow, const int mypcol, const int nprow, const int npcol) {
    isSetup_hsoc_base = true;
    int irow_lm, jcol_lm, irow_l, jcol_l, tlp1, irow_m, jcol_m, irow_idxl, jcol_idxl, iprow, ipcol; // global
    int irow_lm_loc, jcol_lm_loc; // local
    for(int iatom = 0; iatom < nions; iatom++)
    #pragma omp parallel for private(irow_lm, jcol_lm, irow_l, jcol_l, tlp1, irow_m, jcol_m, irow_idxl, jcol_idxl)
    for(int ij_lm = node_rank; ij_lm < lmmax[iatom] * lmmax[iatom]; ij_lm += node_size) {
        irow_lm = ij_lm % lmmax[iatom];
        jcol_lm = ij_lm / lmmax[iatom];
        irow_l = wvc->atoms[iatom].potc->each_l[irow_lm];
        jcol_l = wvc->atoms[iatom].potc->each_l[jcol_lm];
        irow_m = wvc->atoms[iatom].potc->each_m[irow_lm];
        jcol_m = wvc->atoms[iatom].potc->each_m[jcol_lm];
        irow_idxl = wvc->atoms[iatom].potc->each_idxl[irow_lm];
        jcol_idxl = wvc->atoms[iatom].potc->each_idxl[jcol_lm];
        
        if( (irow_l != jcol_l) || (irow_l == 0) ) {
            for(int iss = 0; iss < 4; iss++) hsoc_base[iatom][ iss * lmmax[iatom] * lmmax[iatom] + ij_lm ] = 0.0;
            continue; // only l = l' > 0 has non-vanished values
        }
        tlp1 = 2 * irow_l + 1;
        for(int iss = 0; iss < 4; iss++)
        hsoc_base[iatom][ iss * lmmax[iatom] * lmmax[iatom] + ij_lm ] // column major
        = LS_mmp[irow_l - 1][ iss * tlp1 * tlp1 +
                              (irow_m + irow_l) * tlp1 + (jcol_m + jcol_l) ]; // row major, irow_l = jcol_l
    }
    MPI_Barrier(world_comm);

    return;
}

void socclass::ReadSocRad(const char *socradcar) {
    ifstream inf(socradcar, ios::in);
    if(!inf.is_open()) { cerr << "SocRadCar file " << socradcar << " not open" << endl; EXIT(1); }
    string line;
    while(getline(inf, line)) {
        if(line.find("L_END") != string::npos) break;
    }
    int iatom = 0, irow_l = 0, jcol_l = 0;
    int iprow, ipcol;
    vector<string> vecstrtmp;
    while(getline(inf, line)) {
        if(line == " ") { // update ion
            iatom++;
            irow_l = 0;
        }
        else if(irow_l < lmax[iatom]) { // loop for current ion
            vecstrtmp = StringSplitByBlank(line);
            for(jcol_l = 0; jcol_l < lmax[iatom]; jcol_l++)
            socrad[iatom][ irow_l * lmax[iatom] + jcol_l ] = stod(vecstrtmp[jcol_l]);
            /*cout << "iatom, irow_l = " << iatom << ", " << irow_l << ": " << flush;
            for(jcol_l = 0; jcol_l < lmax[iatom]; jcol_l++)
            cout << socrad[iatom][ irow_l * lmax[iatom] + jcol_l ] << ' ';
            cout << endl;*/
            irow_l++;
        }
    }

    inf.close();
    return;
}

void socclass::Update_hsoc(const bool is_write_socallcar) {
    int irow_idxl, jcol_idxl;
    for(int iatom = 0; iatom < nions; iatom++)
    for(int iss = 0; iss < 4; iss++)
    #pragma omp parallel for private(irow_idxl, jcol_idxl)
    for(int ij_lm = node_rank - malloc_root; ij_lm < lmmax[iatom] * lmmax[iatom]; ij_lm += share_memory_len) {
        irow_idxl = wvc->atoms[iatom].potc->each_idxl[ ij_lm % lmmax[iatom] ];
        jcol_idxl = wvc->atoms[iatom].potc->each_idxl[ ij_lm / lmmax[iatom] ];
             hsoc[iatom][iss * lmmax[iatom] * lmmax[iatom] + ij_lm] =
        hsoc_base[iatom][iss * lmmax[iatom] * lmmax[iatom] + ij_lm] *
        socrad[iatom][ irow_idxl * lmax[iatom] + jcol_idxl ];
    }

    // write to file
    if(is_write_socallcar) {
        ofstream otf;
        if(is_sub_root) {
            otf.open((dirpath + "socallcar").c_str(), ios::out|ios::binary);
            if(!otf.is_open()) { Cerr << "can't open " << dirpath + "socallcar" << endl; exit(1); }
            for(int iatom = 0; iatom < nions; iatom++)
            for(int iss = 0; iss < 4; iss++)
            otf.write((char*)(hsoc[iatom] + (size_t)iss * lmmax[iatom] * lmmax[iatom]),
                      sizeof(complex<double>) * lmmax[iatom] * lmmax[iatom]);
            otf.close();
        }
        MPI_Barrier(group_comm);
    }
    MPI_Barrier(group_comm);
   
    return;
}

void socclass::SetSocmatDiag(const int in_kpt) {
    const int nspns = wvc->nspns;
    const int nkpts = wvc->nkpts;
    const int nbnds = wvc->nbnds;
    
    for(int is = 0; is < 2; is++)
    for(int ib = 0; ib < nbnds; ib++) {
        socmat[ (is * nbnds + ib) + (is * nbnds + ib) * twonbnds ] = 
        wvc->eigens[min(is, nspns - 1) * nkpts * nbnds + in_kpt * nbnds + ib].energy;
    }
    return;
}

void socclass::GetSpinorCoeff(const int in_kpt, complex<double> *spinorpw) {
    const int nspns = wvc->nspns;
    const int nkpts = wvc->nkpts;
    const int nbnds = wvc->nbnds;
    const int npw = wvc->npw[in_kpt];
    const int npw_loc = wvc->npw_loc[in_kpt];
    for(int is = 0; is < 2; is++) {
        const int iss = min(is, nspns - 1); // for nspns = 1, iss always 0
        for(int iatom = 0; iatom < nions; iatom++) {
            const int nrow_pp_loc = wvc->atoms[iatom].nrow_pp_loc;
            const int ncol_pp_loc = wvc->atoms[iatom].ncol_pp_loc;
            Zgemm("CblasColMajor", "CblasNoTrans", "CblasNoTrans", lmmax[iatom], twonbnds, nbnds,
                  1.0, wvc->atoms[iatom].projphi + iss * lmmax[iatom] * wvc->nkpts * wvc->nbnds
                                                 + lmmax[iatom] * in_kpt * wvc->nbnds, lmmax[iatom],
                       socmat + is * nbnds, twonbnds,
                  0.0, spinor_pp[iatom] + is * lmmax[iatom] * twonbnds, lmmax[iatom]);
            Zgemm("CblasColMajor", "CblasNoTrans", "CblasNoTrans", lmmax[iatom], twonbnds, lmmax[iatom],
                   1.0, wvc->atoms[iatom].potc->Qij_z_full, lmmax[iatom],
                        spinor_pp[iatom] + is * lmmax[iatom] * twonbnds, lmmax[iatom],
                   0.0, Qijpp[iatom] + is * lmmax[iatom] * twonbnds, lmmax[iatom]);
        }
        Zgemm("CblasColMajor", "CblasNoTrans", "CblasNoTrans", npw, twonbnds, nbnds,
              1.0, wvc->coeff_malloc_in_node[in_kpt] + (size_t)(iss * nbnds) * npw, npw,
                   socmat + is * nbnds, twonbnds,
              0.0, spinorpw + is * npw * twonbnds, npw);
    }
    for(int is1 = 0; is1 < 2; is1++) for(int is2 = 0; is2 < 2; is2++)
    for(int ib = 0; ib < twonbnds; ib++) {
        band_spin_polarization[(is1 * 2 + is2) * twonbnds + ib] =
        Zdotc(npw, spinorpw + is1 * npw * twonbnds + ib * npw,
                   spinorpw + is2 * npw * twonbnds + ib * npw);
        for(int iatom = 0; iatom < nions; iatom++) {
            band_spin_polarization[(is1 * 2 + is2) * twonbnds + ib] +=
            Zdotc(lmmax[iatom], spinor_pp[iatom] + is1 * lmmax[iatom] * twonbnds + ib * lmmax[iatom],
                                Qijpp[iatom] + is2 * lmmax[iatom] * twonbnds + ib * lmmax[iatom]);
        }
    }
    return;
}

void socclass::CorrectSocmatReadIn() {
    int N = 0;
    ifstream inf("input", ios::in);
    if(!inf.is_open()) { CERR << "no input file" << endl; EXIT(1); }
    string line;
    vector<string> vecstrtmp;
    while(getline(inf, line)) {
        if(line.empty()) continue;
        vecstrtmp = StringSplitByBlank(line);
        if(vecstrtmp[0] == "spinor2") {
            N = stoi(vecstrtmp[2]);
            if(N > 0) {
                getline(inf, line);
                vecstrtmp = StringSplitByBlank(line);
                for(int i = 0; i < N; i++) read_in_kpts.push_back(stoi(vecstrtmp[i]) - 1);
            }
            break;
        }
    }
    inf.close();
    return;
}

void socclass::CorrectSocmat(const int ikpt) {
    for(int i = 0; i < read_in_kpts.size(); i++) {
        if(ikpt == read_in_kpts[i]) {
            double newen = 0.0;
            for(int ib = 0; ib < wvc->nbnds; ib++) {
                ReCombineAB(socmat + 2 * ib * twonbnds, socmat + (2 * ib + 1) * twonbnds, wvc->nbnds);
                newen = (eigenvals[2 * ib] + eigenvals[2 * ib + 1]) / 2.0;
                eigenvals[2 * ib]     = newen;
                eigenvals[2 * ib + 1] = newen;
            }
            break;
        }
    }
    return;
}

void socclass::WriteSpinorHead(ofstream &wavecar, ofstream &socout) {
    if(is_sub_root) {
        double rdum     = (double)(2 * maxnpw * sizeof(complex<float>));
        double totnspns = 1.0;
        double rtag     = 45200.0;
        memcpy(oneline,     &rdum,     sizeof(double));
        memcpy(oneline + 1, &totnspns, sizeof(double));
        memcpy(oneline + 2, &rtag,     sizeof(double));
        wavecar.write((char*)oneline, sizeof(double) * 2 * maxnpw); // first line, three double numbers
        double totnkpts = (double)wvc->nkpts;
        double totnbnds = (double)twonbnds;
        memcpy(oneline,     &totnkpts,  sizeof(double));
        memcpy(oneline + 1, &totnbnds,  sizeof(double));
        memcpy(oneline + 2, &wvc->emax, sizeof(double));
        for(int i = 0; i < 3; i++) memcpy(oneline + (3 + 3 * i), wvc->a[i], sizeof(double) * 3);
        wavecar.write((char*)oneline, sizeof(double) * 2 * maxnpw); // second line, 12 double numbers
        wavecar.close();
    }
    MPI_Barrier(group_comm);
    
    return;
}

void socclass::WriteSpinor(const int in_kpt, const complex<double> *spinorpw, 
                           ofstream &wavecar, ofstream &socout, const char *suffix) {
    // setup totweight, once is enouth
    if(!isSettotweight) {
        totweight = 0.0;
        for(int is = 0; is < wvc->nspns; is++) for(int ib = 0; ib < wvc->nbnds; ib ++) 
        totweight += wvc->eigens[is * wvc->nkpts * wvc->nbnds + ib].weight;
        totweight *= (3 - wvc->nspns); // if nspns = 1, should times 2
        isSettotweight = true;
    }
    
    const int npw     = wvc->npw[in_kpt];
    const int nspns   = wvc->nspns;
    const int nkpts   = wvc->nkpts;
    const int nbnds   = wvc->nbnds;
    
    // wavecar
    wavecar.open((dirpath + "wavecar" + suffix).c_str(), ios::out|ios::binary|ios::app);
    double dbtmp;
    dbtmp = 2 * (double)npw; // 2 x npw
    memcpy(oneline,     &dbtmp, sizeof(double));
    memcpy(oneline + 1, wvc->kptvecs[in_kpt], sizeof(double) * 3); // kptvecs[in_kpt][0..2]
    for(int ib = 0; ib < twonbnds; ib++) {
         memcpy(oneline + 4 + ib * 3,     eigenvals + ib, sizeof(double)); // energy
         dbtmp = 0.0;
         memcpy(oneline + 4 + ib * 3 + 1, &dbtmp, sizeof(double));         // useless zero
         dbtmp = (ib + 1 < totweight + 0.1 ? 1.0 : 0.0);
         memcpy(oneline + 4 + ib * 3 + 2, &dbtmp, sizeof(double));         // weight, ONLY right for semiconductor
    }
    wavecar.write((char*)oneline, sizeof(double) * 2 * maxnpw); // (4 + 3 x nbnds) double numbers
    for(int ib = 0; ib < twonbnds; ib++) {
        for(int is = 0; is < 2; is++) {
            #pragma omp parallel for
            for(int ipw = 0; ipw < npw; ipw++)
            oneline[is * npw + ipw] = spinorpw[(size_t)npw * (size_t)twonbnds * is +
                                               ipw + (size_t)npw * ib];
        }
        wavecar.write((char*)oneline, sizeof(complex<float>) * 2 * maxnpw); // 2 x npw complex<float> numbers
    }
    wavecar.close();

    // socout
    socout.open((dirpath +  "socout"  + suffix).c_str(), ios::out|ios::app);
    double ienergy;
    socout << "kpoint = " << in_kpt + 1 << " :     " << flush;
    for(int i = 0; i < 3; i++)
    socout << fixed << setprecision(4) << wvc->kptvecs[in_kpt][i] << "  "; socout << endl;
    socout << "band No.   KS energies    Spinor energies     occupation     spin polarization x,y,z" << endl;
    for(int ib = 0; ib < twonbnds; ib++) {
        socout << std::right << setw(4) << setfill(' ') << ib + 1 << flush;
        ienergy = wvc->eigens[min(ib % 2, nspns - 1) * nkpts * nbnds + in_kpt * nbnds + ib / 2].energy;
        socout << std::right << setw(16) << setfill(' ') << fixed << setprecision(4) 
               << ienergy << flush;
        socout << std::right << setw(16) << setfill(' ') << fixed << setprecision(4) 
               << eigenvals[ib] << flush;
        socout << std::right << setw(16) << setfill(' ') << fixed << setprecision(4) 
               << (ib + 1 < totweight + 0.1 ? 1.0 : 0.0) << flush;
        socout << std::right << setw(14) << setfill(' ') << fixed << setprecision(4) 
               << real(band_spin_polarization[twonbnds + ib] + band_spin_polarization[2 * twonbnds + ib])
               << std::right << setw(10) << setfill(' ') << fixed << setprecision(4) 
               << imag(band_spin_polarization[twonbnds + ib] - band_spin_polarization[2 * twonbnds + ib])
               << std::right << setw(10) << setfill(' ') << fixed << setprecision(4) 
               << real(band_spin_polarization[ib] - band_spin_polarization[3 * twonbnds + ib]) << endl;
    }
    if(in_kpt < nkpts - 1) socout << endl; // add an empty line
    socout.close();

    return;
}

bool socclass::MakeSpinor(const char *idirpath, const char *suffix) {
    dirpath = idirpath;
    if(CheckSpinorWavecar()) return false; // spinor file "wavecar" exists, not make a new one
    double tstart = omp_get_wtime();
    ReadSocRad((dirpath + "SocRadCar").c_str());
    Update_hsoc();
    complex<double> *tmpmat = NULL;
    complex<double> *spinorpw = NULL;
    ofstream wavecar, socout;
    if(is_sub_root) {
        wavecar.open((dirpath + "wavecar" + suffix).c_str(), ios::out|ios::binary);
        if(!wavecar.is_open()) { cerr << "ERROR: can't open " << dirpath + "wavecar" + suffix << endl; exit(1); }
        socout.open((dirpath +  "socout"  + suffix).c_str(), ios::out);
        if(!socout.is_open())  { cerr << "ERROR: can't open " << dirpath + "socout"  + suffix << endl; exit(1); }
        socout.close();
    }
    WriteSpinorHead(wavecar, socout);
    GetProjPhi();
    socmat = new complex<double>[ twonbnds * twonbnds ]();
    for(int iatom = 0; iatom < nions; iatom++) {
        Qijpp[iatom]     = new complex<double>[ 2 * lmmax[iatom] * twonbnds ]();
        spinor_pp[iatom] = new complex<double>[ 2 * lmmax[iatom] * twonbnds ]();
    }
    for(int ik = sub_rank;
            ik < ( wvc->nkpts % sub_size == 0 ? wvc->nkpts : (wvc->nkpts / sub_size + 1) * sub_size );
            ik += sub_size) { // the trick here avoids barrier stuck
        if(ik < wvc->nkpts) {
            fill_n(socmat, twonbnds * twonbnds, complex<double>(0.0, 0.0));
            SetSocmatDiag(ik);
            for(int iatom = 0; iatom < nions; iatom++) {
                tmpmat = new complex<double>[lmmax[iatom] * wvc->nbnds];
                for(int iss = 0; iss < 4; iss++) {
                    const int is1 = min(iss / 2, wvc->nspns - 1); // for nspns = 1, is1(2) always 0
                    const int is2 = min(iss % 2, wvc->nspns - 1); // for nspns = 2, is1(2) represents correct spin channel
                    Zgemm("CblasColMajor", "CblasNoTrans", "CblasNoTrans",
                          lmmax[iatom], wvc->nbnds, lmmax[iatom],
                          1.0, hsoc[iatom] + iss * lmmax[iatom] * lmmax[iatom], lmmax[iatom],
                               wvc->atoms[iatom].projphi + is2 * lmmax[iatom] * wvc->nkpts * wvc->nbnds
                                                         + lmmax[iatom] * ik * wvc->nbnds, lmmax[iatom],
                          0.0, tmpmat, lmmax[iatom]);
                    Zgemm("CblasColMajor", "CblasConjTrans", "CblasNoTrans",
                          wvc->nbnds, wvc->nbnds, lmmax[iatom],
                          1.0, wvc->atoms[iatom].projphi + is1 * lmmax[iatom] * wvc->nkpts * wvc->nbnds
                                                         + lmmax[iatom] * ik * wvc->nbnds, lmmax[iatom],
                               tmpmat, lmmax[iatom],
                          1.0, socmat + ( iss / 2 * wvc->nbnds + iss % 2 * wvc->nbnds * twonbnds), twonbnds);
                                       /* ----- row idx ------   ----- col idx ------ */
                }
                delete[] tmpmat;
            }
            
            Zheev(LAPACK_COL_MAJOR, 'V', 'U', twonbnds, socmat, twonbnds, eigenvals);
            CorrectSocmat(ik);

            /* 
             * C_{G, i} x S_{i, n}: C_{G, i}-pw coefficients of ith state
             *                      S_{i, n}-nth eigenvec
             * the first  half C_{NG, :nbnds} x S_{:nbnds, twonbnds} should be the spin up   part of cooresponding spinor states
             * the second half C_{NG, nbnds:} x S_{nbnds:, twonbnds} should be the spin down part of cooresponding spinor states
             */
            spinorpw = new complex<double>[2 * wvc->npw[ik] * twonbnds]();
            GetSpinorCoeff(ik, spinorpw);
        }
        MPI_Barrier(group_comm); // IMPORTANT: wait work in every rank has done
        for(int irk = 0; irk < sub_size; irk++) { // ordered loop for ordered write to file
            if(sub_rank == irk && ik < wvc->nkpts) {
                WriteSpinor(ik, spinorpw, wavecar, socout, suffix);
            }
            MPI_Barrier(group_comm);
        }
        if(ik < wvc->nkpts) delete[] spinorpw;
    }
    delete[] socmat;
    for(int iatom = 0; iatom < nions; iatom++) {
        delete[] Qijpp[iatom];
        delete[] spinor_pp[iatom];
    }
    if(is_sub_root) { wavecar.close(); socout.close(); }
    double tend = omp_get_wtime();
    Cout << "Make spinor file \"" << dirpath << "wavecar\" " << setw(4) << setfill(' ') << (int)(tend - tstart) << " s" << endl; 
    
    MPI_Barrier(group_comm);
    return true;
}

socclass::socclass(waveclass *wvc_aux):wvc(wvc_aux) {}
void socclass::Initial() {
    isInitial = true;
    malloc_root      = wvc->malloc_root;
    malloc_end       = wvc->malloc_end;
    share_memory_len = wvc->share_memory_len;
    
    nbnds_loc_r = Numroc(wvc->nbnds, MB_ROW, myprow_group, nprow_group);
    nbnds_loc_c = Numroc(wvc->nbnds, NB_COL, mypcol_group, npcol_group);
    twonbnds = 2 * wvc->nbnds;
    twonbnds_loc_r = Numroc(twonbnds, MB_ROW, myprow_group, nprow_group);
    twonbnds_loc_c = Numroc(twonbnds, NB_COL, mypcol_group, npcol_group);
    
    nions = wvc->numatoms;
    lmmax        = new int[nions];
    lmax         = new int[nions];
    lmmax_loc_r  = new int[nions];
    lmmax_loc_c  = new int[nions];
    lmmax2_loc   = new int[nions];
    
    hsoc_base    = new complex<double>*[nions];
    hsoc         = new complex<double>*[nions];
    socrad       = new double*[nions];
    spinor_pp    = new complex<double>*[nions];
    Qijpp        = new complex<double>*[nions];
    size_t tottmp = 0;
    for(int iatom = 0; iatom < nions; iatom++) {
        lmmax[iatom] = wvc->atoms[iatom].potc->lmmax;
        lmax[iatom]  = wvc->atoms[iatom].potc->projL.size();
        lmmax_loc_r[iatom] = wvc->atoms[iatom].potc->lmmax_loc_r;
        lmmax_loc_c[iatom] = wvc->atoms[iatom].potc->lmmax_loc_c;
        lmmax2_loc[iatom]  = wvc->atoms[iatom].potc->lmmax2_loc;
        socrad[iatom]       = new double[ lmax[iatom] * lmax[iatom] ]();
        tottmp += 4 * lmmax[iatom] * lmmax[iatom];
    }
    MpiWindowShareMemoryInitial(tottmp, hsoc_baseall, local_hsoc_base_node, window_hsoc_base);
    MpiWindowShareMemoryInitial(tottmp, hsocall,      local_hsoc,           window_hsoc, malloc_root);
    size_t sumtmp = 0;
    for(int iatom = 0; iatom < nions; iatom++) {
        hsoc_base[iatom] = hsoc_baseall + sumtmp;
        hsoc[iatom] = hsocall + sumtmp;
        sumtmp += 4 * lmmax[iatom] * lmmax[iatom];
    }

    eigenvals = new double[twonbnds]();
    band_spin_polarization = new complex<double>[4 * twonbnds];
    
    maxnpw = wvc->npw[0];
    for(int ik = 1; ik < wvc->nkpts; ik++) maxnpw = max(wvc->npw[ik], maxnpw);
    maxnpw = max(2 + 3 * wvc->nbnds, maxnpw);
    oneline = new complex<float>[2 * maxnpw]();

    SetupLS();
    Setup_hsoc_base();
    CorrectSocmatReadIn();
    
    return;
}

socclass::~socclass() {
    if(isSetupLS) for(int l = 1; l <= 4; l++) { 
        delete[] Lplus_mmp[l - 1];
        delete[] Lminus_mmp[l - 1];
        delete[] Lz_mmp[l - 1];
        delete[] U_C2R[l - 1];
        delete[] LS_mmp[l - 1];
    }
    if(isInitial) {
        delete[] lmmax; delete[] lmax; delete[] lmmax_loc_r; delete[] lmmax_loc_c; delete[] lmmax2_loc;
        for(int iatom = 0; iatom < nions; iatom++) { 
            delete[] socrad[iatom];
        }
        MPI_Win_free(&window_hsoc_base);
        MPI_Win_free(&window_hsoc);
        delete[] hsoc_base; delete[] hsoc; delete[] socrad; 
        delete[] spinor_pp; delete[] Qijpp;
        delete[] eigenvals;
        delete[] band_spin_polarization;
        delete[] oneline;
    }
}
