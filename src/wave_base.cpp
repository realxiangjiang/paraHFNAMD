#include "wave_base.h"

void atomclass::LoadPosition(string posstr, double *a[]) {
    vector<string> vecstrtmp = StringSplitByBlank(posstr);
    for(int i = 0; i < 3; i++) posfrac[i] = stod(vecstrtmp[i]);
    double dbtmp;
    for(int j = 0; j < 3; j++) { // loop for x y z
        dbtmp = 0.0;
        for(int i = 0; i < 3; i++) dbtmp += posfrac[i] * a[i][j];
        poscart[j] = dbtmp;
    }
    return;
}

void atomclass::Getcrexp(const int nkpts, double **kptvecs, const int *npw, int ng[], int **gidx,
                         int rk, int sz, MPI_Comm &comm, const int addkptv) {
    if(!isGetcrexp) {
        size_t totnpw = 0;
        for(int ik = 0; ik < nkpts; ik++) totnpw += npw[ik];
        MpiWindowShareMemoryInitial(totnpw, crexp_forall, local_crexp, window_crexp, malloc_root);
        crexp = new complex<double>*[nkpts];
        size_t sumnpw = 0;
        for(int ik = 0; ik < nkpts; ik++) {
            crexp[ik] = crexp_forall + sumnpw;
            sumnpw += npw[ik];
        }
        this->nkpts = nkpts;
        isGetcrexp = true;
    }

    int gx, gy, gz;
    for(int ik = 0; ik < nkpts; ik++) {
        for(int ig = node_rank - malloc_root; ig < npw[ik]; ig += share_memory_len) {
            IdxNat1toSym3(gidx[ik][ig], gx, gy, gz, ng[0], ng[1], ng[2]);
            crexp[ik][ig] = exp( 2.0 * M_PI * iu_d * ( (gx + kptvecs[ik][0] * addkptv) * posfrac[0] +
                                                       (gy + kptvecs[ik][1] * addkptv) * posfrac[1] +
                                                       (gz + kptvecs[ik][2] * addkptv) * posfrac[2] ) );
        }
    }

    MPI_Barrier(comm);
    return;
}

void atomclass::Getcrexp_q(const int nqpts, double **qptvecs, const int *npw,
                           int **gidx, const int ng_bse[3], const int addqptv) {
    if(!isGetcrexp_q) {
        size_t totnpw = 0;
        for(int iqpt = 0; iqpt < nqpts; iqpt++) totnpw += npw[iqpt];
        MpiWindowShareMemoryInitial(totnpw, crexp_q_forall, local_crexp_q, window_crexp_q, malloc_root);
        crexp_q = new complex<double>*[nqpts];
        size_t sumnpw = 0;
        for(int iqpt = 0; iqpt < nqpts; iqpt++) {
            crexp_q[iqpt] = crexp_q_forall + sumnpw;
            sumnpw += npw[iqpt];
        }
        this->nqpts = nqpts;
        isGetcrexp_q = true;
    }
    
    int ig_glb, gx, gy, gz;
    for(int iqpt = 0; iqpt < nqpts; iqpt++) {
        for(int ig = node_rank - malloc_root; ig < npw[iqpt]; ig += share_memory_len) {
            IdxNat1toSym3(gidx[iqpt][ig], gx, gy, gz, ng_bse[0], ng_bse[1], ng_bse[2]);
            crexp_q[iqpt][ig] = exp( 2.0 * M_PI * iu_d * 
                                   ( (gx + qptvecs[iqpt][0] * addqptv) * posfrac[0] +
                                     (gy + qptvecs[iqpt][1] * addqptv) * posfrac[1] +
                                     (gz + qptvecs[iqpt][2] * addqptv) * posfrac[2] ) );
        }
    }

    MPI_Barrier(group_comm);
    return;
}

void atomclass::Getprojphi(const int nspns, const int nkpts, const int nbnds, vector<int> &kpoints, 
                           complex<double> **coeffs, const int *npw, const double volume, const int is_spinor) {
    // coeffs: nspns x nkpts x nbnds *coeff with size of npw[ik]
    if(!isMallocProjphi) {
        nrow_pp_loc = Numroc(potc->lmmax,   MB_ROW, myprow_group, nprow_group);
        ncol_pp_loc = Numroc(nkpts * nbnds, NB_COL, mypcol_group, npcol_group);
        MpiWindowShareMemoryInitial(nspns * (1 + is_spinor) * potc->lmmax * (nkpts * nbnds),
                                    projphi, local_projphi, window_projphi, malloc_root);
        if(nrow_pp_loc * ncol_pp_loc) projphi_bc = new complex<double>[nspns * (1 + is_spinor) * nrow_pp_loc * ncol_pp_loc]();
        isMallocProjphi = true;
    }
    complex<double> *proj = NULL;
    for(int is = 0; is < nspns; is++) 
    for(int ilm = 0; ilm < potc->lmmax; ilm++)
    for(int ik = 0; ik < nkpts; ik++) {
        int ikpt = kpoints[ik];
        proj = new complex<double>[npw[ik]]();
        Zvmul(npw[ik], potc->qProjsKpt[ikpt * potc->lmmax + ilm], crexp[ik], proj);
        for(int ib = node_rank - malloc_root; ib < nbnds; ib += share_memory_len) {                                               
            projphi[ is * potc->lmmax * (nkpts * nbnds) + (ilm + (ik * nbnds + ib) * potc->lmmax) ] =
            1.0 / sqrt(volume) * Zdot(npw[ik], proj, coeffs[is * nkpts * nbnds + ik * nbnds + ib]);
            if(is_spinor)
            projphi[      potc->lmmax * (nkpts * nbnds) + (ilm + (ik * nbnds + ib) * potc->lmmax) ] =
            1.0 / sqrt(volume) * Zdot(npw[ik], proj, coeffs[is * nkpts * nbnds + ik * nbnds + ib] + npw[ik]);
        }
        MPI_Barrier(group_comm);
        delete[] proj;
    }
    MPI_Barrier(group_comm);

    for(int is = 0; is < nspns; is++) 
    for(int isigma = 0; isigma < 1 + is_spinor; isigma++) {
        Blacs_MatrixZScatter(potc->lmmax, nkpts * nbnds,
                             projphi 
                             + (size_t)(is * (1 + is_spinor) + isigma) * potc->lmmax * (nkpts * nbnds), potc->lmmax,
                             projphi_bc 
                             + (size_t)(is * (1 + is_spinor) + isigma) * nrow_pp_loc * ncol_pp_loc, nrow_pp_loc);
    }

    MPI_Barrier(group_comm);
    return;
}

void atomclass::Writeprojphi(const char *normcar, const int nspns, const int nkpts, const int nbnds, const int is_spinor) {
    if(is_sub_root) {
        ofstream otf(normcar, ios::out|ios::binary|ios::app);
        if(!otf.is_open()) { cerr << "can't open " << normcar << endl; exit(1); }
        otf.write((char*)projphi, nspns * (1 + is_spinor)  * (nkpts * nbnds) * potc->lmmax * sizeof(complex<double>));
        otf.close();
    }
    MPI_Barrier(group_comm);
    return;
}

atomclass::atomclass() {}
void atomclass::Initial(string posstr, double *a[], pawpotclass *input_potc,
                        const int malloc_root, const int malloc_end, const int share_memory_len) {
    potc = input_potc;
    element = potc->element;
    LoadPosition(posstr, a);
    Aij[0] = potc->Qij_z;
    for(int s = 0; s < 3; s++) Aij[1 + s] = potc->Gij_z + s * potc->lmmax2_loc;
    this->malloc_root      = malloc_root;
    this->malloc_end       = malloc_end;
    this->share_memory_len = share_memory_len;
    return;
}
atomclass::~atomclass() {
    if(isGetcrexp) {
        MPI_Win_free(&(window_crexp));
        delete[] crexp;
    }
    if(isGetcrexp_q) {
        MPI_Win_free(&(window_crexp_q));
        delete[] crexp_q;
    }
    if(isMallocProjphi) {
        MPI_Win_free(&window_projphi);
        if(nrow_pp_loc * ncol_pp_loc) delete[] projphi_bc;
    }
}

void VecCross3d(double *vec1, double *vec2, double *res) {
    int j, k;
    #pragma omp parallel for private(j, k)
    for(int i = 0; i < 3; i++){
        j = (i + 1) % 3;
        k = (i + 2) % 3;
        res[i] = vec1[j] * vec2[k] - vec1[k] * vec2[j];
    }
    return;
}

double VecDot3d(double *vec1, double *vec2) {
    double res = 0.0;
    for (int i = 0; i < 3; i++) res += vec1[i] * vec2[i];
    return res;
}

void waveclass::eigenclass::BandReorder(int *wvorder, int info) { // modify band #
    bnd = wvorder[bnd];
    return;
}

void waveclass::eigenclass::OnlyGetEn(const char *wavecar) { // get eigen energy and weight of current eigenstate
    ifstream inf(wavecar, ios::in|ios::binary);
    if(!inf.is_open()) { CERR << "WAVECAR file " << wavecar << " not open" << endl; EXIT(1); }
    double *dbtmp = new double[3];
    inf.read((char*)dbtmp, sizeof(double)); // read the first line, total 3 values
    long irdum  = (long)dbtmp[0]; // line length

    double inpw;
    inf.seekg(irdum * (2 + (spn * tnkpts + kpt) * (1 + tnbnds)) + 
              sizeof(double) * (4 + 3 * bnd), ios::beg);
    inf.read((char*)dbtmp, sizeof(double) * 3);
    inf.close();
    energy = dbtmp[0]; weight = dbtmp[2]; // in VASP, maximum of weight is 1.0 wheather ispin = 1 or 2
    delete[] dbtmp;
    return;
}

void waveclass::eigenclass::Getcoeff(const char *wavecar, int rk, int sz, MPI_Comm &comm, 
                                     double *energies, double *weights,
                                     const int isgamma) { // alse get energy and weight
    ifstream inf(wavecar, ios::in|ios::binary);
    if(!inf.is_open()) { CERR << "WAVECAR file " << wavecar << " not open" << endl; EXIT(1); }
    double *dbtmp = new double[3];
    inf.read((char*)dbtmp, sizeof(double)); // read the first line, total 3 values
    size_t irdum  = (size_t)dbtmp[0]; // line length

    inf.seekg(irdum * (2 + (spn * tnkpts + kpt) * (1 + tnbnds)), ios::beg);
    inf.read((char*)dbtmp, sizeof(double));
    int inpw = round(dbtmp[0]);
    complex<float> *coeff_qs = new complex<float>[inpw](); // the subscript "_qs" is initial from vasp

    // energy and weight
    inf.seekg(sizeof(double) * (3 + 3 * bnd), ios::cur);
    inf.read((char*)dbtmp, sizeof(double) * 3);
    energies[istate] = dbtmp[0]; weights[istate] = dbtmp[2];
    delete[] dbtmp;
    
    // coeff
    inf.seekg(irdum * (2 + (spn * tnkpts + kpt) * (1 + tnbnds) + (1 + bnd)), ios::beg);
    inf.read((char*)coeff_qs, sizeof(complex<float>) * inpw);
    if(isgamma) {
        coeff[0] = coeff_qs[0];
        #pragma omp parallel for
        for(int ig = 1; ig < inpw; ig++) {
            coeff[ig] = 1.0 / sqrt(2.0) * coeff_qs[ig];
            coeff[ig - 1 + inpw] = 1.0 / sqrt(2.0) * conj(coeff_qs[ig]);
        }
    }
    else {
        #pragma omp parallel for
        for(int ig = 0; ig < inpw; ig++) {
            coeff[ig] = coeff_qs[ig];
        }
    }
    inf.close();
    delete[] coeff_qs;

    return;
}

void waveclass::eigenclass::AddPhase(complex<double> newphase, int info) { // multiple another phase factor
    Zscal(npw, newphase, coeff, 1);
    return;
}

void waveclass::eigenclass::Getrealc() {
    for(int isigma = 0; isigma < (1 + spinor); isigma++) {
        Compacted2Cubiod(npw, gidx, ng,  coeff + isigma * npw,
                                    ngf, realc + isigma * ngftot);
        Zfft_3d(realc + isigma * ngftot, ngf, FFTW_BACKWARD);
    }
    return;
}

waveclass::eigenclass::eigenclass() {}
void waveclass::eigenclass::Initial(waveclass *wvc, const int ie, const int malcoeff) {
    spinor = wvc->spinor;
    IdxNat1toNat3(ie, ispn, ikpt, ibnd, wvc->nspns, wvc->nkpts, wvc->nbnds);
    istate = ie;
    spn = wvc->spins[ispn];       tnspns = wvc->totnspns; 
    kpt = wvc->kpoints[ikpt];     tnkpts = wvc->totnkpts; 
    bnd = wvc->bands[ispn][ibnd]; tnbnds = wvc->totnbnds; 
    for(int s = 0; s < 3; s++) {
        kptvec[s] = wvc->kptvecs[ikpt][s];
        ng[s]     = wvc->ng[s];
        ngf[s]    = wvc->ngf[s];
    }
    ngftot = wvc->ngftot;
    ngf_loc_n0 = wvc->ngf_loc_n0; ngf_loc_0_start = wvc->ngf_loc_0_start;
    all_ngf_loc_n0 = wvc->all_ngf_loc_n0; all_ngf_loc_0_start = wvc->all_ngf_loc_0_start;
    tot_loc_ngf = wvc->tot_loc_ngf;
    gidx = wvc->gidx[ikpt]; gidxRev = wvc->gidxRev[ikpt];
    npw = wvc->npw[ikpt];
    npw_loc = wvc->npw_loc[ikpt];
    if(sub_size > 1) {
        copy(wvc->locbeg[ikpt].begin(), wvc->locbeg[ikpt].end(), back_inserter(locbeg));
        copy(wvc->glbbeg[ikpt].begin(), wvc->glbbeg[ikpt].end(), back_inserter(glbbeg));
        copy(wvc->bcklen[ikpt].begin(), wvc->bcklen[ikpt].end(), back_inserter(bcklen));
    }
    coeff = NULL; realc = NULL;
    extraphase = 1.0;
   
    // need real-space coefficients, current when calculating bse
    /* realc = new complex<double>[max(tot_loc_ngf * (1 + spinor), 1)](); */
}

waveclass::eigenclass::~eigenclass() {
}

void waveclass::GetRecpLatt() {
    double *dbtmp = new double[3];
    VecCross3d(a[0], a[1], dbtmp);
    volume = VecDot3d(dbtmp, a[2]);
    delete[] dbtmp;
    int j, k;
    for(int i = 0; i < 3; i++) {
        b[i] = new double[3];
        j = (i + 1) % 3;
        k = (i + 2) % 3;
        VecCross3d(a[j], a[k], b[i]);
        for(int s = 0; s < 3; s++) b[i][s] = b[i][s] / volume;
    }

    return;
}

void waveclass::Getng() {
    #pragma omp parallel for
    for(int i = 0; i < 3; i++) {
        double dbtmp = sqrt(emax / rytoev) / (2.0 * M_PI / (sqrt(VecDot3d(a[i], a[i])) / autoa));
        ng[i] = (int)(2 * dbtmp + 5);
        ngf[i] = (int)(2 * dbtmp) * 2;
    }
    ngtot = ng[0] * ng[1] * ng[2];
    FFTch3(ngf);
    ngftot = ngf[0] * ngf[1] * ngf[2];
    Pzfft3SetMallocSize(ngf[0], ngf[1], ngf[2], ngf_loc_n0, ngf_loc_0_start, tot_loc_ngf, col_comm);
    MPI_Barrier(col_comm);
    all_ngf_loc_n0 = new int[col_size];
    all_ngf_loc_0_start = new int[col_size];
    MPI_Allgather(&ngf_loc_n0,      1, MPI_INT, all_ngf_loc_n0,      1, MPI_INT, col_comm);
    MPI_Allgather(&ngf_loc_0_start, 1, MPI_INT, all_ngf_loc_0_start, 1, MPI_INT, col_comm);
    
    /*for(int irk = 0; irk < world_sz; irk++) {
        if(world_rk == irk) {
            cout << world_rk << ' ' << col_rank << '/' << col_size << ": " << ngf_loc_n0 << '/' << ngf[0]
                 << ' ' << ngf_loc_0_start << ' ' << tot_loc_ngf << '/' << ngf_loc_n0 * ngf[1] * ngf[2] << endl;
        }
        MPI_Barrier(world_comm);
    }*/

    return;
}

void waveclass::DetermineShareMemoryRanks() {
    int malloc_rt_nodecolor;
    int malloc_end_nodecolor;
    // node_rank from malloc_root to malloc end should share meomory malloced in malloc_root
    if(is_sub_root) {
        if(!is_node_root) {
            malloc_root = node_rank;
            malloc_rt_nodecolor = node_color;
        }
        else {
            malloc_root = node_root;
            malloc_rt_nodecolor = node_color;
        }
        // the result of above if-else is:
        // malloc_root = node_rank, malloc_rt_nodecolor = node_color
    }
    MPI_Bcast(&malloc_root,         1, MPI_INT, sub_root, group_comm);
    MPI_Bcast(&malloc_rt_nodecolor, 1, MPI_INT, sub_root, group_comm);
    if(node_color != malloc_rt_nodecolor) malloc_root = node_root; // exclude out-node processes

    MpiWindowShareMemoryInitial(sub_size, all_malloc_root, local_allroot, window_allroot, malloc_root);
    MpiWindowShareMemoryInitial(sub_size, all_mal_nodeclr, local_allnclr, window_allnclr, malloc_root);
    MPI_Allgather(&malloc_root, 1, MPI_INT, all_malloc_root, 1, MPI_INT, group_comm);
    MPI_Allgather(&node_color,  1, MPI_INT, all_mal_nodeclr, 1, MPI_INT, group_comm);

    if(sub_rank == sub_size - 1) {
        malloc_end = node_rank;
        malloc_end_nodecolor = node_color;
    }
    MPI_Bcast(&malloc_end,           1, MPI_INT, sub_size - 1, group_comm);
    MPI_Bcast(&malloc_end_nodecolor, 1, MPI_INT, sub_size - 1, group_comm);
    if(node_color != malloc_end_nodecolor) malloc_end = node_size - 1; // exclude out-node processes

    share_memory_len = malloc_end - malloc_root + 1;
    
    /*for(int irk = 0; irk < world_sz; irk++) {
        if(world_rk == irk) {
            cout << "world/sub/node/malloc_root rank/malloc_end_rank = "
                 << world_rk << ' ' << sub_rank << ' ' << node_rank << ' '
                             << malloc_root << ' ' << malloc_end << "  " << flush;
            cout << "rt ~ ";   for(int i = 0; i < sub_size; i++) cout << all_malloc_root[i] << ' ';
            cout << " clr ~ "; for(int i = 0; i < sub_size; i++) cout << all_mal_nodeclr[i] << ' ';
            cout << endl;
        }
        MPI_Barrier(world_comm);
    } while(1); */

    MPI_Barrier(world_comm);
    return;
}

void waveclass::IniAtoms(const char *posfile, pawpotclass *pawpots) {
    ifstream inf(posfile, ios::in);
    if(!inf.is_open()) { COUT << "POSITION file " << posfile << " doesn't find, please check." << endl; EXIT(1); }
    isIniAtoms = true;
    if(dftcode == "vasp") {
        string        posfilestr = WholeFile2String(inf);
        istringstream posfiless(posfilestr);
        string line;
        vector<string> vecstrtmp;
        for(int i = 0; i < 7; i++) getline(posfiless, line); // locate to line for # of each element
        vecstrtmp = StringSplitByBlank(line);
        vector<int> num_of_each_element;
        for(int i = 0; i < vecstrtmp.size(); i++) num_of_each_element.push_back(stoi(vecstrtmp[i]));
        numatoms = accumulate(num_of_each_element.begin(), num_of_each_element.end(), 0);
        atoms = new atomclass[numatoms];
        getline(posfiless, line); // skip one line
        if(line.find("Selective dynamics") != string::npos) getline(posfiless, line); // skip extra one line
        int sumtmp = 0; // temporary counting
        for(int ielem = 0; ielem < num_of_each_element.size(); ielem++) { // loop for all elements
            int num_of_this_element = num_of_each_element[ielem];
            for(int iatom = 0; iatom < num_of_this_element; iatom++) { // loop for current element
                getline(posfiless, line);
                atoms[sumtmp + iatom].Initial(line, a, pawpots + ielem, malloc_root, malloc_end, share_memory_len);
            }
            sumtmp += num_of_this_element;
        }
    }
    inf.close();
    return;
}

void waveclass::AtomsRefresh(const char *posfile) { // refresh positions and exp(i(G+k).R)
    ifstream inf(posfile, ios::in);
    if(!inf.is_open()) { COUT << "POSITION file " << posfile << " doesn't find, please check." << endl; EXIT(1); }
    if(dftcode == "vasp") {
        string        posfilestr = WholeFile2String(inf);
        istringstream posfiless(posfilestr);
        string line;
        vector<string> vecstrtmp;
        for(int i = 0; i < 8; i++) getline(posfiless, line); // skip lines to atom positions
        if(line.find("Selective dynamics") != string::npos) getline(posfiless, line); // skip extra one line
        for(int iatom = 0; iatom < numatoms; iatom++) {
            getline(posfiless, line);
            atoms[iatom].LoadPosition(line, a);
            atoms[iatom].Getcrexp(nkpts, kptvecs, npw, ng, gidx, sub_rank, sub_size, group_comm);
        }
    }
    inf.close();
    return;
}

void waveclass::StatesRead(const vector<int> inSpins, const vector<int> inKpoints,
                           const vector<int> inBmin, const vector<int> inBmax, 
                           const vector<int> inExclude) {
    isStatesRead = true;
    copy(inSpins.begin(), inSpins.end(), back_inserter(spins));
    spins.erase( unique(spins.begin(), spins.end()), spins.end() );
    for(int ii = 0; ii < inKpoints.size(); ii++) kpoints.push_back(inKpoints[ii] - 1);

    const int exlnum = inExclude[0];
    bands = new vector<int>[inSpins.size()];
    for(int ii = 0; ii < inSpins.size(); ii++) {
        for(int ibnd = inBmin[ii]; ibnd < inBmax[ii] + 1; ibnd++) {
            if(find(inExclude.begin() + 1 +  ii      * exlnum, 
                    inExclude.begin() + 1 + (ii + 1) * exlnum, ibnd)
               ==   inExclude.begin() + 1 + (ii + 1) * exlnum) bands[ii].push_back(ibnd - 1); 
        }
    }

    nspns = spins.size(); nkpts = kpoints.size(); nbnds = bands[0].size();
    nstates = nspns * nkpts * nbnds;

    if(carrier == "electron")  { dimC = nbnds; dimV = 0; }
    else if(carrier == "hole") { dimC = 0;     dimV = nbnds; }
    
    return;
}

void waveclass::StatesRead(const vector<int> inSpins, const vector<int> inKpoints,
                           const vector<int> inCmin, const vector<int> inCmax,
                           const vector<int> inVmin, const vector<int> inVmax, 
                           const vector<int> inExclude) {
    isStatesRead = true;
    copy(inSpins.begin(), inSpins.end(), back_inserter(spins));
    spins.erase( unique(spins.begin(), spins.end()), spins.end() );
    for(int ii = 0; ii < inKpoints.size(); ii++) kpoints.push_back(inKpoints[ii] - 1);
    
    const int exlnum = inExclude[0];
    bands = new vector<int>[inSpins.size()];
    vector<int> dimCtmp, dimVtmp;
    for(int ii = 0; ii < inSpins.size(); ii++) {
        for(int icb = inCmin[ii]; icb < inCmax[ii] + 1; icb++) {
            if(find(inExclude.begin() + 1 +  ii      * exlnum, 
                    inExclude.begin() + 1 + (ii + 1) * exlnum, icb)
               ==   inExclude.begin() + 1 + (ii + 1) * exlnum) bands[ii].push_back(icb - 1);
        }
        dimCtmp.push_back(bands[ii].size());
        for(int jvb = inVmax[ii]; jvb > inVmin[ii] - 1; jvb--) {
            if(find(inExclude.begin() + 1 +  ii      * exlnum, 
                    inExclude.begin() + 1 + (ii + 1) * exlnum, jvb)
               ==   inExclude.begin() + 1 + (ii + 1) * exlnum) bands[ii].push_back(jvb - 1);
        }
        dimVtmp.push_back(bands[ii].size() - dimCtmp[ii]);
    }
    if(inSpins.size() == 2) {
        if(dimCtmp[0] != dimCtmp[1] || dimVtmp[0] != dimVtmp[1]) { CERR << "ERROR: # of conduction or/and valance bands are different for spin up and down." << endl; EXIT(1); }
    }
    dimC = dimCtmp[0]; dimV = dimVtmp[0];
    
    nspns = spins.size(); nkpts = kpoints.size(); nbnds = bands[0].size();
    nstates = nspns * nkpts * nbnds;
    
    return;
}

void waveclass::StatesRead() { // Load all states of current wavefunction
    isStatesRead = true;
      spins.resize(totnspns); iota(  spins.begin(),   spins.end(), 0);
    kpoints.resize(totnkpts); iota(kpoints.begin(), kpoints.end(), 0);
    bands = new vector<int>[totnspns];
    for(int is = 0; is < totnspns; is++) {
        bands[is].resize(totnbnds);
        iota(bands[is].begin(), bands[is].end(), 0);
    }
    
    nspns = spins.size(); nkpts = kpoints.size(); nbnds = bands[0].size();
    nstates = nspns * nkpts * nbnds;
    
    return;
}

void waveclass::ReadKptvecNpw(const char *wavecar, const int isgamma, const int isncl) {
    ifstream inf(wavecar, ios::in|ios::binary);
    if(!inf.is_open()) { CERR << "WAVECAR file " << wavecar << " not open" << endl; EXIT(1); }
    isReadKptvecNpw = true;
    double *dbtmp = new double[4];
    inf.read((char*)dbtmp, sizeof(double)); // read the first line, total 3 values
    size_t irdum  = (size_t)dbtmp[0];
    inf.seekg(irdum, ios::beg); // skip to the second line
    inf.read((char*)dbtmp, sizeof(double) * 2);
    int itotnbnds = round(dbtmp[1]);

    double *kptvecs_1d = NULL;
    MpiWindowShareMemoryInitial(nkpts,     npw,        local_npw_node,     window_npw);
    MpiWindowShareMemoryInitial(nkpts * 3, kptvecs_1d, local_kptvecs_node, window_kptvecs);
    
    kptvecs = new double*[nkpts];
    npw_loc = new int[nkpts]; // for each column, npw_loc are the same. Here maloc for all columns for paw calculation
    for(int ikpt = node_rank; ikpt < nkpts; ikpt += node_size) {
        inf.seekg(irdum * (2 + kpoints[ikpt] * (1 + itotnbnds)), ios::beg);
        inf.read((char*)dbtmp, sizeof(double) * 4);
        npw[ikpt] = dbtmp[0];
        if(isgamma) npw[ikpt] = npw[ikpt] * 2 - 1;
        if(isncl)   npw[ikpt] /= 2;
        for(int i = 0; i < 3; i++) kptvecs_1d[ikpt * 3 + i] = dbtmp[1 + i];
    }
    MPI_Win_fence(0, window_npw);
    MPI_Win_fence(0, window_kptvecs);
    MPI_Barrier(node_comm);
    for(int ikpt = 0; ikpt < nkpts; ikpt++) {
        kptvecs[ikpt] = kptvecs_1d + ikpt * 3;
        npw_loc[ikpt] = Numroc(npw[ikpt], MB_ROW, myprow_group, nprow_group);
    }

    inf.close();
    delete[] dbtmp;
    
    return;
}

bool waveclass::WithinSphere(int i, int j, int k, double *kpoint) {
    // G vector (i, j, k) is or not in the ENCUT sphere at kpoint
    double tmp, res = 0.0;
    for(int n = 0; n < 3; n++) {
        tmp = b[0][n] * (i + kpoint[0])+ b[1][n] * (j + kpoint[1]) + b[2][n] * (k + kpoint[2]);
        res += tmp * tmp * 4.0 * M_PI * M_PI;
    }

    res = res * rytoev * autoa * autoa;
    if(res < emax) return true;
    else return false;
}

void waveclass::GetgBegidxBlocklen(int rk, int sz, MPI_Comm &comm) {
    locbeg = new vector<int>[nkpts];
    glbbeg = new vector<int>[nkpts];
    bcklen = new vector<int>[nkpts];
    for(int ik = 0; ik < nkpts; ik++) BlacsBegidxBlocklen(myprow_group, nprow_group, npw[ik], MB_ROW,
                                                          locbeg[ik], glbbeg[ik], bcklen[ik]);
    return;
}

void waveclass::Getgabsdir(int rk, int sz, MPI_Comm &comm) {
    gabs               = new double*[nkpts];     gkabs               = new double*[nkpts]; 
    gtheta             = new double*[nkpts];     gktheta             = new double*[nkpts];
    gphi               = new double*[nkpts];     gkphi               = new double*[nkpts];
    const size_t totnpw = accumulate(npw, npw + nkpts, 0);
    MpiWindowShareMemoryInitial(totnpw, gabsall,    local_gabs_node,    window_gabs);
    MpiWindowShareMemoryInitial(totnpw, gthetaall,  local_gtheta_node,  window_gtheta);
    MpiWindowShareMemoryInitial(totnpw, gphiall,    local_gphi_node,    window_gphi);
    MpiWindowShareMemoryInitial(totnpw, gkabsall,   local_gkabs_node,   window_gkabs);
    MpiWindowShareMemoryInitial(totnpw, gkthetaall, local_gktheta_node, window_gktheta);
    MpiWindowShareMemoryInitial(totnpw, gkphiall,   local_gkphi_node,   window_gkphi);
    size_t sumnpw = 0;
    for(int ik = 0; ik < nkpts; ik++) {
        gabs[ik]    = gabsall    + sumnpw;
        gtheta[ik]  = gthetaall  + sumnpw;
        gphi[ik]    = gphiall    + sumnpw;
        gkabs[ik]   = gkabsall   + sumnpw;
        gktheta[ik] = gkthetaall + sumnpw;
        gkphi[ik]   = gkphiall   + sumnpw;
        sumnpw += npw[ik];
    }
    MPI_Barrier(node_comm);

    int gx, gy, gz;
    for(int ik = node_rank; ik < nkpts; ik += node_size) {
        #pragma omp parallel for private(gx, gy, gz)
        for(int ig = 0; ig < npw[ik]; ig++) {
            IdxNat1toSym3(gidx[ik][ig], gx, gy, gz, ng[0], ng[1], ng[2]);
            XYZtoRTP<double, double>(2 * M_PI * gx, 2 * M_PI * gy, 2 * M_PI * gz, b,
                                     gabs[ik][ig], gtheta[ik][ig], gphi[ik][ig]);
            XYZtoRTP<double, double>(2 * M_PI * (gx + kptvecs[ik][0]), 
                                     2 * M_PI * (gy + kptvecs[ik][1]),
                                     2 * M_PI * (gz + kptvecs[ik][2]), b,
                                     gkabs[ik][ig], gktheta[ik][ig], gkphi[ik][ig]);
        }
    }
    
    MPI_Barrier(world_comm);
    return;
}

void waveclass::Getgidx(const int isgamma, const int vasp_ver, const int checknpw) {
    if(!(vasp_ver == 2 || vasp_ver == 4 || vasp_ver == 6)) { CERR << "wrong vasp version: only support for vasp 5.2.x, vasp 5.4.x or vasp 6.x" << endl; EXIT(1); }
    isGetgidx = true;

    gidx    = new int*[nkpts];
    gidxRev = new int*[nkpts];
    int nn;
    // Basic, get gidx and gidxRev
    if(isgamma) {
        int *gidxTmp = new int[ngtot * 2];
        MpiWindowShareMemoryInitial(ngtot,  gidxRev[0], local_gidxRev_node, window_gidxRev);
        MpiWindowShareMemoryInitial(npw[0], gidx[0],    local_gidx_node,    window_gidx);
        
        if(is_node_root) {
            #pragma omp parallel for
            for(int ig = 0; ig < ngtot; ig++) gidxRev[0][ig] = -1;

            nn = 0; 
            for(int kk = 0; kk < ng[2]; kk++) {
                int kng = IdxNat1toSym1(kk, ng[2]);
                for(int jj = 0; jj < ng[1]; jj++) {
                    int jng = IdxNat1toSym1(jj, ng[1]);
                    for(int ii = 0; ii < ng[0]; ii++) {    
                        int ing = IdxNat1toSym1(ii, ng[0]);
                        
                        int ijk   = IdxNat3toNat1( ii,   jj,   kk,  ng[0], ng[1], ng[2]);
                        int ijkCI = IdxSym3toNat1(-ing, -jng, -kng, ng[0], ng[1], ng[2]); // CI: center inverse
                        
                        // vasp 5.2.x
                        if(vasp_ver == 2) {
                            if( (kng > 0) ||
                                (kng == 0 && jng > 0) ||
                                (kng == 0 && jng == 0 && ing >= 0) ) {
                                if(WithinSphere(ing, jng, kng, kptvecs[0])) {
                                    gidxRev[0][ijkCI]   = nn + ngtot; // don't change order of 
                                    gidxRev[0][ijk]     = nn;         // the 
                                    gidxTmp[nn + ngtot] = ijkCI;      // four
                                    gidxTmp[nn++]       = ijk;        // lines
                                }
                            }
                        }
                        // vasp 5.4.x/6.x
                        else if(vasp_ver == 4 || vasp_ver == 6) {
                            if( (ing > 0) ||
                                (ing == 0 && jng > 0) ||
                                (ing == 0 && jng == 0 && kng >= 0) ) {
                                if(WithinSphere(ing, jng, kng, kptvecs[0])) {
                                    gidxRev[0][ijkCI]   = nn + ngtot; // don't change order of
                                    gidxRev[0][ijk]     = nn;         // the
                                    gidxTmp[nn + ngtot] = ijkCI;      // four
                                    gidxTmp[nn++]       = ijk;        // lines
                                }
                            }
                        }

                    } // ii
                } // jj
            } // kk
            if(npw[0] != 2 * nn - 1) { CERR << "ERROR in Getgidx: No. of planewaves for Gamma: " << npw[0] / 2 << ' ' << nn << endl; EXIT(1); } // check npw
            gidx[0] = new int[2 * nn - 1];
            gidx[0][0] = gidxTmp[0]; // should be 0
            #pragma omp parallel for
            for(int ig = 1; ig < nn; ig++) {
                gidx[0][ig]          = gidxTmp[ig];          //  1 ~ nn - 1
                gidx[0][ig - 1 + nn] = gidxTmp[ig + ngtot];  // nn ~ 2 * nn - 2
            }
            #pragma omp parallel for
            for(int ig = 0; ig < ngtot; ig++) {
                // there should be nn - 1 effective values from ngtot + 1 to ngtot + nn - 1
                if(gidxRev[0][ig] > ngtot) gidxRev[0][ig] += (nn - ngtot - 1); // gidxRev[0][ig] == ngtot not useful
            }
            delete[] gidxTmp;
        } // node root
        MPI_Win_fence(0, window_gidx);
        MPI_Win_fence(0, window_gidxRev);
        MPI_Barrier(node_comm);
    } // is gamma
    else {
        int *gidxTmp = new int[ngtot];
        const size_t totnpw = accumulate(npw, npw + nkpts, 0);
        MpiWindowShareMemoryInitial((size_t)ngtot * nkpts, gidxRevall, local_gidxRev_node, window_gidxRev);
        MpiWindowShareMemoryInitial(totnpw,                gidxall,    local_gidx_node,    window_gidx);
        size_t sumnpw = 0;
        for(int ikpt = 0; ikpt < nkpts; ikpt++) {
            gidxRev[ikpt] = gidxRevall + (size_t)ngtot * ikpt;
            gidx[ikpt] = gidxall + sumnpw;
            sumnpw += npw[ikpt];
        }
        for(int ikpt = node_rank; ikpt < nkpts; ikpt += node_size) {
            nn = 0;
            for(int kk = 0; kk < ng[2]; kk++) {            
                int kng = IdxNat1toSym1(kk, ng[2]);
                for(int jj = 0; jj < ng[1]; jj++) {        
                    int jng = IdxNat1toSym1(jj, ng[1]);
                    for(int ii = 0; ii < ng[0]; ii++) {    
                        int ing = IdxNat1toSym1(ii, ng[0]);
                        int ijk = IdxNat3toNat1(ii, jj, kk, ng[0], ng[1], ng[2]);
                        if(WithinSphere(ing, jng, kng, kptvecs[ikpt])) {
                            gidxRev[ikpt][ijk] = nn;
                            gidxTmp[nn++]      = ijk;
                        }
                        else gidxRev[ikpt][ijk] = -1;
                    } // ii
                } // jj
            } // kk
            if(npw[ikpt] != nn) {  
                cerr << "kpoint # = " << ikpt + 1 << ": " << npw[ikpt] << ' ' << nn << endl;
                cerr << "ERROR in Getgidx: No. of planewaves" << endl; exit(1);
            }
            #pragma omp parallel for
            for(int ig = 0; ig < nn; ig++) gidx[ikpt][ig] = gidxTmp[ig];
        } // ikpt
        MPI_Barrier(node_comm);
        delete[] gidxTmp;
    }

    // Advanced, get gabs, gtheta, gphi, gkabs, gktheta, gkphi
    Getgabsdir(world_rk, world_sz, world_comm);
    
    // multiple processes
    GetgBegidxBlocklen(sub_rank, sub_size, group_comm);
    MPI_Barrier(group_comm);
    
    return;
}

void waveclass::IniEigens(const int malcoeff) { // malloc memory for coefficients
    isIniEigens = true;
    is_malcoeff = malcoeff;
    eigens = new eigenclass[nstates];
    for(int ie = 0; ie < nstates; ie++) eigens[ie].Initial(this, ie, malcoeff);
    return;
}

void waveclass::ReadCoeff(const char *wavecar, const int isgamma, const int isrealc) {
    if(!isReadCoeff) {
        coeff_malloc_in_node = new complex<double>*[nkpts];
        const size_t tottmp1 = (size_t)nspns * nbnds * (1 + spinor) * accumulate(npw, npw + nkpts, 0);
        MpiWindowShareMemoryInitial(tottmp1, coeff_malloc_in_nodeall, local_recpc, window_recpc, malloc_root);
        if(isrealc) {
            const size_t tottmp2 = (size_t)nspns * nbnds * (1 + spinor) * ngftot * nkpts;
            real_space_c_in_node = new complex<double>*[nkpts];
            MpiWindowShareMemoryInitial(tottmp2, real_space_c_in_nodeall, local_realc, window_realc, malloc_root);
        }
        size_t sumtmp1 = 0, sumtmp2 = 0;
        for(int ikpt = 0; ikpt < nkpts; ikpt++) {
            coeff_malloc_in_node[ikpt] = coeff_malloc_in_nodeall + sumtmp1;
            sumtmp1 += (size_t)nspns * nbnds * (1 + spinor) * npw[ikpt];
            if(isrealc) {
                real_space_c_in_node[ikpt] = real_space_c_in_nodeall + sumtmp2;
                sumtmp2 += (size_t)nspns * nbnds * (1 + spinor) * ngftot;
            }
            for(int ispn = 0; ispn < nspns; ispn++) {
                for(int ibnd = 0; ibnd < nbnds; ibnd++) {
                    eigens[ispn * nkpts * nbnds + ikpt * nbnds + ibnd].coeff = 
                    coeff_malloc_in_node[ikpt] + ((size_t)ispn * nbnds + ibnd) * npw[ikpt] * (1 + spinor);
                    if(isrealc)
                    eigens[ispn * nkpts * nbnds + ikpt * nbnds + ibnd].realc = 
                    real_space_c_in_node[ikpt] + ((size_t)ispn * nbnds + ibnd) * ngftot    * (1 + spinor);
                }
            }
        }
        MpiWindowShareMemoryInitial(nstates, energies, local_energies, window_energies, malloc_root);
        MpiWindowShareMemoryInitial(nstates, weights,  local_weights,  window_weights,  malloc_root);
        isReadCoeff = true;
        if(isrealc) isRealCoeff = true;
        MPI_Barrier(group_comm);
    }

    if(dftcode == "vasp") {
        int ikb_beg = node_rank - malloc_root, ikb_delta = share_memory_len;
        for(int is = 0; is < nspns; is++) {
            for(int ikb = ikb_beg; ikb < nkpts * nbnds; ikb += ikb_delta)
            eigens[is * nkpts * nbnds + ikb].Getcoeff(wavecar, sub_rank, sub_size, group_comm,
                                                      energies, weights, isgamma);
            MPI_Barrier(group_comm);
            for(int ikb = 0; ikb < nkpts * nbnds; ikb++) { // extra setting
                int ie = is * nkpts * nbnds + ikb;
                eigens[ie].energy = energies[ie];
                eigens[ie].weight = weights[ie];
            }
        }
        MPI_Barrier(group_comm);
    }
    return;
}

void waveclass::CalcRealc() {
    // the pw coefficients may update by CalcNAC in nac.cpp etc.
    // so this routine is outside Getcoeff 
    int ikb_beg = node_rank - malloc_root, ikb_delta = share_memory_len;
    for(int is = 0; is < nspns; is++) {
        for(int ikb = ikb_beg; ikb < nkpts * nbnds; ikb += ikb_delta) eigens[is * nkpts * nbnds + ikb].Getrealc();
    }
    MPI_Barrier(group_comm);
    return;
}

void waveclass::CalcProjPhi(const char *normcar, const bool is_write_normcar) {
    complex<double> **coeffs = new complex<double>*[nstates];
    for(int ie = 0; ie < nstates; ie++) coeffs[ie] = eigens[ie].coeff;
    for(int iatom = 0; iatom < numatoms; iatom++) atoms[iatom].Getprojphi(nspns, nkpts, nbnds, kpoints, coeffs, npw, volume, spinor);
    delete[] coeffs;
    if(is_sub_root && is_write_normcar) {
        ofstream otf(normcar, ios::out|ios::binary);
        if(!otf.is_open()) { cerr << "can't open " << normcar << endl; exit(1); }
        otf.write((char*)&numatoms, sizeof(int));
        for(int iatom = 0; iatom < numatoms; iatom++) otf.write((char*)&atoms[iatom].potc->lmmax, sizeof(int));
        otf.write((char*)&nspns, sizeof(int)); for(int is = 0; is < nspns; is++) otf.write((char*)&spins[is],     sizeof(int));
        otf.write((char*)&nkpts, sizeof(int)); for(int ik = 0; ik < nkpts; ik++) otf.write((char*)&kpoints[ik],   sizeof(int));
        otf.write((char*)&nbnds, sizeof(int)); 
        for(int is = 0; is < nspns; is++) for(int ib = 0; ib < nbnds; ib++)      otf.write((char*)&bands[is][ib], sizeof(int));
        otf.close();
    }
    if(is_write_normcar) for(int iatom = 0; iatom < numatoms; iatom++) atoms[iatom].Writeprojphi(normcar, nspns, nkpts, nbnds, spinor);
    MPI_Barrier(group_comm);
    return;
}

int waveclass::ReadProjPhi(const char *normcar) {
    return 0;
    int read_status = 1;
    ifstream inf;
    if(is_sub_root) {
        inf.open(normcar, ios::in|ios::binary);
        if(!inf.is_open()) {
            read_status = 0;
        }
        else {
            int itmp = 0;
            inf.read((char*)&itmp, sizeof(int)); if(itmp != numatoms) read_status = 0;
            if(read_status) for(int iatom = 0; iatom < numatoms; iatom++) { 
                inf.read((char*)&itmp, sizeof(int)); if(itmp != atoms[iatom].potc->lmmax) { read_status = 0; break; }
            }
            inf.read((char*)&itmp, sizeof(int)); if(itmp != nspns) read_status = 0;
            if(read_status) for(int is = 0; is < nspns; is++) { 
                inf.read((char*)&itmp, sizeof(int)); if(itmp != spins[is]) { read_status = 0; break; }
            }
            inf.read((char*)&itmp, sizeof(int)); if(itmp != nkpts) read_status = 0;
            if(read_status) for(int ik = 0; ik < nkpts; ik++) { 
                inf.read((char*)&itmp, sizeof(int)); if(itmp != kpoints[ik]) { read_status = 0; break; }
            }
            inf.read((char*)&itmp, sizeof(int)); if(itmp != nbnds) read_status = 0;
            if(read_status) for(int isib = 0; isib < nspns * nbnds; isib++) { 
                int is = isib / nbnds, ib = isib % nbnds;
                inf.read((char*)&itmp, sizeof(int)); if(itmp != bands[is][ib]) { read_status = 0; break; }
            }
        }
        if(read_status) cout << "read " << normcar << " successful" << endl;
    }
    MPI_Bcast(&read_status, 1, MPI_INT, sub_root, group_comm);
    if(read_status) {
    }
    
    MPI_Barrier(group_comm);
    return read_status;
}

void waveclass::WriteBandEnergies(const char *filename) {
    ofstream otf(filename, ios::out|ios::binary);
    if(!otf.is_open()) { Cerr << filename << " doesn't open when writing band energies" << endl; exit(1); }
    for(int ist = 0; ist < nstates; ist++) otf.write((char*)(&eigens[ist].energy), sizeof(double));
    otf.close();
    return;
}

void waveclass::WriteExtraPhases(const char *filename) {
    ofstream otf(filename, ios::out|ios::binary);
    if(!otf.is_open()) { Cerr << filename << " doesn't open when writing wavefunction phases" << endl; exit(1); }
    for(int ist = 0; ist < nstates; ist++) otf.write((char*)(&eigens[ist].extraphase), sizeof(complex<double>));
    otf.close();
    return;
}

waveclass::waveclass(const char *wfcFile, const int inSpinor) {
    spinor = (inSpinor ? 1 : 0);
    ifstream inf(wfcFile, ios::in|ios::binary);
    if(!inf.is_open()) { CERR << wfcFile << " doesn't open when initial wavecalss" << endl; EXIT(1); }
    isStatesRead    = false;
    isReadKptvecNpw = false;
    isGetgidx       = false;
    isIniAtoms      = false;
    isIniEigens     = false;
    isReadCoeff     = false;
    isRealCoeff     = false;
    if(dftcode == "vasp") {
        double *dbtmp = new double[12];
        
        inf.read((char*)dbtmp, sizeof(double) * 3); // read the first line, total 3 values
        rdum        = (size_t)round(dbtmp[0]);      // line length
        totnspns    =         round(dbtmp[1]);
        double rtag =               dbtmp[2];

        inf.seekg(rdum, ios::beg); // skip to the second line
        inf.read((char*)dbtmp, sizeof(double) * 12);
        totnkpts = round(dbtmp[0]);
        totnbnds = round(dbtmp[1]);
        emax     =       dbtmp[2];
        for(int i = 0; i < 3; i++) {
            a[i] = new double[3];
            for(int j = 0; j < 3; j++) a[i][j] = dbtmp[3 + i * 3 + j];
        }
        
        inf.close();
        delete[] dbtmp;
    }
    
    if(spinor && is_make_spinor) { totnspns = 1; totnbnds *= 2; rdum *= 2; }
    
    GetRecpLatt();
    Getng();

    DetermineShareMemoryRanks();
}

waveclass::~waveclass() {
    for(int i = 0; i < 3; i++) { delete[] a[i]; delete[] b[i]; }
    delete[] all_ngf_loc_n0; delete[] all_ngf_loc_0_start;
    MPI_Win_free(&window_allroot); MPI_Win_free(&window_allnclr);
    if(isReadKptvecNpw) { 
        delete[] npw_loc;
        MPI_Win_free(&window_npw);
        MPI_Win_free(&window_kptvecs);
    }
    if(isGetgidx) { 
        MPI_Win_free(&window_gidx);
        MPI_Win_free(&window_gidxRev);
        MPI_Win_free(&window_gabs);  MPI_Win_free(&window_gtheta);  MPI_Win_free(&window_gphi);
        MPI_Win_free(&window_gkabs); MPI_Win_free(&window_gktheta); MPI_Win_free(&window_gkphi);
        delete[] gabs;  delete[] gtheta;  delete[] gphi;
        delete[] gkabs; delete[] gktheta; delete[] gkphi;
        delete[] gidx; delete[] gidxRev;
        delete[] locbeg; delete[] glbbeg; delete[] bcklen;
    }
    if(isIniEigens) delete[] eigens;
    if(isReadCoeff) {
        MPI_Win_free(&window_recpc);
        delete[] coeff_malloc_in_node;
        if(isRealCoeff) {
            MPI_Win_free(&window_realc);
            delete[] real_space_c_in_node;
        }
        MPI_Win_free(&window_energies);
        MPI_Win_free(&window_weights);
    }
    if(isIniAtoms) delete[] atoms;
}
