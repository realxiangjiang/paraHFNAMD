#include "paw.h"

void Gradradial(vector<double> &rgrid, vector<double> &f, double *dfdr) { // dfdr = df/dr
/*
    refer to SUBROUTINE GRAD in vasp:src/radial.F
*/  
    int radnmax = rgrid.size();
    double H = log(rgrid.back() / rgrid.front()) / (radnmax - 1);
    
    // first point use first order differentiation
    dfdr[0] = (f[1] - f[0]) / H / rgrid[0];
    // three points formula
    for(int i = 1; i < radnmax - 1; i++) dfdr[i] = (f[i + 1] - f[i - 1]) / (2.0 * H) / rgrid[i];
    // last point
    dfdr[radnmax - 1] = (f[radnmax - 1] - f[radnmax - 2]) / H / rgrid[radnmax - 1];

    return;
}

void pawpotclass::ReadProj() { // read the projector functions from a element's potcar string
    isReadProj = true;
    string line, nonlocstr;
    vector<string> vecstrtmp;
    StringSplit(*ppcstr, nonlocstr, " PAW radial sets\n", false); // remain first part
    nonlocstr = nonlocstr.substr(nonlocstr.find(" Non local Part\n")); // store all "Non local Part"
    istringstream nonlocss(nonlocstr);
    while(getline(nonlocss, line)) {
        if(line.find("Non local Part") != string::npos) {
            getline(nonlocss, line);
            vecstrtmp = StringSplitByBlank(line);
            int L = stoi(vecstrtmp[0]), nprojL = stoi(vecstrtmp[1]);
            for(int i = 0; i < nprojL; i++) projL.push_back(L);
            projRmax = stod(vecstrtmp[2]);

            while(getline(nonlocss, line)) if(line.find("Reciprocal Space Part") != string::npos) break; // find "Reciprocal Space Part" line
            for(int iL = 0; iL < nprojL; iL++) {
                if(iL > 0) getline(nonlocss, line); // skip "Reciprocal Space Part" line
                for(int i = 0; i < NL_NPSQNL; i++) { // loop for all reciprocal projectors
                    getline(nonlocss, line);
                    vecstrtmp = StringSplitByBlank(line);
                    for(vector<string>::iterator it = vecstrtmp.begin(); it != vecstrtmp.end(); it++) qProjs.push_back(stod(*it));
                }
                getline(nonlocss, line); // skip "Real Space Part" line
                for(int i = 0; i < NL_NPSRNL; i++) { // loop for all real projectors
                    getline(nonlocss, line);
                    vecstrtmp = StringSplitByBlank(line);
                    for(vector<string>::iterator it = vecstrtmp.begin(); it != vecstrtmp.end(); it++) rProjs.push_back(stod(*it));
                }
            }
        }
    }
    
    lmmax = 0; lmax = projL.size(); projLbeg.push_back(0);
    for(int i = 0; i < lmax; i++) {
        for(int j = lmmax; j < lmmax + 2 * projL[i] + 1; j++) {
            each_l.push_back(projL[i]);
            each_m.push_back(j - lmmax - projL[i]);
            each_idxl.push_back(i);
        }
        lmmax += 2 * projL[i] + 1;
    }
    lmmax2 = lmmax * lmmax;
    for(int i = 1; i < lmax; i++) projLbeg.push_back(projLbeg[i - 1] + 2 * projL[i - 1] + 1);
    lmmax_loc_r = Numroc(lmmax, MB_ROW, myprow_group, nprow_group);
    lmmax_loc_c = Numroc(lmmax, NB_COL, mypcol_group, npcol_group);
    lmmax2_loc = lmmax_loc_r * lmmax_loc_c;
    maxL = *max_element(projL.begin(), projL.end());
    
    return;
}

void pawpotclass::ReadPartialWFC() {
/*
        Read the ps/ae partial waves. From zqj github.
        
        The data structure in POTCAR:
             
             grid
             aepotential
             core charge-density
             kinetic energy-density
             mkinetic energy-density pseudized
             local pseudopotential core
             pspotential valence only
             core charge-density (pseudized)
             pseudo wavefunction
             ae wavefunction
             ...
             pseudo wavefunction
             ae wavefunction
*/
    isReadPWFC = true;
    string line, locstr;
    vector<string> vecstrtmp;
    locstr = ppcstr->substr(ppcstr->find(" PAW radial sets\n"));
    istringstream locss(locstr);
    for(int i = 0; i < 2; i++) getline(locss, line); // skip to the second line
    radnmax = stoi(StringSplitByBlank(line)[0]);
    vector<double> alldata;
    while(getline(locss, line)) {
        if(line.find("grid")                != string::npos ||
           line.find("pseudo wavefunction") != string::npos ||
           line.find("ae wavefunction")     != string::npos ) {
            for(int ii = 0; ii < radnmax / NCINPOTCAR + (radnmax % NCINPOTCAR ? 1 : 0); ii++) {
                getline(locss, line);
                vecstrtmp = StringSplitByBlank(line);
                for(vector<string>::iterator it = vecstrtmp.begin(); it != vecstrtmp.end(); it++) alldata.push_back(stod(*it));
            }
        }
    }
    
    // store to rgrid, pswfc and aewfc
    rgrid.insert(rgrid.end(), alldata.begin(), alldata.begin() + radnmax);
    pswfcs = new vector<double>[lmax];
    aewfcs = new vector<double>[lmax];
    for(int i = 0; i < lmax; i++) { 
        pswfcs[i].insert(pswfcs[i].end(), alldata.begin() + (2 * i + 1) * radnmax, alldata.begin() + (2 * i + 2) * radnmax);
        aewfcs[i].insert(aewfcs[i].end(), alldata.begin() + (2 * i + 2) * radnmax, alldata.begin() + (2 * i + 3) * radnmax);
    }

    // calculate Simpson integral weight
    SetSimpiWeight();

    return;
}

void pawpotclass::SetSimpiWeight() {
/* copy from zqj Github
       Setup weights for simpson integration on radial grid any radial integral
       can then be evaluated by just summing all radial grid points with the weights
                          
                        \int dr = \sum_i w(i) * f(i)
*/
    // R(i+1) / R(i) = exp(H)
    // R(i) = R(0) * exp(H*i)
    double H = log(rgrid.back() / rgrid.front()) / (radnmax - 1);
    simp_wt = new double[radnmax]();
    for(int ii = radnmax - 1; ii >= 2; ii -= 2) {
        simp_wt[ii]     =       rgrid[ii]     * H / 3.0 + simp_wt[ii];
        simp_wt[ii - 1] = 4.0 * rgrid[ii - 1] * H / 3.0;
        simp_wt[ii - 2] =       rgrid[ii - 2] * H / 3.0;
    }
    return;
}

double pawpotclass::CalcSimpInt(double *f) {
    return Ddot(radnmax, simp_wt, f);
}

void pawpotclass::CalcQij() {
    isCalcQij = true;
/* copy from zqj Github
        
        Calculate the quantity
            
            Q_{ij} = < \phi_i^{AE} | \phi_j^{AE} > - < phi_i^{PS} | phi_j^{PS} >
        
        where \phi^{AE/PS} are the PAW AE/PS waves, which are real functions in VASP PAW POTCAR.
        In POTCAR, only the radial part of the AE/PS partial waves are stored. In order to get 
        the total AE/PS waves, a spherical harmonics should be multiplied to the radial part, i.e. 
        
            \psi^{AE/PS}(r) = (1. / r) * \phi^{AE/PS}(r) * Y(l, m)
        
        where \psi(r) is the total AE/PS partial waves, and \phi(r) are the ones stored in POTCAR.
        Y(l, m) is the real spherical harmonics with quantum number l and m. Note the "1 / r" 
        in front of the equation. In practice, the r**(-2) term disappears when multiply with the
        integration volume "r**2 sin(theta) d theta d phi". In theory, the "Qij" should be integrated 
        inside the PAW cut-off radius, but since the AE and PS partial waves are indentical outside 
        the cut-off radius, therefore the terms outside the cut-off radius cancel each other.
*/
    Qij_z = new complex<double>[max(lmmax2_loc, 1)]();
    MpiWindowShareMemoryInitial(lmmax * lmmax, Qij_z_full, local_Qijzfull_node, window_Qijzfull);
    int ii, jj;
    double *dphi_ij = new double[radnmax]; // AE - PS
    for(int iijj = node_rank; iijj < lmmax2; iijj += node_size) { // column major
        ii = iijj % lmmax;
        jj = iijj / lmmax;
        if( each_l[ii] == each_l[jj] && each_m[ii] == each_m[jj] ) { 
            #pragma omp parallel for
            for(int irad = 0; irad < radnmax; irad++) dphi_ij[irad] = 
            aewfcs[ each_idxl[ii] ][irad] * aewfcs[ each_idxl[jj] ][irad] -
            pswfcs[ each_idxl[ii] ][irad] * pswfcs[ each_idxl[jj] ][irad];
            Qij_z_full[iijj] = CalcSimpInt(dphi_ij);
        }
        else Qij_z_full[iijj] = 0.0;
    }
    delete[] dphi_ij;
    MPI_Barrier(world_comm);

    Blacs_MatrixZScatter(lmmax, lmmax, Qij_z_full, lmmax, Qij_z, lmmax_loc_r);

    MPI_Barrier(group_comm);
    return;
}

void pawpotclass::CalcGij() {
    isCalcGij = true;
/*
        Calculate

        G_{ij} = <\psi_i^{AE} | \grad | \psi_j^{AE}> - <\psi_i^{PS} | \grad | \psi_j^{PS}>

        where \psi^{AE/PS} are the PAW AE/PS waves, \grad is the gradient operator.
        The total AE/PS waves, a spherical harmonics should be multiplied to the radial part, i.e. 
        
            \psi^{AE/PS}(r) = (1. / r) * \phi^{AE/PS}(r) * Y(l, m)

        then one can obtain the final formula:

        <\psi_i| \grad | \psi_j> = sqrt(4pi/3)\int dr [\phi_i(d\phi_j/dr) - \phi_i\phi_j / r]
                                 x \int dOmega Y_lmY_l'm'[e_xY_{1,1} + e_yY_{1,-1} + e_zY_{1,0}]
                                 
                                 + \int dr \phi_i\phi_j / r
                                 x \int dOmega [ e_\theta Y_lm\partialY_l'm'/\partial\theta
                                               + e_\phi Y_lm/sin\theta \partialY_l'm'/\partial\phi ]
*/
    Gij   = new double[max(3 * lmmax2_loc, 1)]();
    Gij_z = new complex<double>[max(3 * lmmax2_loc, 1)]();
    int ii, jj, ii_glb, jj_glb;
    int i_l, i_m, j_l, j_m, i_idxl, j_idxl;
    int idx_1p1lm12, idx_1m1lm12, idx_10lm12;
    int idx_ilm, idx_jlm;
    double *grad_phiAE_jj = new double[radnmax]; // (d/dr)phi_j^{AE}
    double *grad_phiPS_jj = new double[radnmax]; // (d/dr)phi_j^{PS}
    double *dtmp = new double[radnmax];
    double radintres;
    for(int iijj = 0; iijj < lmmax2_loc; iijj++) { // column major
        ii = iijj % lmmax_loc_r;
        jj = iijj / lmmax_loc_r;
        ii_glb = BlacsIdxloc2glb(ii, lmmax, MB_ROW, myprow_group, nprow_group);
        jj_glb = BlacsIdxloc2glb(jj, lmmax, NB_COL, mypcol_group, npcol_group);
        i_l = each_l[ii_glb]; i_m = each_m[ii_glb]; i_idxl = each_idxl[ii_glb];
        j_l = each_l[jj_glb]; j_m = each_m[jj_glb]; j_idxl = each_idxl[jj_glb];
        idx_ilm = i_l * i_l + (i_m + i_l);
        idx_jlm = j_l * j_l + (j_m + j_l);
        
        // L = 1, M = +1
        idx_1p1lm12 = ( 1 * 1 + ( 1 + 1) ) * totlm12 * totlm12 + idx_ilm * totlm12 + idx_jlm;
        // L = 1, M = -1
        idx_1m1lm12 = ( 1 * 1 + (-1 + 1) ) * totlm12 * totlm12 + idx_ilm * totlm12 + idx_jlm;
        // L = 1, M = 0
        idx_10lm12  = ( 1 * 1 + ( 0 + 1) ) * totlm12 * totlm12 + idx_ilm * totlm12 + idx_jlm;

        if(abs(i_l - j_l) <= 1 && 1 <= i_l + j_l && 
           ( abs(YLMY1Y2[idx_1p1lm12]) > 1e-18 || 
             abs(YLMY1Y2[idx_1m1lm12]) > 1e-18 || 
             abs(YLMY1Y2[idx_10lm12])  > 1e-18 )) { // if true, need calculate \phi_i(d\phi_j/dr) - \phi_i\phi_j / r
            Gradradial(rgrid, aewfcs[j_idxl], grad_phiAE_jj);
            Gradradial(rgrid, pswfcs[j_idxl], grad_phiPS_jj);
            
            #pragma omp parallel for
            for(int irad = 0; irad < radnmax; irad++)
            dtmp[irad] = ( aewfcs[i_idxl][irad] * grad_phiAE_jj[irad]
                         - aewfcs[i_idxl][irad] * aewfcs[j_idxl][irad] / rgrid[irad] )
                       - ( pswfcs[i_idxl][irad] * grad_phiPS_jj[irad]
                         - pswfcs[i_idxl][irad] * pswfcs[j_idxl][irad] / rgrid[irad] );
            radintres = CalcSimpInt(dtmp);
            
            Gij[iijj]                  = sqrt(4.0 * M_PI / 3.0) * radintres * YLMY1Y2[idx_1p1lm12];
            Gij[iijj + lmmax2_loc]     = sqrt(4.0 * M_PI / 3.0) * radintres * YLMY1Y2[idx_1m1lm12];
            Gij[iijj + lmmax2_loc * 2] = sqrt(4.0 * M_PI / 3.0) * radintres * YLMY1Y2[idx_10lm12];
        }

        if( abs(piDpj_x[idx_ilm * totlm12 + idx_jlm]) > 1e-18 ||
            abs(piDpj_y[idx_ilm * totlm12 + idx_jlm]) > 1e-18 ||
            (i_m == j_m && abs(piDpj_z[idx_ilm * 4 + i_m]) > 1e-18) ) { // if true, need calculate \phi_i\phi_j / r
            #pragma omp parallel for
            for(int irad = 0; irad < radnmax; irad++)
            dtmp[irad] = aewfcs[i_idxl][irad] * aewfcs[j_idxl][irad] / rgrid[irad]
                       - pswfcs[i_idxl][irad] * pswfcs[j_idxl][irad] / rgrid[irad];
            radintres = CalcSimpInt(dtmp);
            
            Gij[iijj]                  += radintres * piDpj_x[idx_ilm * totlm12 + idx_jlm];
            Gij[iijj + lmmax2_loc]     += radintres * piDpj_y[idx_ilm * totlm12 + idx_jlm];
            if(i_m == j_m) 
            Gij[iijj + lmmax2_loc * 2] += radintres * piDpj_z[idx_ilm * 4 + i_m];
        }
        
        Gij_z[iijj]                  = Gij[iijj];
        Gij_z[iijj + lmmax2_loc]     = Gij[iijj + lmmax2_loc];
        Gij_z[iijj + lmmax2_loc * 2] = Gij[iijj + lmmax2_loc * 2];
    }

    delete[] grad_phiAE_jj; delete[] grad_phiPS_jj; delete[] dtmp;
    return;
}

void pawpotclass::CalcJLij() {
/*
    Calculate the quantity

        JL_{ij} = < \phi_i^{AE} | j_L(kr) | \phi_j^{AE} > - < \phi_i^{PS} | j_L(kr) | \phi_j^{PS} >

    For each L and k, in which we used
        
        \int dr r^2 j_L(kr) (1. / r)\phi_i^{AE,PS} x (1. / r)\phi_j^{AE,PS} 
        
       = < \phi_i^{AE,PS} | j_L(kr) | \phi_j^{AE,PS} >
    
    and j_L() is the spherical Bessel function
*/
    // First get effective ij
    int il, im, jl, jm, idx_LMlm12;
    for(int iijj = 0; iijj < lmmax2; iijj++) {
        il = each_l[ iijj / lmmax ]; im = each_m[ iijj / lmmax ];
        jl = each_l[ iijj % lmmax ]; jm = each_m[ iijj % lmmax ];
        for(int L = 0; L <= maxL; L++) {
            for(int M = -L; M <= L; M++) {
                idx_LMlm12 = ( L * L + (M + L) ) * totlm12 * totlm12 + ( il * il + (im + il) ) * totlm12 
                                                                     + ( jl * jl + (jm + jl) );
                if( abs(il - jl) <= L && L <= il + jl && abs(YLMY1Y2[idx_LMlm12]) > 1e-18 ) {
                    all_nv_ij.push_back(iijj);
                    break;
                }
            } // M
        } // L
    } // iijj
    all_nv_ij.erase(unique(all_nv_ij.begin(), all_nv_ij.end()), all_nv_ij.end());
    tot_nv_ij = all_nv_ij.size();
    //for(int i = 0; i < tot_nv_ij; i++) cout << all_nv_ij[i] << ' '; cout << tot_nv_ij; cout << endl;

    const int totij_loc_col = Numroc(tot_nv_ij, NB_COL, mypcol_group, npcol_group);
    int ij_glb, ii, jj;
    vector<int> idxl_pair;
    for(int ij = 0; ij < tot_nv_ij; ij++) { // first collect all idx pairs
        ii = all_nv_ij[ij] / lmmax;
        jj = all_nv_ij[ij] % lmmax;
        idxl_pair.push_back( each_idxl[ii] * lmax + each_idxl[jj] );
        unique_idxlpair.push_back( each_idxl[ii] * lmax + each_idxl[jj] );
    }
    sort(unique_idxlpair.begin(), unique_idxlpair.end()); // then find the unique pairs
    unique_idxlpair.erase(unique(unique_idxlpair.begin(), unique_idxlpair.end()), unique_idxlpair.end());
    for(int ij = 0; ij < tot_nv_ij; ij++) { // finally give the corresponding idx of each pair
        for(int idx = 0; idx < unique_idxlpair.size(); idx++) {
            if(idxl_pair[ij] == unique_idxlpair[idx]) { // every idxl_pair should find the idx in this loop
                idx_of_idxlpair.push_back(idx);
                break;
            }
        } // loop to search among the elements of unique_idxlpair
    }

    JLij = new double*[maxL + 1];
    MpiWindowShareMemoryInitial((size_t)NPSQNL * unique_idxlpair.size() * (maxL + 1), JLijall, local_JLij_node, window_JLij);
    size_t sumtmp = 0;
    for(int L = 0; L <= maxL; L++) {
        JLij[L] = JLijall + sumtmp;
        sumtmp += (size_t)NPSQNL * unique_idxlpair.size();
    }
    MPI_Barrier(node_comm);

    int ik, ilp, iil, jjl;
    double *jldphi= new double[radnmax]; // j_l x (phi^AE - phi^PS)
    for(int L = 0; L <= maxL; L++) {
        for(int ikilp = node_rank; ikilp < NPSQNL * unique_idxlpair.size(); ikilp += node_size) {
            ik  = ikilp / unique_idxlpair.size();
            ilp = ikilp % unique_idxlpair.size();
            iil = unique_idxlpair[ilp] / lmax;
            jjl = unique_idxlpair[ilp] % lmax;
            #pragma omp parallel for
            for(int irad = 0; irad < radnmax; irad++) jldphi[irad] = 
            ( aewfcs[iil][irad] * aewfcs[jjl][irad] -
              pswfcs[iil][irad] * pswfcs[jjl][irad] ) * sph_bessel( L, projGmax * ik / NPSQNL );
            JLij[L][ik + ilp * NPSQNL] = CalcSimpInt(jldphi);
        }
        MPI_Barrier(node_comm);
        /*for(int i = 0; i < NPSQNL; i++) {
            for(int j = 0; j < unique_idxlpair.size(); j++) cout << JLij[L][i + j * NPSQNL] << ' ';
            cout << endl;
        }*/
    }
    
    delete[] jldphi;
    MPI_Barrier(world_comm);
    return;
}

void pawpotclass::CalcQprojKpts(const int totnkpts, const vector<int> &kpoints, const int *npw,
                                double **gkabs, double **gktheta, double **gkphi) {
    // gkabs/gktheta/gkphi: g + k distance/direction.
    // kpoints[ik]: the ikth kpoint # in [0, totnkpts), kpoint.size() could less than totnkpts to exclude unnecessary ones
    if(!isReadProj) { CERR << "ERROR in CalcQprojKpts: no basic qProjs read in, please check." << endl; EXIT(1); }

    if(!isCalcQprojKpts) {
        size_t tmp_tot_size = 0;
        for(int ik = 0; ik < kpoints.size(); ik++) tmp_tot_size += (size_t)lmmax * npw[ik];
        MpiWindowShareMemoryInitial(tmp_tot_size,
                                    qProjsKpt_all, local_qProjsKpt_node, window_qProjsKpt);
        qProjsKpt = new complex<double>*[totnkpts * lmmax];
        size_t tmp_cur_size = 0;
        for(int ik = 0; ik < kpoints.size(); ik++) {
            int ikpt = kpoints[ik];
            this->kpoints.push_back(ikpt);
            for(int iL = 0; iL < lmax; iL++) {
                int L = projL[iL], iLbeg = projLbeg[iL];
                for(int M = -L; M < L + 1; M++) {
                    qProjsKpt[ikpt * lmmax + iLbeg + (M + L)] = qProjsKpt_all + tmp_cur_size;
                    tmp_cur_size += npw[ik];
                }
            } // iL
        } // ik
        isCalcQprojKpts = true;
    }

    // Interpolation for fgk_c
    int status;
    double *spline_coeff = NULL; // this will malloc in SplineInterpolationPre function
    DFTaskPtr task;              // Data Fitting operations are task based
    double gbeginend[2] = {0.0, projGmax * (NPSQNL - 1) / NPSQNL};
    SplineInterpolationPre(task, spline_coeff, NPSQNL, lmax, gbeginend, (const double*)(&qProjs[0]));

    vector<int> idx_within_gmax;
    int ikpt; // ikpt # in [0, totnkpts)
    int gksize;
    double *gkabs_c = NULL, *fgk_c = NULL;
    double theta, phi;
    int ig_full;
    for(int ik = node_rank; ik < kpoints.size(); ik += node_size) {
        ikpt = kpoints[ik]; 
        idx_within_gmax.clear();
        for(int ig = 0; ig < npw[ik]; ig++) 
        if(gkabs[ik][ig] < projGmax) idx_within_gmax.push_back(ig);

        gksize = idx_within_gmax.size();
        if(gksize == 0) continue;
        gkabs_c = new double[gksize];
        fgk_c   = new double[lmax * gksize];
        Dgather(gksize, gkabs[ik], gkabs_c, (int*)(&idx_within_gmax[0]));
        SplineInterpolation(task, gksize, gkabs_c, fgk_c); // Interpolation for all L
        
        for(int iL = 0; iL < lmax; iL++) {
            int L = projL[iL], iLbeg = projLbeg[iL];
            for(int M = -L; M < L + 1; M++) {
                #pragma omp parallel for private(ig_full, theta, phi)
                for(int ig = 0; ig < gksize; ig++) { // compacted ig
                    ig_full = idx_within_gmax[ig]; 
                    theta   = gktheta[ik][ig_full];
                    phi     =   gkphi[ik][ig_full];
                    qProjsKpt[ikpt * lmmax + iLbeg + (M + L)][ig_full] = pow(iu_d, L) * fgk_c[iL * gksize + ig] * RealSphHarFun(L, M, theta, phi);
                }
            }
        }
        delete[] gkabs_c; delete[] fgk_c;
    }
    status = dfDeleteTask(&task); delete[] spline_coeff;
    
    MPI_Barrier(world_comm);
    return;
}

void pawpotclass::CalcEkrij(const int nqpts,
                            const int *npw, const int *npw_loc,
                            double **qgabs, double **qgtheta, double **qgphi) {
    // qgabs/qgtheta/qgphi: q + G distance/direction
    if(isCalcEkrij) return; // may multiplly call this routine, set this return to avoid repeatedly calculate
    isCalcEkrij = true;

    // First calculate JLij for ik = [0:NPSQNL]
    CalcJLij();
    
    // Interpolation for JLij
    int status;
    double **spline_coeff = new double*[maxL + 1];
    DFTaskPtr *task = new DFTaskPtr[maxL + 1]; // Data Fitting operations are task based
    double gbeginend[2] = {0.0, projGmax * (NPSQNL - 1) / NPSQNL};
    for(int L = 0; L <= maxL; L++) {
        spline_coeff[L] = NULL; // each will malloc in SplineInterpolationPre function
        if(unique_idxlpair.size() > 0)
        SplineInterpolationPre(task[L], spline_coeff[L], NPSQNL, unique_idxlpair.size(),
                               gbeginend, JLij[L]);
    }

    this->nqpts = nqpts;
    Ekrij = new complex<double>*[nqpts];
    if(!sexpikr) { // calculate Ekrij in memory
        size_t tmp_tot_size = 0;
        for(int iqpt = 0; iqpt < nqpts; iqpt++) tmp_tot_size += npw[iqpt] * tot_nv_ij;
        MpiWindowShareMemoryInitial(tmp_tot_size, Ekrij_all, local_Ekrij_node, window_Ekrij);
        size_t tmp_cur_size = 0;
        for(int iqpt = 0; iqpt < nqpts; iqpt++) {
            Ekrij[iqpt] = Ekrij_all + tmp_cur_size;
            tmp_cur_size += (size_t)npw[iqpt] * tot_nv_ij;
        }
    }
    
    vector<int> sub_idx_within_gmax;
    vector<int> glb_idx_within_gmax;
    int ig_loc, ig_glb;
    int qgsize;
    int il, im, jl, jm, ij_glb;
    int idx_LMlm12;
    double *qgabs_c = NULL, *JLij_c = NULL;
    
    const int totij_loc_col = Numroc(tot_nv_ij, NB_COL, mypcol_group, npcol_group);
    
    int qptbeg, qptstep;
    if(!sexpikr) { qptbeg = node_rank; qptstep = node_size; }
    else         { qptbeg = world_rk;  qptstep = world_sz;  }
    for(int iqpt = qptbeg; iqpt < nqpts; iqpt += qptstep) {
        if(!sexpikr) fill_n(Ekrij[iqpt], (size_t)npw[iqpt] * tot_nv_ij, complex<double>(0.0, 0.0));
        else Ekrij[iqpt] = new complex<double>[(size_t)npw[iqpt] * tot_nv_ij]();
        glb_idx_within_gmax.clear();
        for(int ig = 0; ig < npw[iqpt]; ig++) {
            if(qgabs[iqpt][ig] < projGmax) {
                glb_idx_within_gmax.push_back(ig);
            }
        }
        
        qgsize = glb_idx_within_gmax.size();
        if(qgsize == 0) continue;
        qgabs_c = new double[max(qgsize, 1)];
        JLij_c = new double[max(unique_idxlpair.size() * qgsize, (size_t)1)];
        if(unique_idxlpair.size() > 0)
        Dgather(qgsize, qgabs[iqpt], qgabs_c, (int*)(&glb_idx_within_gmax[0]));
        //if(iqpt == 0) cout << element << ":" << endl;
        for(int L = 0; L <= maxL; L++) {
            if(unique_idxlpair.size() > 0)
            SplineInterpolation(task[L], qgsize, qgabs_c, JLij_c); // integration for each idxl pair
            for(int M = -L; M <= L; M++)
            for(int iijj = 0; iijj < tot_nv_ij; iijj++) { // loop for each local columns
                ij_glb = all_nv_ij[ iijj ];
                il = each_l[ ij_glb / lmmax ]; im = each_m[ ij_glb / lmmax ];
                jl = each_l[ ij_glb % lmmax ]; jm = each_m[ ij_glb % lmmax ];
                idx_LMlm12 = ( L * L + (M + L) ) * totlm12 * totlm12 + ( il * il + (im + il) ) * totlm12 
                                                                     + ( jl * jl + (jm + jl) );
                if( !(abs(il - jl) <= L && L <= il + jl && abs(YLMY1Y2[idx_LMlm12]) > 1e-18) ) continue;
                /*if(iqpt == 0) {
                    cout << '(' << L << ',' << M << ") " << ij_glb << '/' << lmmax2 << ' '
                         << ' ' << iijj << '/' << tot_nv_ij << '/' << lmmax2 
                         << ' ' << il << ' ' << jl << ' ' << im << ' ' << jm << ' '
                         << YLMY1Y2[idx_LMlm12] << endl;
                }*/
                #pragma omp parallel for private(ig_loc, ig_glb)
                for(int ig = 0; ig < qgsize; ig++) {
                    ig_glb = glb_idx_within_gmax[ig];
                    Ekrij[iqpt][ig_glb + iijj * npw[iqpt]] += 
                    4.0 * M_PI * pow(iu_d, L) * // 4pi x i^L
                    RealSphHarFun(L, M, qgtheta[iqpt][ig_glb], qgphi[iqpt][ig_glb]) * // Y_lm(q+G)
                    JLij_c[ ig + idx_of_idxlpair[iijj] * qgsize ] // \int dr r^2 j_l(kr) (\phi^{AE}_ij - \phi^{PS}_ij)
                    * YLMY1Y2[idx_LMlm12]; // \int dOmega Y_LM Y_ilim Y_jljm
                }
            }
        }
        delete[] qgabs_c; delete[] JLij_c;
        if(sexpikr) { // store Ekrij for current element
            ofstream otf( (namddir + "/.tmpEkrij/" + element + to_string(iqpt)).c_str(), ios::out|ios::binary );
            assert(otf.is_open());
            otf.write((char*)(Ekrij[iqpt]), sizeof(complex<double>) * npw[iqpt] * tot_nv_ij);
            otf.close();
            delete[] Ekrij[iqpt];
        }

        /*if(iqpt == 0) {
            for(int i = 0; i < npw_loc[iqpt]; i++) {
                cout << "ig = " << i << ":"; for(int j = 0; j < totij_loc_col; j++) cout << Ekrij[iqpt][i + j * npw_loc[iqpt]];
                cout << endl;
            }
        }*/
    } // iqpt

    for(int iL = 0; iL <= maxL; iL++) {
        if(unique_idxlpair.size() > 0) {
            status = dfDeleteTask(task + iL);
            delete[] spline_coeff[iL];
        }
    }
    delete[] spline_coeff;
    
    MPI_Win_free(&window_JLij);
    delete[] JLij;
    
    MPI_Barrier(group_comm);
    return;
}

void pawpotclass::ReadEkrij(const int iqpt, const int *npw) {
    ifstream inf( (namddir + "/.tmpEkrij/" + element + to_string(iqpt)).c_str(), ios::in|ios::binary );
    assert(inf.is_open());
    Ekrij[iqpt] = new complex<double>[(size_t)npw[iqpt] * tot_nv_ij]();
    inf.read((char*)(Ekrij[iqpt]), sizeof(complex<double>) * npw[iqpt] * tot_nv_ij);
    inf.close();
}

void pawpotclass::DelEkrij(const int iqpt) {
    delete[] Ekrij[iqpt];
}

pawpotclass::pawpotclass() {}
void pawpotclass::Initial(string &ppcstr_in) { // ppc: paw potcar
    ppcstr = &ppcstr_in;
    string preline, line; // previous line, line
    vector<string> vecstr;
    istringstream ppcss(*ppcstr);

    /* some test
    for(int i = 0; i < 5; i++) {
        getline(ppcss, line);
        cout << line << endl;
        vecstr = StringSplitByBlank(line);
        for(vector<string>::iterator it = vecstr.begin(); it != vecstr.end(); it++) cout << *it << endl;
    }*/

    getline(ppcss, line); // read the first line, usually "PAW_PBE ELEMENT data"
    vecstr = StringSplitByBlank(line);
    element = vecstr[1];  // the second word is the element name
    do{
        preline = line;
        getline(ppcss, line);
    } while(line.find("Non local Part") == string::npos); // Find first Non local Part, previous line stores maximal G
    vecstr = StringSplitByBlank(preline);
    projGmax = stod(vecstr[0]);

    isReadProj = false;
    isReadPWFC = false;
}

pawpotclass::~pawpotclass() {
    if(isReadPWFC) { delete[] pswfcs; delete[] aewfcs; delete[] simp_wt; }
    if(isCalcQprojKpts) {
        MPI_Win_free(&window_qProjsKpt);
        delete[] qProjsKpt;
    }
    if(isCalcQij) { delete[] Qij_z; MPI_Win_free(&window_Qijzfull); }
    if(isCalcGij) { delete[] Gij; delete[] Gij_z; }
    if(isCalcEkrij) {
        if(!sexpikr) MPI_Win_free(&window_Ekrij);
        else {
            for(int iqpt = world_rk; iqpt < nqpts; iqpt += world_sz) {
                remove( (namddir + "/.tmpEkrij/" + element + to_string(iqpt)).c_str() );
            }
        }
        delete[] Ekrij;
    }
}

int ReadAllpawpot(pawpotclass *&pawpots, const char* potcar) {
    ifstream inf(potcar, ios::in);
    if(!inf.is_open()) { COUT << "POTCAR file " << potcar << " doesn't find, please check." << endl; EXIT(1); }
    string potcarstr = WholeFile2String(inf);
    inf.close();
    const int nelements = StringMatchCount(potcarstr, "End of Dataset\n");
    pawpots = new pawpotclass[nelements];
    string onepotstr;
    long idx = 0;
    for(int i = 0; i < nelements; i++) {
        potcarstr = potcarstr.substr(idx);
        idx = StringSplit(potcarstr, onepotstr, "End of Dataset\n");
        pawpots[i].Initial(onepotstr);
        pawpots[i].ReadProj();
        pawpots[i].ReadPartialWFC();
        pawpots[i].CalcQij();
        pawpots[i].CalcGij();
    }
    
    return nelements;
}
