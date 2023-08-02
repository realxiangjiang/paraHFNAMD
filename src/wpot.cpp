#include "wpot.h"

void wpotclass::Qclass::ReadHeadWing(double omega0, int ncells, double *gabsGW) {
/*
    omega = volume of the entire crystal 
          = omega0(cell volume of GW cell, usually primitive cell) 
          x ncells(# of kpoints x cell ratio), cell ratio = det(M)
    in the unit of Angstrom^3
*/
    isReadHeadWing = true;
    ifstream wgamma(wpotdir + "/W0001.tmp", ios::in|ios::binary);
    if(!wgamma.is_open()) { CERR << "wpot file W0001.tmp doesn't find, please check." << endl; EXIT(1); }
    head = new complex<double>[10](); // thr former 9 are read from W0001.tmp, last one is used in current code
    wgamma.seekg(sizeof(int), ios::beg); // skip first length record
    int npwGW; wgamma.read((char*)&npwGW, sizeof(int)); // read GW cell npw
    wgamma.seekg(5 * sizeof(int), ios::beg); // skip (NP, 0) and head length record
    wgamma.read((char*)head, sizeof(complex<double>) * 9);
    head[9] = (head[0] + head[4] + head[8]) / 3.0 + 1.0;  // W0001.tmp stores ([epsilon(0,0)^-1] - 1), so there's a "+1"
    head[9] *= 2.0 * pow(6.0 / (M_PI * ncells * omega0), 1.0 / 3.0) // 2[6/(pi x omega)]^(1/3)
             * autoa * 2.0 * rytoev;                                // Hartree to eV
    size_t szjump = 2 * sizeof(int) + 2 * sizeof(int) // NP, 0
                  + 2 * sizeof(int) + 9 * sizeof(complex<double>) // head
                  + sizeof(int); // first "int" of wing
    size_t szjumpC = szjump + 3 * npwGW * sizeof(complex<double>) + sizeof(int) // wing
                   + sizeof(int); // first "int" of cwing
    
    /* 
       wing/cwing is the first column/row
       
       | head~q^2 cwing~q |
       | wing~q    body   |
       
       for wing in vasp, W0001.tmp stores 
          4 * pi / omega0 / |g|^2 * [epsilon(G,0)^-1] 
       in the unit of eV
    */
    const double wingcoeff = pow(3.0 / (4 * M_PI * ncells * omega0), 2.0 / 3.0) // 4pi[3/(4pi x omega)]^(2/3)
                           * omega0;                                            // and "4pi" cancels in W0001.tmp
    // because gabsGW will multipled later and [wingcoeff x |g|] is the unit of "1", no unit transforms are needed here
    MpiWindowShareMemoryInitial(npwSC,  wing, local_wing_node,   window_wing);
    MpiWindowShareMemoryInitial(npwSC, cwing, local_cwing_node, window_cwing);
    MPI_Barrier(node_comm);
    if(is_node_root) {
        fill_n( wing, npwSC, complex<double>(0.0, 0.0));
        fill_n(cwing, npwSC, complex<double>(0.0, 0.0));
    }
    MPI_Barrier(node_comm);

    complex<double> wingtmp;
    // for iG = 0, wing/cwing should automatically be zero because of multipling |g|
    for(int iG = node_rank; iG < npwSC; iG += node_size) {
        if(qind[iG] == 0 && gind[iG] < npwGW) {
            for(int iw = 0; iw < 3; iw++) {
                wgamma.seekg( szjump + (gind[iG] + iw * npwGW) * sizeof(complex<double>), ios::beg );
                wgamma.read( (char*)&wingtmp, sizeof(complex<double>) );
                wing[iG] += wingtmp;
            }
            wing[iG] *= wingcoeff * gabsGW[ gind[iG] ] / 3.0; // "3.0" for average
        }
        if(qind[iG] == 0 && gind[iG] < npwGW) {
            for(int iw = 0; iw < 3; iw++) {
                wgamma.seekg( szjumpC + (gind[iG] + iw * npwGW) * sizeof(complex<double>), ios::beg );
                wgamma.read( (char*)&wingtmp, sizeof(complex<double>) );
                cwing[iG] += wingtmp;
            }
            cwing[iG] *= wingcoeff * gabsGW[ gind[iG] ] / 3.0; // "3.0" for average
        }
    }
    wgamma.close();

    if(is_node_root) cwing[0] = head[9]; // for convenience, add average head to first element of cwing

    return;
}

wpotclass::Qclass::Qclass() {}
wpotclass::Qclass::~Qclass() {
    if(isReadHeadWing) {
        delete[] head;
        MPI_Win_free(&window_wing);
        MPI_Win_free(&window_cwing);
    }
    if(isInitial) {
    }
}

void RecpSC2pc(double &x0, double &x1, double &x2,
               double  X0, double  X1, double  X2, double T[][3]) { 
    // (x1 x2 x3) = (X1 X2 X3)T
    x0 = X0 * T[0][0] + X1 * T[0][1] + X2 * T[0][2];
    x1 = X0 * T[1][0] + X1 * T[1][1] + X2 * T[1][2];
    x2 = X0 * T[2][0] + X1 * T[2][1] + X2 * T[2][2];
    return;
}

void wpotclass::ReadGWOUTCAR(const char *gwoutcar) {
    ifstream inf(gwoutcar, ios::in);
    if(!inf.is_open()) { CERR << "OUTCAR file " << gwoutcar << " doesn't find, please check."; EXIT(1); }
    string line;
    vector<string> vecstrtmp;
   
    int itmp;
    getline(inf, line);
    if(line.find("vasp.5.2.") != string::npos) itmp = 0;
    else if(line.find("vasp.5.4") != string::npos) itmp = 1;
    else if(line.find("vasp.6.") != string::npos) itmp = 1;

    while(getline(inf, line)) {
        if(line.find("direct lattice vectors") != string::npos) { // first get the lattice infomation
            for(int ii = 0; ii < 3; ii++) {
                getline(inf, line);
                vecstrtmp = StringSplitByBlank(line);
                for(int jj = 0; jj < 3; jj++) {
                    a_GW[ii][jj] = stod(vecstrtmp[jj]);
                    b_GW[ii][jj] = stod(vecstrtmp[jj + 3]);
                }
            }
            double *dbtmp = new double[3];
            VecCross3d(a_GW[0], a_GW[1], dbtmp);
            omega0_GW = VecDot3d(dbtmp, a_GW[2]);
            delete[] dbtmp;
            //break; // Comment this "break" because there may be another "primitive" lattice shown in OUTCAR
            //       // Last lattice vectors are always right
        }
    }
    inf.clear(); // clear the error flags
    inf.seekg(0, ios::beg);
    
    while(getline(inf, line)) {
        if(line.find("Subroutine INISYM returns") != string::npos) {
            vecstrtmp = StringSplitByBlank(line);
            numopt = stoi(vecstrtmp[4]);
            optcs = new operatorclass[numopt];
        }
        else if(line.find("Space group operators") != string::npos) {
            getline(inf, line);
            for(int ii = 0; ii < numopt; ii++) {
                getline(inf, line);
                vecstrtmp = StringSplitByBlank(line);
                optcs[ii].Initial( ii, round(stod(vecstrtmp[1])), stod(vecstrtmp[2]) / 180.0 * M_PI,
                                   stod(vecstrtmp[3]), stod(vecstrtmp[4]), stod(vecstrtmp[5]),
                                   stod(vecstrtmp[6]), stod(vecstrtmp[7]), stod(vecstrtmp[8]), a_GW, b_GW );
            }
        }
        else if(line.find("IBZKPT_HF") != string::npos) {
            while(getline(inf, line)) {
                if(line.find("in 1st BZ") != string::npos) {
                    vecstrtmp = StringSplitByBlank(line);
                    nktotgw = stoi(vecstrtmp[1]);
                    if(nktotgw != NK_GW[0] * NK_GW[1] * NK_GW[2]) {
                        CERR << "# of kpoints WRONG: nkptsgw not equal to wpotdir/OUTCAR" << endl; EXIT(1);
                    }
                    allgwkpts = new double*[nktotgw];
                    gwkpinibz = new int[nktotgw];
                    gwkp_trev = new int[nktotgw];
                    gwkpgr2ibz = new int*[nktotgw];
                    gwkpoptno = new int[nktotgw]();
                }
                else if(line.find("Following reciprocal coordinates") != string::npos) break;
            }
            for(int ii = 0; ii < nktotgw; ii++) {
                allgwkpts[ii] = new double[3];
                gwkpgr2ibz[ii] = new int[3]();
                getline(inf, line);
                vecstrtmp = StringSplitByBlank(line);
                for(int jj = 0; jj < 3; jj++) allgwkpts[ii][jj] = stod(vecstrtmp[jj]);
                gwkpinibz[ii] = stoi(vecstrtmp[3 + itmp]) - 1;
                gwkp_trev[ii] = (vecstrtmp[5 + itmp] == "F" ? 1 : -1); // F or T: w/o or w/ time-reversal
            }
            nkibrgw = *max_element(gwkpinibz, gwkpinibz + nktotgw) + 1;
            /*for(int ii = 0; ii < nkibrgw; ii++) {
                cout << ii + 1 
                     << ": (" << gwkpgr2ibz[ii][0] << ' ' << gwkpgr2ibz[ii][1] << ' ' << gwkpgr2ibz[ii][2] << ") "
                     <<  gwkpoptno[ii] << ' ' << gwkp_trev[ii] << endl;
            }*/
            for(int ii = nkibrgw; ii < nktotgw; ii++) {
                gwkpoptno[ii] = FindOpNum(gwkpgr2ibz[ii], gwkp_trev[ii], allgwkpts[ gwkpinibz[ii] ], allgwkpts[ii],
                                          optcs, numopt);
                /*cout << ii + 1
                     << ": (" << gwkpgr2ibz[ii][0] << ' ' << gwkpgr2ibz[ii][1] << ' ' << gwkpgr2ibz[ii][2] << ") "
                     <<  gwkpoptno[ii] << ' ' << gwkp_trev[ii] << endl;*/
            }
        }
        else if(line.find("ENCUT ") != string::npos) {
            vecstrtmp = StringSplitByBlank(line);
            emax_GW = stod(vecstrtmp[2]); // temporary
        }
        else if(line.find("ENCUTGW ") != string::npos) {
            vecstrtmp = StringSplitByBlank(line);
            emax_SCGW = min(stod(vecstrtmp[2]), wvc->emax);
            if(encutgw < 0) emax_GW *= 2.0 / 3.0; // not set encutgw manually
            else {
                emax_GW = stod(vecstrtmp[2]);
                if(abs(emax_GW - encutgw) > 0.2) {
                    CERR << "ENCUTGW = " << emax_GW << " eV in " << wpotdir << "/OUTCAR, ";
                    CERR << "but encutgw = " << encutgw << " eV in file input, please check" << endl;
                    EXIT(1);
                }
            }
            break; // this "break" is necessary to avoid repeating reading
        }
    }

    inf.close();
    return;
}

bool wpotclass::GetM() {
/*
    let A and a(b) are row vectors for SC and pc(GW)
    
    (A1)     (a1)
    (A2) = M (a2) --> A x b^T =  M x a x b^T --> M = Ab^T
    (A3)     (a3)

*/
    double dbtmp;
    int flag = 1;
    for(int ir = 0; ir < 3; ir++)
    for(int jc = 0; jc < 3; jc++) {
        dbtmp = 0.0;
        for(int s = 0; s < 3; s++) dbtmp += wvc->a[ir][s] * b_GW[jc][s];
        M[ir][jc] = round(dbtmp);
        if( !(abs(dbtmp - round(dbtmp)) < 1e-3) ) flag = 0;
    }
    if(!flag) {
        for(int ir = 0; ir < 3; ir++) {
            for(int jc = 0; jc < 3; jc++) cout << M[ir][jc] << ' ';
            cout << endl;
        }
        return false;
    }

    detM = M[0][0] * (M[1][1] * M[2][2] - M[1][2] * M[2][1])
         - M[0][1] * (M[1][0] * M[2][2] - M[1][2] * M[2][0])
         + M[0][2] * (M[1][0] * M[2][1] - M[1][1] * M[2][0]);
    ncells = detM * (NK_SC[0] * NK_SC[1] * NK_SC[2]);

    double *MT = new double[9]; // transpose of M: M^T
    for(int ir = 0; ir < 3; ir++)
    for(int jc = 0; jc < 3; jc++) {
        MT[ir * 3 + jc] = (double)M[jc][ir];
    }
    Dgetri(3, MT, 3); // (M^T)^(-1)
    for(int ir = 0; ir < 3; ir++)
    for(int jc = 0; jc < 3; jc++) {
        MTinv[ir][jc] = MT[ir * 3 + jc];
    }
    delete[] MT;

    return true;
}

void wpotclass::GetNG() {
    #pragma omp parallel for
    for(int i = 0; i < 3; i++) {
        double dbtmp = sqrt(emax_SCGW / rytoev) / (2.0 * M_PI / (sqrt(VecDot3d(wvc->a[i], wvc->a[i])) / autoa));
        NG_SCGW[i] = (int)(2 * dbtmp + 5);
    }
    NGtot_SCGW = NG_SCGW[0] * NG_SCGW[1] * NG_SCGW[2];

    return;
}

void wpotclass::Getng() {
    #pragma omp parallel for
    for(int i = 0; i < 3; i++) {
        double dbtmp = sqrt(emax_GW / rytoev) / (2.0 * M_PI / (sqrt(VecDot3d(a_GW[i], a_GW[i])) / autoa));
        ng_GW[i] = (int)(2 * dbtmp + 5);
    }
    ngtot_GW = ng_GW[0] * ng_GW[1] * ng_GW[2];

    return;
}

bool wpotclass::WithinSphere(int i, int j, int k, double *qpoint) {
    // G vector (i, j, k) is or not in the ENCUT sphere at kpoint
    double tmp, res = 0.0;
    for(int n = 0; n < 3; n++) {
        tmp = wvc->b[0][n] * (i + qpoint[0]) + wvc->b[1][n] * (j + qpoint[1]) + wvc->b[2][n] * (k + qpoint[2]);
        res += tmp * tmp * 4.0 * M_PI * M_PI;
    }

    res = res * rytoev * autoa * autoa;
    if(res < emax_SCGW) return true;
    else return false;
}

bool wpotclass::WithinSphereGW(int i, int j, int k, double *kpoint) {
    // G vector (i, j, k) is or not in the ENCUT sphere at kpoint
    double tmp, res = 0.0;
    for(int n = 0; n < 3; n++) {
        tmp = b_GW[0][n] * (i + kpoint[0]) + b_GW[1][n] * (j + kpoint[1]) + b_GW[2][n] * (k + kpoint[2]);
        res += tmp * tmp * 4.0 * M_PI * M_PI;
    }

    res = res * rytoev * autoa * autoa;
    if(res < emax_GW) return true;
    else return false;
}

void wpotclass::GetgidxGW() {
    int nn, readnpw;
    MpiWindowShareMemoryInitial(nkibrgw, npwGW, local_npwGW_node, window_npwGW);
    MpiWindowShareMemoryInitial(ngtot_GW, gabsGW, local_gabsGW_node, window_gabsGW); // only [0, npwGW[0]) valid
    int *gidxTmp = new int[ngtot_GW];
    gidxRevGW            = new int*[nkibrgw];
    size_t tottmp = ngtot_GW * nkibrgw;
    MpiWindowShareMemoryInitial(tottmp, gidxRevGWall, local_gidxRevGW_node, window_gidxRevGW);
    for(int ikpt = 0; ikpt < nkibrgw; ikpt++) {
        gidxRevGW[ikpt] = gidxRevGWall + (size_t)ikpt * ngtot_GW;
    }
    ifstream wpotfile;
    for(int ikpt = node_rank; ikpt < nkibrgw; ikpt += node_size) {
        nn = 0;
        for(int kk = 0; kk < ng_GW[2]; kk++) {            
            int kng = IdxNat1toSym1(kk, ng_GW[2]);
            for(int jj = 0; jj < ng_GW[1]; jj++) {        
                int jng = IdxNat1toSym1(jj, ng_GW[1]);
                for(int ii = 0; ii < ng_GW[0]; ii++) {    
                    int ing = IdxNat1toSym1(ii, ng_GW[0]);
                    int ijk = IdxNat3toNat1(ii, jj, kk, ng_GW[0], ng_GW[1], ng_GW[2]);
                    if(WithinSphereGW(ing, jng, kng, allgwkpts[ikpt])) { // ibr part is the [0, nkibrgw) of allgwkpts[ikpt]
                        gidxRevGW[ikpt][ijk] = nn;
                        gidxTmp[nn++]        = ijk;
                    }
                    else gidxRevGW[ikpt][ijk] = -1;
                } // ii
            } // jj
        } // kk
        
        // check npw with Wxxxx.tmp
        wpotfile.open(wpotdir + "/W" + Int2Str(ikpt + 1, 4) + ".tmp", ios::in|ios::binary);
        if(!wpotfile.is_open()) { CERR << wpotdir + "/W" + Int2Str(ikpt + 1, 4) + ".tmp" << " doesn't find, please check." << endl; EXIT(1); }
        wpotfile.seekg(sizeof(int), ios::cur); // skip the first "int"
        wpotfile.read((char*)&readnpw, sizeof(int));
        wpotfile.close();
        if(nn != readnpw) { CERR << "ERROR: read in npw = " << readnpw << " for GW in " << wpotdir + "/W" + Int2Str(ikpt + 1, 4) + ".tmp" << " doesn't equal to computed npw = " << nn << endl; EXIT(1); }
        
        // set npwGW
        npwGW[ikpt] = nn;
        // reset "-1" to npwGW[ikpt]
        #pragma omp parallel for
        for(int ig = 0; ig < ngtot_GW; ig++) {
            if(gidxRevGW[ikpt][ig] < 0) gidxRevGW[ikpt][ig] = nn;
        }

        // set gabsGW for ikpt = 0
        if(ikpt == 0) {
            int gx, gy, gz;
            double theta, phi;
            #pragma omp parallel for private(gx, gy, gz, theta, phi)
            for(int ig = 0; ig < npwGW[0]; ig++) {
                IdxNat1toSym3(gidxTmp[ig], gx, gy, gz, ng_GW[0], ng_GW[1], ng_GW[2]);
                XYZtoRTP<double, double>(2 * M_PI * gx, 2 * M_PI * gy, 2 * M_PI * gz, b_GW,
                                         gabsGW[ig], theta, phi);
            }
        }
    } // ikpt
    delete[] gidxTmp;
    
    return;
}

void wpotclass::MatchGWkpt(const double q0, const double q1, const double q2, // input: -1.0 <= q012 <= 1.0
                           int &ikfull, double &distance) {                   // output: kpt index and distance of q and k
    double rr, theta, phi;
    distance = 1.0e200;
    for(int ikpt = 0; ikpt < nktotgw; ikpt++) {
        XYZtoRTP<double, double>(q0 - allgwkpts[ikpt][0], 
                                 q1 - allgwkpts[ikpt][1], 
                                 q2 - allgwkpts[ikpt][2], b_GW, rr, theta, phi);
        if(rr < distance) { distance = rr; ikfull = ikpt; }
    }
    return;
}

void wpotclass::Doublex012toqg(const double x0, const double x1, const double x2,
                               vector<int> &qind, vector<int> &gind,
                               vector< complex<double> > &eigtau, vector<int> &time_reve_kfull) {
    int g[3] = {round(x0), round(x1), round(x2)};
    double q[3] = {x0 - g[0], x1 - g[1], x2 - g[2]};
    int i, j, k, ii, jj, kk;
    int kfull, ikfull;
    double distance = 1.0e200, idist;
    double *gpgr = new double[3];   // [+/-]G + G_R
    double *rtgpgr = new double[3]; // R^-1([+/-]G + G_R)
    int g1[3] = {0, 0, 0};          // G1 = R^-1([+/1]G + G_R)
                                    // [+/-] for w/o and w/ time-reversal symmetry
    
    for(int ijk = 0; ijk < 27; ijk++) {
        i = ijk / 9;
        j = ijk % 9 / 3;
        k = ijk % 3;
        i -= 1; j -= 1; k -= 1;
        if( q[0] + i < -1.0000001 || q[0] + i > 1.0000001 ||
            q[1] + j < -1.0000001 || q[1] + j > 1.0000001 ||
            q[2] + k < -1.0000001 || q[2] + k > 1.0000001 ) continue;
        MatchGWkpt(q[0] + i, q[1] + j, q[2] + k, ikfull, idist);
        if(idist < distance) { 
            distance = idist; kfull = ikfull;
            ii = i; jj = j; kk = k;
        }
    }
    g[0] -= ii; g[1] -= jj; g[2] -= kk;
    
    const int idx_of_ibzkpt = gwkpinibz[kfull];
    qind.push_back( idx_of_ibzkpt );
    time_reve_kfull.push_back( gwkp_trev[kfull] * kfull );
    if(kfull < nkibrgw) {
        if(g[0] < ng_GW[0] / 2 + 1 && g[0] > - (ng_GW[0] + 1) / 2 &&
           g[1] < ng_GW[1] / 2 + 1 && g[1] > - (ng_GW[1] + 1) / 2 &&
           g[2] < ng_GW[2] / 2 + 1 && g[2] > - (ng_GW[2] + 1) / 2 )
        gind.push_back( gidxRevGW[idx_of_ibzkpt]
                                 [IdxSym3toNat1(g[0], g[1], g[2], ng_GW[0], ng_GW[1], ng_GW[2])] );
        else gind.push_back( npwGW[idx_of_ibzkpt] );
        eigtau.push_back(1.0);
    } // kfull in ibz
    else {
        gpgr[0] = (double)gwkp_trev[kfull] * g[0] + gwkpgr2ibz[kfull][0];
        gpgr[1] = (double)gwkp_trev[kfull] * g[1] + gwkpgr2ibz[kfull][1];
        gpgr[2] = (double)gwkp_trev[kfull] * g[2] + gwkpgr2ibz[kfull][2];
        Dgemv("CblasColMajor", "CblasNoTrans", 3, 3,
              1.0, optcs[ gwkpoptno[kfull] ].aRTb, 3, gpgr, 1,
              0.0,                                  rtgpgr, 1);
        for(int s = 0; s < 3; s++) g1[s] = round(rtgpgr[s]);
        if(g1[0] < ng_GW[0] / 2 + 1 && g1[0] > - (ng_GW[0] + 1) / 2 &&
           g1[1] < ng_GW[1] / 2 + 1 && g1[1] > - (ng_GW[1] + 1) / 2 &&
           g1[2] < ng_GW[2] / 2 + 1 && g1[2] > - (ng_GW[2] + 1) / 2 ) {
            gind.push_back( gidxRevGW[idx_of_ibzkpt]
                            [IdxSym3toNat1(g1[0], g1[1], g1[2], ng_GW[0], ng_GW[1], ng_GW[2])] );
            eigtau.push_back(
                exp(2.0 * M_PI * iu_d * ( (double)g[0] * optcs[ gwkpoptno[kfull] ].tau[0] +
                                          (double)g[1] * optcs[ gwkpoptno[kfull] ].tau[1] +
                                          (double)g[2] * optcs[ gwkpoptno[kfull] ].tau[2] ))
            );
        }
        else {
            gind.push_back( npwGW[idx_of_ibzkpt] );
            eigtau.push_back(1.0);
        }
    } // kfull out of ibz

    delete[] gpgr; delete[] rtgpgr;

    return;
}

void wpotclass::QpG2qpg() {
/*
    Q+G to q+g, Q+G for Super Cell(SC), q+g for primitive cell (pc)
    
                    (b1)                     (B1)
    q+g = (x1 x2 x3)(b2)  =  Q+G = (X1 X2 X3)(B2)
                    (b3)                     (B3)
                                 
                                                         (b1)
                                 = (X1 X2 X3)[(M^T)^(-1)](b2)
                                                         (b3)
    
    (x1 x2 x3) = (X1 X2 X3)[(M^T)^(-1)]
*/
    numQ = (2 * NK_SC[0] - 1) * (2 * NK_SC[1] - 1) * (2 * NK_SC[2] - 1); // number of kpoints difference: Q = k1 - k2
    Qcs = new Qclass[numQ];
    double x0, x1, x2;
    vector<int> *qind = new vector<int>[numQ];
    vector<int> *gind = new vector<int>[numQ];
    vector<int> *Gind = new vector<int>[numQ];
    vector< complex<double> > *eigtau = new vector< complex<double> >[numQ];
    vector<int> *time_reve_kfull = new vector<int>[numQ];
    int nn;
    size_t tottmp = nkibrgw * numQ;
    MpiWindowShareMemoryInitial(tottmp, numq_of_ibrkall, local_nqofibr_node, window_nqofibr);
    for(int iQQ = 0; iQQ < numQ; iQQ++) {
        Qcs[iQQ].numq_of_ibrk = numq_of_ibrkall + nkibrgw * iQQ;
    }
    MPI_Barrier(node_comm);

    for(int iQQ = node_rank; iQQ < numQ; iQQ += node_size) {
        int iQ[3];
        double Qpoint[3];
        IdxNat1toSym3(iQQ, iQ[2], iQ[1], iQ[0], // slow to fast axis: z->y->x
                           2 * NK_SC[2] - 1, 2 * NK_SC[1] - 1, 2 * NK_SC[0] - 1);
        for(int s = 0; s < 3; s++) {
            Qpoint[s] = (double)iQ[s] / NK_SC[s];
            Qcs[iQQ].Qptvec[s] = Qpoint[s]; 
            Qcs[iQQ].Qpt[s]  = iQ[s];
        }
        
        nn = 0;
        for(int kk = 0; kk < NG_SCGW[2]; kk++) {
            int kng = IdxNat1toSym1(kk, NG_SCGW[2]);
            for(int jj = 0; jj < NG_SCGW[1]; jj++) {
                int jng = IdxNat1toSym1(jj, NG_SCGW[1]);
                for(int ii = 0; ii < NG_SCGW[0]; ii++) {    
                    int ing = IdxNat1toSym1(ii, NG_SCGW[0]);
                    
                    int ijk = IdxNat3toNat1(ii, jj, kk, NG_SCGW[0], NG_SCGW[1], NG_SCGW[2]);
                    if(WithinSphere(ing, jng, kng, Qpoint)) {
                        Gind[iQQ].push_back(ijk);
                        RecpSC2pc(x0, x1, x2,              // (x0, x1, x2)
                                  Qpoint[0] + (double)ing, // (X0, X1, X2)
                                  Qpoint[1] + (double)jng,
                                  Qpoint[2] + (double)kng, MTinv); 
                        Doublex012toqg(x0, x1, x2, qind[iQQ], gind[iQQ], eigtau[iQQ], time_reve_kfull[iQQ]);
                        nn++;
                    }
                }
            }
        }
        assert(qind[iQQ].size() == nn);
        assert(qind[iQQ].size() == gind[iQQ].size());
        assert(qind[iQQ].size() == Gind[iQQ].size());
        assert(qind[iQQ].size() == eigtau[iQQ].size());
        assert(qind[iQQ].size() == time_reve_kfull[iQQ].size());
        Qcs[iQQ].npwSC = nn; // npwSC
        
        // this four lines gurantee qind/gind/Gind sorted in ascending order
        vector<int> asqind = Argsort((const int*)&(qind[iQQ][0]), nn);
        Igather_inplace(nn, (int*)&(qind[iQQ][0]), (const int*)&(asqind[0]));
        Igather_inplace(nn, (int*)&(gind[iQQ][0]), (const int*)&(asqind[0]));
        Igather_inplace(nn, (int*)&(Gind[iQQ][0]), (const int*)&(asqind[0]));
        Zgather_inplace(nn, (complex<double>*)&(eigtau[iQQ][0]), (const int*)&(asqind[0]));
        Igather_inplace(nn, (int*)&(time_reve_kfull[iQQ][0]), (const int*)&(asqind[0]));
        /*cout << "iQQ = " << iQQ << endl;
        for(int i = 0; i < nn; i++) cout << qind[iQQ][i] << ' '; cout << endl;
        for(int i = 0; i < nn; i++) cout << gind[iQQ][i] << ' '; cout << endl;
        for(int i = 0; i < nn; i++) cout << Gind[iQQ][i] << ' '; cout << endl;
        for(int i = 0; i < nn; i++) cout << time_reve_kfull[iQQ][i] << ' '; cout << endl;*/
        // # of same ibrkpt
        for(int ik = 0; ik < nkibrgw; ik++) Qcs[iQQ].numq_of_ibrk[ik] = 0;
        for(int iG = 0; iG < nn; iG++) Qcs[iQQ].numq_of_ibrk[ qind[iQQ][iG] ]++;
        assert( nn == accumulate(Qcs[iQQ].numq_of_ibrk,
                                 Qcs[iQQ].numq_of_ibrk + nkibrgw, 0) );
    } // iQQ
    MPI_Barrier(node_comm);

    for(int rt = 0; rt < node_size; rt++) { // loop for root(rt) to broadcast
        for(int iQQ = rt; iQQ < numQ; iQQ += node_size) {
            MPI_Bcast((int*)(&Qcs[iQQ].npwSC), 1, MPI_INT, rt, node_comm); // first broadcast npwSC etc.
            MPI_Bcast(Qcs[iQQ].Qptvec, 3, MPI_DOUBLE, rt, node_comm);
            MPI_Bcast(Qcs[iQQ].Qpt,    3, MPI_INT,    rt, node_comm);
        }
    }
    MPI_Barrier(node_comm);
    
    tottmp = 0;
    for(int iQQ = 0; iQQ < numQ; iQQ++) tottmp += Qcs[iQQ].npwSC;
    MpiWindowShareMemoryInitial(tottmp, Gindall,   local_Gind_node, window_Gind);
    MpiWindowShareMemoryInitial(tottmp, qindall,   local_qind_node, window_qind);
    MpiWindowShareMemoryInitial(tottmp, gindall,   local_gind_node, window_gind);
    MpiWindowShareMemoryInitial(tottmp, eigtauall, local_eigtau_node, window_eigtau);
    MpiWindowShareMemoryInitial(tottmp, time_reve_kfullall, local_trk_node, window_trk);
    size_t sumtmp = 0;
    for(int iQQ = 0; iQQ < numQ; iQQ++) {
        Qcs[iQQ].Gind = Gindall + sumtmp;
        Qcs[iQQ].qind = qindall + sumtmp;
        Qcs[iQQ].gind = gindall + sumtmp;
        Qcs[iQQ].eigtau = eigtauall + sumtmp;
        Qcs[iQQ].time_reve_kfull = time_reve_kfullall + sumtmp;
        sumtmp += Qcs[iQQ].npwSC;
        Qcs[iQQ].isInitial = 1;
        Qcs[iQQ].nkibrgw = nkibrgw;
        // npwSC_loc, npwSC_loc_row, npwSC_loc_col
        Qcs[iQQ].npwSC_loc     = Numroc(Qcs[iQQ].npwSC, MB_ROW, myprow_group, nprow_group);
        Qcs[iQQ].npwSC_loc_row = Numroc(Qcs[iQQ].npwSC, MB_ROW, myprow_group, nprow_group);
        Qcs[iQQ].npwSC_loc_col = Numroc(Qcs[iQQ].npwSC, NB_COL, mypcol_group, npcol_group);
    }
    MPI_Barrier(node_comm);
    
    for(int iQQ = node_rank; iQQ < numQ; iQQ += node_size) {
        // Gind, qind, gind
        copy(Gind[iQQ].begin(), Gind[iQQ].end(), Qcs[iQQ].Gind);
        copy(qind[iQQ].begin(), qind[iQQ].end(), Qcs[iQQ].qind);
        copy(gind[iQQ].begin(), gind[iQQ].end(), Qcs[iQQ].gind);
        // eigtau, time_reve_kfull
        copy(eigtau[iQQ].begin(), eigtau[iQQ].end(), Qcs[iQQ].eigtau);
        copy(time_reve_kfull[iQQ].begin(), time_reve_kfull[iQQ].end(), Qcs[iQQ].time_reve_kfull);
    }
    MPI_Barrier(node_comm);

    delete[] qind; delete[] gind; delete[] Gind;
    delete[] eigtau; delete[] time_reve_kfull;
    
    MPI_Barrier(world_comm);
    return;
}

void wpotclass::ReadWxxxx() {
/*
    only the diagnal elements are read in
    to calculate W_{GG} x B^{kc}_{k'c'}(G) where the later B
    are distributed stored, so for same processes row, W_{GG}
    should be read the same sub(part of)-values
*/
    isReadWxxxx = true;
    
    size_t tottmp = 0;
    for(int iQQ = 0; iQQ < numQ; iQQ++) tottmp += Qcs[iQQ].npwSC;
    MpiWindowShareMemoryInitial(tottmp, wggall, local_wgg_node, window_wgg);
    size_t sumtmp = 0;
    for(int iQQ = 0; iQQ < numQ; iQQ++) {
        Qcs[iQQ].wgg = wggall + sumtmp;
        sumtmp += Qcs[iQQ].npwSC;
    }
    MPI_Barrier(node_comm);

    for(int iQQ = 0; iQQ < numQ; iQQ++) 
    if(is_node_root) fill_n(Qcs[iQQ].wgg, Qcs[iQQ].npwSC, complex<double>(0.0, 0.0));
    MPI_Barrier(node_comm);
    
    ifstream wxxxx;
    complex<double> *cdtmp = NULL;
    
    //for(int ikpt = 0; ikpt < nkibrgw; ikpt++) {
    for(int ikpt = node_rank; ikpt < nkibrgw; ikpt += node_size) {
        wxxxx.open(wpotdir + "/W" + Int2Str(ikpt + 1, 4) + ".tmp", ios::in|ios::binary);
        if(!wxxxx.is_open()) { Cerr << "wpot file " << "W" + Int2Str(ikpt + 1, 4) + ".tmp" << " doesn't find, please check."; exit(1); }
        cdtmp = new complex<double>[ npwGW[ikpt] + 1 ](); // "+1" for the last temporary zero-value
        wxxxx.seekg(  2 * sizeof(int) + 2 * sizeof(int) // NP, 0
                    + 2 * sizeof(int) + 9 * sizeof(complex<double>) // head
                    + 2 * sizeof(int) + 3 * npwGW[ikpt] * sizeof(complex<double>)  //  wing
                    + 2 * sizeof(int) + 3 * npwGW[ikpt] * sizeof(complex<double>)  // cwing
                    + sizeof(int), ios::beg ); // first "int" in body
        wxxxx.read((char*)cdtmp, sizeof(complex<double>) * npwGW[ikpt]);
        wxxxx.close();

        for(int iQQ = 0; iQQ < numQ; iQQ++) {
        //for(int iQQ = node_rank; iQQ < numQ; iQQ += node_size) {
            if(Qcs[iQQ].numq_of_ibrk[ikpt] == 0) continue;
            int idxbeg = (ikpt > 0 ? accumulate(Qcs[iQQ].numq_of_ibrk, Qcs[iQQ].numq_of_ibrk + ikpt, 0) : 0);
            // wxxxx.tmp to wgg
            Zgather(Qcs[iQQ].numq_of_ibrk[ikpt], cdtmp, Qcs[iQQ].wgg + idxbeg, Qcs[iQQ].gind + idxbeg);
            #pragma omp parallel for
            for(int ig = 0; ig < Qcs[iQQ].numq_of_ibrk[ikpt]; ig++) {
                if(Qcs[iQQ].time_reve_kfull[idxbeg + ig] < 0)
                Qcs[iQQ].wgg[idxbeg + ig] = conj(Qcs[iQQ].wgg[idxbeg + ig]);
            }
        }
        delete[] cdtmp;
    } // ikpt in ibrkpt

    // add head to wgg, multiply "ncells" to cancel a factor of "1 / ncells" when calculating direct terms
    if(is_node_root) Qcs[0].wgg[0] = Qcs[0].head[9] * (double)ncells;
    
    MPI_Barrier(world_comm);
    return;
}

void wpotclass::ReadOneWFull(const int iQQ) {
/*
    Including the whole W_{GG'} stored in WFULLxxxx.tmp are read in
    to calculate [B^{kc}_{k'c'}(G)]^T x W_{GG'} x [B^{kv}_{k'v'}(G')]*
    where B^{kn}_{k'n'}(G) are the matrix with dimension (N_G x N_nN_n')
    the whole W_{GG'} are distributed stored among the processes grid
*/
    Qcs[iQQ].wfullggp = new complex<double>[(size_t)Qcs[iQQ].npwSC * Qcs[iQQ].npwSC]();
    
    ifstream wfull;
    complex<double> *cdtmp = NULL;
    for(int ikpt = 0; ikpt < nkibrgw; ikpt++) {
        if(Qcs[iQQ].numq_of_ibrk[ikpt] == 0) continue;
        int idxbeg = (ikpt > 0 ? accumulate(Qcs[iQQ].numq_of_ibrk, Qcs[iQQ].numq_of_ibrk + ikpt, 0) : 0);
        
        wfull.open(wpotdir + "/WFULL" + Int2Str(ikpt + 1, 4) + ".tmp", ios::in|ios::binary);
        if(!wfull.is_open()) { Cerr << "wpot file " << "WFULL" + Int2Str(ikpt + 1, 4) + ".tmp" << " doesn't find, please check."; exit(1); }
        cdtmp = new complex<double>[ (size_t)npwGW[ikpt] * npwGW[ikpt] + 1](); // "+1" for the last temporary zero
        wfull.seekg(  2 * sizeof(int) + 2 * sizeof(int) // NP, NP
                    + 2 * sizeof(int) + 9 * sizeof(complex<double>) // head
                    + 2 * sizeof(int) + 3 * npwGW[ikpt] * sizeof(complex<double>)  //  wing
                    + 2 * sizeof(int) + 3 * npwGW[ikpt] * sizeof(complex<double>)  // cwing
                    + sizeof(int), ios::beg ); // first "int" in body
        wfull.read((char*)cdtmp, sizeof(complex<double>) * npwGW[ikpt] * npwGW[ikpt]);
        wfull.close();
            
        for(int jcol = 0; jcol < Qcs[iQQ].numq_of_ibrk[ikpt]; jcol++) {
            if(Qcs[iQQ].gind[idxbeg + jcol] == npwGW[ikpt]) continue;
            size_t idxbeg_col = (size_t)Qcs[iQQ].gind[idxbeg + jcol] * npwGW[ikpt];
            complex<double> cdtmp_last = cdtmp[ idxbeg_col + npwGW[ikpt] ];
            cdtmp[ idxbeg_col + npwGW[ikpt] ] = 0.0; // temporary set to "0"
            Zgather(Qcs[iQQ].numq_of_ibrk[ikpt],
                    cdtmp + idxbeg_col,
                    Qcs[iQQ].wfullggp + (idxbeg + (size_t)(idxbeg + jcol) * Qcs[iQQ].npwSC),
                    Qcs[iQQ].gind + idxbeg);
            #pragma omp parallel for
            for(int irow = 0; irow < Qcs[iQQ].numq_of_ibrk[ikpt]; irow++) {
                if( abs(Qcs[iQQ].time_reve_kfull[idxbeg + irow]) !=
                    abs(Qcs[iQQ].time_reve_kfull[idxbeg + jcol]) ) {
                     Qcs[iQQ].wfullggp[(idxbeg + irow) + (size_t)(idxbeg + jcol) * Qcs[iQQ].npwSC] = 0.0;
                     continue;
                }
                if(Qcs[iQQ].time_reve_kfull[idxbeg + irow] < 0)
                     Qcs[iQQ].wfullggp[(idxbeg + irow) + (size_t)(idxbeg + jcol) * Qcs[iQQ].npwSC] =
                conj(Qcs[iQQ].wfullggp[(idxbeg + irow) + (size_t)(idxbeg + jcol) * Qcs[iQQ].npwSC]);
                Qcs[iQQ].wfullggp[(idxbeg + irow) + (size_t)(idxbeg + jcol) * Qcs[iQQ].npwSC] *=
                conj(Qcs[iQQ].eigtau[idxbeg + irow]) * Qcs[iQQ].eigtau[idxbeg + jcol];
            }
            cdtmp[ idxbeg_col + npwGW[ikpt] ] = cdtmp_last;
        }

        delete[] cdtmp;
    } // ikpt in ibrkpt
    
    // add head and wing/cwing
    // multiply "ncells" to cancel a factor of "1 / ncells" when calculating direct terms
    if(iQQ == 0) {
        // wing
        Zcopy(Qcs[0].npwSC, Qcs[0].wing, 1, Qcs[0].wfullggp, 1);
        ZDscal(Qcs[0].npwSC, (double)ncells, Qcs[0].wfullggp, 1); 
        // cwing
        Zcopy(Qcs[0].npwSC, Qcs[0].cwing, 1, Qcs[0].wfullggp, Qcs[0].npwSC);
        ZDscal(Qcs[0].npwSC, (double)ncells, Qcs[0].wfullggp, Qcs[0].npwSC); 
        // head
        Qcs[0].wfullggp[0] = Qcs[0].head[9] * (double)ncells;
    }
    
    return;
}

void wpotclass::ReadOneWFullSub(const int iQQ, const int ikpt, int &idxbeg) {
/*
    Including the whole W_{GG'} stored in WFULLxxxx.tmp are read in
    to calculate [B^{kc}_{k'c'}(G)]^T x W_{GG'} x [B^{kv}_{k'v'}(G')]*
    where B^{kn}_{k'n'}(G) are the matrix with dimension (N_G x N_nN_n')
    the whole W_{GG'} are distributed stored among the processes grid
*/
    if(Qcs[iQQ].numq_of_ibrk[ikpt] == 0) { cerr << "ERROR in ReadOneWFullSub: vanishing size of numq_of_ibrk"; exit(1); }
    Qcs[iQQ].wfullggp = new complex<double>[(size_t)Qcs[iQQ].numq_of_ibrk[ikpt]
                                                  * Qcs[iQQ].numq_of_ibrk[ikpt]]();
    
    ifstream wfull;
    idxbeg = (ikpt > 0 ? accumulate(Qcs[iQQ].numq_of_ibrk, Qcs[iQQ].numq_of_ibrk + ikpt, 0) : 0);
    wfull.open(wpotdir + "/WFULL" + Int2Str(ikpt + 1, 4) + ".tmp", ios::in|ios::binary);
    if(!wfull.is_open()) { Cerr << "wpot file " << "WFULL" + Int2Str(ikpt + 1, 4) + ".tmp" << " doesn't find, please check."; exit(1); }
    complex<double> *cdtmp = new complex<double>[ (size_t)npwGW[ikpt] * npwGW[ikpt] + 1](); // "+1" for the last temporary zero
    wfull.seekg(  2 * sizeof(int) + 2 * sizeof(int) // NP, NP
                + 2 * sizeof(int) + 9 * sizeof(complex<double>) // head
                + 2 * sizeof(int) + 3 * npwGW[ikpt] * sizeof(complex<double>)  //  wing
                + 2 * sizeof(int) + 3 * npwGW[ikpt] * sizeof(complex<double>)  // cwing
                + sizeof(int), ios::beg ); // first "int" in body
    wfull.read((char*)cdtmp, sizeof(complex<double>) * npwGW[ikpt] * npwGW[ikpt]);
    wfull.close();
            
    for(size_t jcol = 0; jcol < Qcs[iQQ].numq_of_ibrk[ikpt]; jcol++) {
        if(Qcs[iQQ].gind[idxbeg + jcol] == npwGW[ikpt]) continue;
        size_t idxbeg_col = (size_t)Qcs[iQQ].gind[idxbeg + jcol] * npwGW[ikpt];
        complex<double> cdtmp_last = cdtmp[ idxbeg_col + npwGW[ikpt] ];
        cdtmp[ idxbeg_col + npwGW[ikpt] ] = 0.0; // temporary set to "0"
        Zgather(Qcs[iQQ].numq_of_ibrk[ikpt],
                cdtmp + idxbeg_col,
                Qcs[iQQ].wfullggp + jcol * Qcs[iQQ].numq_of_ibrk[ikpt],
                Qcs[iQQ].gind + idxbeg);
        #pragma omp parallel for
        for(size_t irow = 0; irow < Qcs[iQQ].numq_of_ibrk[ikpt]; irow++) {
            if( abs(Qcs[iQQ].time_reve_kfull[idxbeg + irow]) !=
                abs(Qcs[iQQ].time_reve_kfull[idxbeg + jcol]) ) {
                 Qcs[iQQ].wfullggp[irow + jcol * Qcs[iQQ].numq_of_ibrk[ikpt]] = 0.0;
                 continue;
            }
            if(Qcs[iQQ].time_reve_kfull[idxbeg + irow] < 0)
                 Qcs[iQQ].wfullggp[irow + jcol * Qcs[iQQ].numq_of_ibrk[ikpt]] =
            conj(Qcs[iQQ].wfullggp[irow + jcol * Qcs[iQQ].numq_of_ibrk[ikpt]]);
                 Qcs[iQQ].wfullggp[irow + jcol * Qcs[iQQ].numq_of_ibrk[ikpt]] *=
            conj(Qcs[iQQ].eigtau[idxbeg + irow]) * Qcs[iQQ].eigtau[idxbeg + jcol];
        }
        cdtmp[ idxbeg_col + npwGW[ikpt] ] = cdtmp_last;
    }

    if(iQQ == 0 && ikpt == 0) { // deal with head and wing/cwing later
        for(size_t irow = 0; irow < Qcs[0].numq_of_ibrk[0]; irow++) Qcs[iQQ].wfullggp[irow] = 0.0;
        for(size_t jcol = 0; jcol < Qcs[0].numq_of_ibrk[0]; jcol++) Qcs[iQQ].wfullggp[jcol * Qcs[0].numq_of_ibrk[0]] = 0.0;
    }

    delete[] cdtmp;
    
    return;
}

void wpotclass::DelOneWFull(const int iQQ) {
    delete[] Qcs[iQQ].wfullggp;
    return;
}

void wpotclass::ReadAllWFull() {
    isReadAllWFull = true;
    COUT << "Reading All WFULLxxxx.tmp ..." << endl;
    double tstart, tend;
    for(int iQQ = 0; iQQ < numQ; iQQ++) {
        tstart = omp_get_wtime();
        ReadOneWFull(iQQ);
        tend = omp_get_wtime();
        COUT << "Q = " << setw((int)log10(numQ) + 1) << setfill(' ') << iQQ + 1 << '/' << numQ << ":  " << setw(6) << setfill(' ') << fixed << setprecision(1) << tend - tstart << " s" << endl;
    }
    return;
}

wpotclass::wpotclass(waveclass *wvc_in):wvc(wvc_in) {
    for(int i = 0; i < 3; i++) {
        a_GW[i] = new double[3];
        b_GW[i] = new double[3];
    }
}
void wpotclass::Initial() {
    isInitial = true;
    for(int s = 0; s < 3; s++) { NK_SC[s] = nkptssc[s]; NK_GW[s] = nkptsgw[s]; }
    ReadGWOUTCAR((wpotdir + "/OUTCAR").c_str());
    if(!GetM()) { CERR << "WRONG in wpot.cpp::GetM" << endl; EXIT(1); }
    GetNG(); Getng();
    GetgidxGW();
    /*if(is_world_root) {
    cout << "nktotgw, nkibrgw = " << nktotgw << ' ' << nkibrgw << endl;
    cout << "emax_SCGW = " << emax_SCGW << endl;
    cout << "NG_SCGW = " << NG_SCGW[0] << ' ' << NG_SCGW[1] << ' ' << NG_SCGW[2] << endl;
    cout << "emax_GW = " << emax_GW << endl;
    cout << "ng_GW = " << ng_GW[0] << ' ' << ng_GW[1] << ' ' << ng_GW[2] << endl;
    }*/
    QpG2qpg();
    /*if(is_world_root) {
    for(int iQQ = 0; iQQ < numQ; iQQ++) {
        if(iQQ == 5) {
        cout << "iQQ = " << iQQ << ": " << Qcs[iQQ].npwG << endl;
        for(int iGG = 0; iGG < 100; iGG++) cout << Qcs[iQQ].qind[iGG] << ' ';
        cout << endl;
        for(int iGG = 0; iGG < 100; iGG++) cout << Qcs[iQQ].gind[iGG] << ' ';
        cout << endl; }
    }}*/
    
    Qcs[0].ReadHeadWing(omega0_GW, ncells, gabsGW);
    // GetgidxGW, free memory
    MPI_Win_free(&window_gidxRevGW);
    delete[] gidxRevGW;
    MPI_Win_free(&window_gabsGW);
    /*if(is_world_root) {
    cout << "head = " << Qcs[0].head[9] << endl;
    cout << " wing: "; for(int i = 0; i < 50; i++) cout << Qcs[0].wing[i] << ' '; cout << endl;
    cout << "cwing: "; for(int i = 0; i < 50; i++) cout << Qcs[0].cwing[i] << ' '; cout << endl;
    cout << "qind: "; for(int i = 0; i < 200; i++) cout << Qcs[0].qind[i] << ' '; cout << endl;
    cout << "gind: "; for(int i = 0; i < 200; i++) cout << Qcs[0].gind[i] << ' '; cout << endl; } */
    if(!iswfull) ReadWxxxx();
    //ReadAllWFull();
    
    return;
}

wpotclass::~wpotclass() {
    for(int i = 0; i < 3; i++) { delete[] a_GW[i]; delete[] b_GW[i]; }
    if(isInitial) {
        // ReadGWOUTCAR
        for(int ii = 0; ii < nktotgw; ii++) {
            delete[] allgwkpts[ii]; 
            delete[] gwkpgr2ibz[ii];
        }
        delete[] allgwkpts; delete[] gwkpinibz; delete[] gwkp_trev;
        delete[] gwkpgr2ibz; delete[] gwkpoptno;
        // GetgidxGW
        MPI_Win_free(&window_npwGW);
    }
    if(isReadWxxxx)    { MPI_Win_free(&window_wgg); }
    if(isReadAllWFull) { for(int iQQ = 0; iQQ < numQ; iQQ++) delete[] Qcs[iQQ].wfullggp; }
    if(isInitial) { 
        delete[] Qcs; // QpG2qpg
        delete[] optcs; // ReadGWOUTCAR
        //QpG2qpg
        MPI_Win_free(&window_nqofibr);
        MPI_Win_free(&window_Gind); MPI_Win_free(&window_qind); MPI_Win_free(&window_gind);
        MPI_Win_free(&window_eigtau); MPI_Win_free(&window_trk);
    }
}
