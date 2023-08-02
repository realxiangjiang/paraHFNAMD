#include "tdcft.h"

void InterpolationForTDMat(const int ntimes, const double time_interval,
                           const char *mat_dir_name,
                           const long istart, const long nelements,
                           const bool is_istart_suffix,
                           const bool is_add_extra_splinecoeff) {
/*
    time-dependent matrix are given at time nodes of 
        t1, t2, ..., tN 
    or 
        (t1 + t2) / 2, (t2 + t3) / 2, ..., (tN-1 + tN) / 2
    the routine interpolates all time between t1 ~ tN or (t1 + t2) / 2 ~ (tN-1 + tN) / 2
    use cubic spline approach

    ntimes:            number of time nodes
    mat_dir_name:      matrix are stored in mat_dir_name/xxxx
    istart, nelements: the target data for interpolation are [istart, istart + nelements)
                       and if the matrix is complex, istart and nelements should be double

    usually, is_istart_suffix and is_add_extra_splinecoeff can't be both true,
    because the former one is setting for time-local diagonal elements
    while the latter one is setting for cross-time nac
*/
    if(nelements <= 0) return;
    if(!is_istart_suffix) { // no suffix, means need judge if there is existing interpolation
        int have_not_interpolate = access((string(mat_dir_name) + "c0123").c_str(), F_OK);
        if(!have_not_interpolate) {
            cout << "Interpolation in  " << std::left << setw(namddir.size() + 14) << setfill(' ')
                 << mat_dir_name << " exists" << endl;
            return;
        }
    } // only works in world_root 
    double tstart = omp_get_wtime();

    double *matdata = new double[nelements * ntimes]; // column-major
    ifstream matfile;
    for(int itime = 0; itime < ntimes; itime++) {
        matfile.open((mat_dir_name + Int2Str(itime + 1)).c_str(), ios::in|ios::binary);
        if(!matfile.is_open()) { cerr << "ERROR: " << mat_dir_name + Int2Str(itime + 1) << " doesn't open when interpolating" << endl; exit(1); }
        matfile.seekg(sizeof(double) * istart, ios::beg);
        matfile.read((char*)(matdata + nelements * itime), sizeof(double) * nelements);
        matfile.close();
    }

    int status;
    double *spline_coeff = NULL; // this will malloc in SplineInterpolationPre function
    DFTaskPtr task;              // Data Fitting operations are task based
    double time_beg_end[2] = {0.0, time_interval * (ntimes - 1)};
    SplineInterpolationPre(task, spline_coeff, ntimes, nelements, time_beg_end, matdata, DF_MATRIX_STORAGE_COLS);
    // spline_coeff can be regard as shape of [nelements, ntimes - 1, 4] or [nelements, (ntimes - 1) x 4] with row-major
    // also can be regard as shape of [(ntimes - 1) x 4, nelements] with column-major
    double *all_splcoeff = NULL;
    if(is_add_extra_splinecoeff) {
        Dimatcopy("CblasRowMajor", "CblasTrans", nelements, (ntimes - 1) * 4,
                  1.0, spline_coeff, (ntimes - 1) * 4, nelements);
        // after the in-place transpose, the shape of spline_coeff is [nelements, (ntimes - 1) x 4] with column-major
        all_splcoeff = new double[(size_t)nelements * (ntimes + 1) * 4]();
        Dcopy((size_t)nelements * (ntimes - 1) * 4, 
               spline_coeff, 1, all_splcoeff + nelements * 4, 1); // skip first 4 columns to copy
        AddExtraSplineCoeff(nelements, time_interval,
                            matdata, matdata + (nelements * (ntimes - 1)),
                            all_splcoeff, all_splcoeff + nelements,
                            all_splcoeff + (size_t)nelements * (4 * ntimes), 
                            all_splcoeff + (size_t)nelements * (4 * ntimes + 1));
        Dimatcopy("CblasColMajor", "CblasTrans", nelements, (ntimes + 1) * 4,
                  1.0, all_splcoeff, nelements, (ntimes + 1) * 4);
        // now shape of all_splcoeff is [(ntimes + 1) * 4, nelements] with column-major
    }
    // if w/o suffix, means no need merging for different "istart" segment, do a transpose here
    if(!is_istart_suffix) // is_add_extra_splinecoeff = false
    Dimatcopy("CblasRowMajor", "CblasTrans", nelements, (ntimes - 1) * 4,
              1.0, spline_coeff, (ntimes - 1) * 4, nelements);
    // after transpose, the shape is regard as [nelements, (ntimes - 1) x 4] with column-major
    
    // write to file
    string otfname = string(mat_dir_name) + "c0123" + (is_istart_suffix ? '_' + to_string(istart) : "");
    ofstream cfile(otfname.c_str(), ios::out|ios::binary);
    if(!cfile.is_open()) { cerr << "ERROR: " << otfname << " can't open when output interpolation coefficients" << endl; exit(1); }
    if(is_add_extra_splinecoeff) cfile.write((char*)all_splcoeff, sizeof(double) * nelements * (ntimes + 1) * 4);
    else                         cfile.write((char*)spline_coeff, sizeof(double) * nelements * (ntimes - 1) * 4);
    cfile.close();
    
    status = dfDeleteTask(&task); delete[] spline_coeff;
    if(is_add_extra_splinecoeff) delete[] all_splcoeff;
    delete[] matdata;
    double tend = omp_get_wtime();
    if(!is_istart_suffix) {
        cout << "Cubic Spline Interpolation in  "
             << std::left << setw(namddir.size() + 13) << setfill(' ') 
             << mat_dir_name << "  finished, used "
             << std::right << setw(8) << setfill(' ') << (int)(tend - tstart) << " s." << endl;
        cout.copyfmt(iosDefaultState);
    }

    return;
}

void AddExtraSplineCoeff(const int ntimes, const size_t totsize,
                         const double time_interval, const char *dirname) {
    /* deprecated */
    ifstream inf;
    ofstream otf;
    complex<double> *matdata = new complex<double>[totsize]();
    int extratime[2] = {0, ntimes};  // the spline coeff for [1, ntimes - 1] exist, need add for "0" and "ntimes"
    for(int it = 0; it < 2; it++) {  // first write vanishing spline coeff
        for(int ci = 2; ci < 4; ci++) { // c2 = 0 and c3 = 0
            otf.open((dirname + Int2Str(extratime[it]) + "_c" + to_string(ci)).c_str(), ios::out|ios::binary);
            assert(otf.is_open());
            otf.write((char*)matdata, sizeof(complex<double>) * totsize);
            otf.close();
        }
    }

    // write left endpoint extra data
    inf.open((dirname + Int2Str(1)).c_str(), ios::in|ios::binary); assert(inf.is_open());
    inf.read((char*)matdata, sizeof(complex<double>) * totsize); inf.close();
    ZDscal(totsize, -1.0, matdata, 1); // nac[0] = - nac[1]
    otf.open((dirname + Int2Str(0)).c_str(), ios::out|ios::binary); assert(otf.is_open());
    otf.write((char*)matdata, sizeof(complex<double>) * totsize); otf.close();
    /* Y = c0 + c1 x X ==> c0 = Y(X = 0) */
    otf.open((dirname + Int2Str(0) + "_c0").c_str(), ios::out|ios::binary); assert(otf.is_open());
    otf.write((char*)matdata, sizeof(complex<double>) * totsize); otf.close();
    /* c0 + c1 * T = Y(T) = -c0 ==> c1 = (-2.0 / T) x c0 */
    ZDscal(totsize, - 2.0 / time_interval, matdata, 1);
    otf.open((dirname + Int2Str(0) + "_c1").c_str(), ios::out|ios::binary); assert(otf.is_open());
    otf.write((char*)matdata, sizeof(complex<double>) * totsize); otf.close();
    
    // write left endpoint extra data
    // matdata at ntimes exsits, but spline coeff doesn't
    inf.open((dirname + Int2Str(ntimes)).c_str(), ios::in|ios::binary); assert(inf.is_open());
    inf.read((char*)matdata, sizeof(complex<double>) * totsize); inf.close();
    /* Y = c0 + c1 x X ==> c0 = Y(X = 0) */
    otf.open((dirname + Int2Str(ntimes) + "_c0").c_str(), ios::out|ios::binary); assert(otf.is_open());
    otf.write((char*)matdata, sizeof(complex<double>) * totsize); otf.close();
    ZDscal(totsize, -1.0, matdata, 1); // nac[ntimes + 1] = - nac[ntimes]
    // this write will cover the nac(ntimes + 1, 1) into -nac(ntimes, ntimes + 1)
    otf.open((dirname + Int2Str(ntimes + 1)).c_str(), ios::out|ios::binary); assert(otf.is_open());
    otf.write((char*)matdata, sizeof(complex<double>) * totsize); otf.close();
    /* c0 + c1 * T = Y(T) = -Y(T) + c1 * T ==> c1 = (2.0 / T) x Y(T) */
    ZDscal(totsize, 2.0 / time_interval, matdata, 1);
    otf.open((dirname + Int2Str(ntimes) + "_c1").c_str(), ios::out|ios::binary); assert(otf.is_open());
    otf.write((char*)matdata, sizeof(complex<double>) * totsize); otf.close();

    delete[] matdata;
    return;
}

void AddExtraSplineCoeff(const size_t totsize, const double time_interval,
                         const double *first, const double *last,
                         double *extra_first_c0, double *extra_first_c1,
                         double *extra_last_c0,  double *extra_last_c1) {
/*
    used for NAC, when begtime + namdtim > totstru, need do bounce-loop
    below is an example with totstru = 10 (ntimes = 9, ntimes - 1 = 8 spline coeff)

                begin                            end 
                  |                               |
                  V                               V
    ... 3   2   1   2   3   4   5   6   7   8   9   10   9   8   7 ...
              ^                                        ^
              |                                        |
            extra                                    extra
*/
    /* Y = c0 + c1 x X ==> c0 = Y(X = 0) */
    Dcopy(totsize, first, 1, extra_first_c0, 1); // nac[1]
    Dscal(totsize, -1.0, extra_first_c0, 1);     // nac[0] = - nac[1], c0 = Y(X = 0)

    /* c0 + c1 * T = Y(T) = -c0 ==> c1 = (-2.0 / T) x c0 */
    Dcopy(totsize, extra_first_c0, 1, extra_first_c1, 1);
    Dscal(totsize, - 2.0 / time_interval, extra_first_c1, 1);
    
    /* Y = c0 + c1 x X ==> c0 = Y(X = 0) */
    Dcopy(totsize, last, 1, extra_last_c0, 1); // nac[ntimes], c0 = Y(X = 0)
    /* c0 + c1 * T = Y(T) = -Y(T) + c1 * T ==> c1 = (2.0 / T) x Y(T) */
    /* c0 + c1 * T = Y(T) = -c0 ==> c1 = (-2.0 / T) x c0 */
    Dcopy(totsize, extra_last_c0, 1, extra_last_c1, 1);
    Dscal(totsize, - 2.0 / time_interval, extra_last_c1, 1);

    return;
}

void CombineAllSplineCoeff(const int ntimes, const int dim_row, const int dim_col,
                           const char *dirname,
                           const int num_of_matrix, const bool is_add_extra_splinecoeff) {
/*
    the shape of interpolated matrix is [dim_row x dim_col]
    the matrix elements are interpolated column by column
    total "num_of_matrix" matrix will be interpolated
*/
    int have_not_interpolate = 1;
    if(is_world_root) {
        have_not_interpolate = access((string(dirname) + "c0123").c_str(), F_OK);
        if(!have_not_interpolate)
        cout << "Interpolation in  " << std::left << setw(namddir.size() + 14) << setfill(' ')
             << dirname << " exists" << endl;
    }
    MPI_Bcast(&have_not_interpolate, 1, MPI_INT, world_root, world_comm);
    if(!have_not_interpolate) return;
    
    double tstart, tend;
    tstart = omp_get_wtime();
    // First interpolate culumn by column
    for(int imat = 0; imat < num_of_matrix; imat++) {
        for(int icol = world_rk; icol < dim_col; icol += world_sz) {
            InterpolationForTDMat(ntimes, iontime, dirname,                           // "2": because of complex matrix
                                  (size_t)imat * dim_row * dim_col * 2 + icol * dim_row * 2, dim_row * 2,
                                  true, is_add_extra_splinecoeff); 
        }
    }
    // Then combine time by time
    MPI_Barrier(world_comm);
    int memoryOK = 1;
    const size_t segmentsize = (size_t)(is_add_extra_splinecoeff ? ntimes + 1 : ntimes - 1) * 4 * dim_row;
    complex<double> *cdtmp = NULL;
    if(is_world_root) {
        try { cdtmp = new complex<double>[(size_t)num_of_matrix * dim_col * segmentsize]; }
        catch(bad_alloc) {
            memoryOK = 0;
        }
    }
    MPI_Barrier(world_comm);
    MPI_Bcast(&memoryOK, 1, MPI_INT, world_root, world_comm);
    if(memoryOK) {
        if(is_world_root) {
            ifstream inf;
            for(int imat = 0; imat < num_of_matrix; imat++) for(int icol = 0; icol < dim_col; icol++) {
                string infname = ( string(dirname) + "c0123"
                                 + '_' + to_string((long)imat * dim_row * dim_col * 2 + icol * dim_row * 2) );
                inf.open(infname.c_str(), ios::in|ios::binary);
                if(!inf.is_open()) { cerr << "ERROR: " << infname << " can't open to read when merging interpolation coefficients" << endl; exit(1); }
                inf.read((char*)(cdtmp + ((size_t)imat * dim_col + icol) * segmentsize),
                         sizeof(complex<double>) * segmentsize);
                inf.close();
                remove(infname.c_str());
            }
            const int nrowtmp = (is_add_extra_splinecoeff ? (ntimes + 1) * 4 : (ntimes - 1) * 4);
            const int ncoltmp = num_of_matrix * dim_row * dim_col;
            Dimatcopy("CblasColMajor", "CblasTrans", nrowtmp, ncoltmp * 2,
                       1.0, (double*)cdtmp, nrowtmp, ncoltmp * 2);
            // MUST use Dimatcopy, because interpolation is respectively done by real and image part
            // after transpose, the shape of spline_coeff is
            // [num_matrix x matrix.size, (ntimes +/- 1) x 4] with column-major
            // the matrix data is successive in one time and also with column-major
            string otfname = string(dirname) + "c0123";
            ofstream otf(otfname.c_str(), ios::out|ios::binary);
            if(!otf.is_open()) { cerr << "ERROR: " << otfname << " can't open to write when merging interpolation coefficients" << endl; exit(1); }
            otf.write((char*)cdtmp, sizeof(complex<double>) * ncoltmp * nrowtmp);
            delete[] cdtmp;
            otf.close();
        }
        MPI_Barrier(world_comm);
    }
    else {
        const int nrowtmp = (is_add_extra_splinecoeff ? (ntimes + 1) * 4 : (ntimes - 1) * 4);
        const int blkncol = dim_row;
        const int ncoltmp = num_of_matrix * dim_row * dim_col;
        fstream iof;
        cdtmp = new complex<double>[segmentsize];
        // First, transpose all c0123_*
        for(int imat = 0; imat < num_of_matrix; imat++) for(int icol = world_rk; icol < dim_col; icol += world_sz) {
            string fname = ( string(dirname) + "c0123"
                           + '_' + to_string((long)imat * dim_row * dim_col * 2 + icol * dim_row * 2) );
            iof.open(fname.c_str(), ios::in|ios::out|ios::binary);
            if(!iof.is_open()) { cerr << "ERROR2: " << fname << " can't open to read when merging interpolation coefficients" << endl; exit(1); }
            // read in one block, can be regarded as [nrowtmp, blkncol] with column-major
            iof.read((char*)cdtmp, sizeof(complex<double>) * segmentsize);
            Dimatcopy("CblasColMajor", "CblasTrans", nrowtmp, blkncol * 2,
                       1.0, (double*)cdtmp, nrowtmp, blkncol * 2);
            // Use Dimatcopy to transpose, result matrix is
            // row-major-[nrowtmp, blkncol] or column-major-[blkncol, nrowtmp]
            // complex<double> matrix
            iof.seekp(0, ios::beg);
            iof.write((char*)cdtmp, sizeof(complex<double>) * segmentsize);
            iof.close();
        }
        MPI_Barrier(world_comm);
        // Second, merge all c0123_* row by row, store to c0123_r*
        for(int irow = world_rk; irow < nrowtmp; irow += world_sz) {
            ofstream otf( (string(dirname) + "c0123_r" + to_string(irow)).c_str(), ios::out|ios::binary );
            assert(otf.is_open());
            ifstream inf;
            for(int imat = 0; imat < num_of_matrix; imat++) for(int icol = 0; icol < dim_col; icol++) {
                inf.open( ( string(dirname) + "c0123_"
                          + to_string((long)imat * dim_row * dim_col * 2 + icol * dim_row * 2)).c_str(),
                          ios::in|ios::binary );
                assert(inf.is_open());
                inf.seekg(sizeof(complex<double>) * irow * blkncol, ios::beg);
                inf.read((char*)cdtmp, sizeof(complex<double>) * blkncol);
                inf.close();
                otf.write((char*)cdtmp, sizeof(complex<double>) * blkncol);
            }
            otf.close();
        }
        delete[] cdtmp;
        MPI_Barrier(world_comm);
        for(int imat = 0; imat < num_of_matrix; imat++) for(int icol = world_rk; icol < dim_col; icol += world_sz) {
            remove(( string(dirname) + "c0123"
                                     + '_' + to_string((long)imat * dim_row * dim_col * 2 + icol * dim_row * 2) ).c_str());
        }
        MPI_Barrier(world_comm);
        // Third, merge all c0123_r*
        if(is_world_root) {
            size_t one_row_size = (size_t)num_of_matrix * dim_col * dim_row;
            cdtmp = new complex<double>[one_row_size];
            ofstream otf((string(dirname) + "c0123").c_str(), ios::out|ios::binary);
            assert(otf.is_open());
            ifstream inf;
            for(int irow = 0; irow < nrowtmp; irow++) {
                string infname = string(dirname) + "c0123_r" + to_string(irow);
                inf.open( infname.c_str(), ios::in|ios::binary );
                assert(inf.is_open());
                inf.read((char*)cdtmp, sizeof(complex<double>) * one_row_size);
                inf.close();
                remove(infname.c_str());
                otf.write((char*)cdtmp, sizeof(complex<double>) * one_row_size);
            }
            delete[] cdtmp;
            otf.close();
        }
        MPI_Barrier(world_comm);
    }
    MPI_Barrier(world_comm);
    tend = omp_get_wtime();
    COUT << "Cubic Spline Interpolation in  "
         << std::left << setw(namddir.size() + 13) << setfill(' ') << dirname << "  finished, used "
         << std::right << setw(8) << setfill(' ') << (int)(tend - tstart) << " s." << endl; cout.copyfmt(iosDefaultState);
    MPI_Barrier(world_comm);
    return;
}

void BuildAllTDMatSplineCoeff() {
    int nspns, nkpts, dimC, dimV;
    ReadInfoTmp(nspns, nkpts, dimC, dimV);
    const int nbnds = dimC + dimV; // for electron/hole, dimC or dimV = 0
    const int dim = numspns * nkpts * dimC * dimV;
    if(carrier == "electron" || carrier == "hole") { 
        if(is_world_root) {
            // energy
            InterpolationForTDMat(totstru, iontime, (namddir + "/tmpEnergy/").c_str(), 0,
                                  nspns * nkpts * nbnds, false, false);
                                //this MUST be nspns which equals to totdiffspns calculted in wave_base.cpp
        }
        MPI_Barrier(world_comm);
        CombineAllSplineCoeff(totstru - 1, nbnds, nbnds,
                              (namddir + "/tmpNAC/").c_str(), nspns * nkpts, true);
    }
    else if(carrier == "exciton") {
        if(is_world_root) {
            // energy difference                                                 should be "dim" here
            InterpolationForTDMat(totstru, iontime, (namddir + "/tmpDiagonal/").c_str(), 0, dim, false, false);
        }
        MPI_Barrier(world_comm);
        CombineAllSplineCoeff(totstru - 1, dimC, dimC, (namddir + "/tmpCBNAC/").c_str(), nspns * nkpts, true);
        CombineAllSplineCoeff(totstru - 1, dimV, dimV, (namddir + "/tmpVBNAC/").c_str(), nspns * nkpts, true);
        CombineAllSplineCoeff(totstru - 1, dimC, dimV, (namddir + "/tmpC2VNAC/").c_str(), nspns * nkpts, true);
        if(is_bse_calc) {
            CombineAllSplineCoeff(totstru, dim, dim, (namddir + "/tmpDirect/").c_str());
            CombineAllSplineCoeff(totstru, dim, dim, (namddir + "/tmpExchange/").c_str());
        }
    }
    
    MPI_Barrier(world_comm);
    return;
}

void Mat_AtoOmega(const int matsize, const double h, // h: time interval, NOT Planck constant
                  const complex<double> *a0, const complex<double> *a1,
                  const complex<double> *a2, const complex<double> *a3, // input a0~a3
                  complex<double> *om_oih) {                            // output omega / ih
/*
    Reference: BIT Numerical Mathematics 40, 434–450 (2000)
    the coefficients evolution depends on the time-evolution operator 
         
            X(t, t0) = e^Omega
    where
            Omega = Om1 + Om2 + Om34
    and 
            Om1  = hB0
            Om2 = h^2[B1, (3/2)B0 - 6B2]
            Om34 = h^2[B0, [B0, (1/2)hB2 - (1/60)Om2]] + (3/5)h[B1, Om2]

            B0 = a0 + (1/12) h^2 a0
            B1 = (1/12) h a1 + (1/80) h^3 a3
            B2 = (1/12)a0 + (1/80) h^2 a2
    
    the output is omega / (ih) 
    in order to get an appreciable Hermitian matrix
*/
    const double h2 = h  * h;
    const double h3 = h2 * h;
    const int nrows_loc = Numroc(matsize, MB_ROW, myprow_group, nprow_group);
    const int ncols_loc = Numroc(matsize, NB_COL, mypcol_group, npcol_group);
    const size_t mn = (size_t)matsize * matsize;
    const size_t mn_loc = (size_t)nrows_loc * ncols_loc;
    complex<double> *B0 = new complex<double>[max(mn_loc, (size_t)1)]();
    complex<double> *B1 = new complex<double>[max(mn_loc, (size_t)1)]();
    complex<double> *B2 = new complex<double>[max(mn_loc, (size_t)1)]();
    Zaxpby_o(mn_loc, 1.0,            a0, 1.0 / 12.0 * h2, a2, B0);
    Zaxpby_o(mn_loc, 1.0 / 12.0 * h, a1, 1.0 / 80.0 * h3, a3, B1);
    Zaxpby_o(mn_loc, 1.0 / 12.0,     a0, 1.0 / 80.0 * h2, a2, B2);
    MPI_Barrier(group_comm);
    
    complex<double> *Om1  = new complex<double>[max(mn_loc, (size_t)1)]();
    complex<double> *Om2  = new complex<double>[max(mn_loc, (size_t)1)]();
    complex<double> *Om34 = new complex<double>[max(mn_loc, (size_t)1)]();
    complex<double> *cdtmp  = new complex<double>[max(mn_loc, (size_t)1)]();
    complex<double> *cdtmp2 = new complex<double>[max(mn_loc, (size_t)1)]();
    Zomatcopy("CblasColMajor", "CblasNoTrans", nrows_loc, ncols_loc, 
               h, B0, nrows_loc, Om1, nrows_loc); // Om1
    Zaxpby_o(mn_loc, 3.0 / 2.0, B0, -6.0, B2, cdtmp); // (3/2)B0 - 6B2
    MPI_Barrier(group_comm);
    Pzgecom(matsize, nrows_loc, h2, B1, cdtmp, Om2);  // Om2 = h^2[B1, (3/2)B0 - 6B2]

    Zaxpby_o(mn_loc, 1.0 / 2.0 * h, B2, -1.0 / 60.0, Om2, cdtmp); // cdtmp = (1/2)hB0 - (1/60)Om2
    MPI_Barrier(group_comm);
    Pzgecom(matsize, nrows_loc, 1.0, B0, cdtmp,  cdtmp2); // cdtmp2 = [B0, (1/2)hB0 - (1/60)Om2]
    Pzgecom(matsize, nrows_loc,  h2, B0, cdtmp2, cdtmp);  // cdtmp = h^2[B0, cdtmp2]
    Pzgecom(matsize, nrows_loc, 3.0 / 5.0 * h, B1, Om2, cdtmp2); // cdtmp2 = (3/5)h[B1, Om2]
    Zaxpby_o(mn_loc, 1.0, cdtmp, 1.0, cdtmp2, Om34);      // Om34 = cdtmp + cdtmp2
    Zaxpby_o(mn_loc, 1.0 / (iu_d * h), Om1, 1.0 / (iu_d * h), Om2, cdtmp); // cdtmp = (Om1 + Om2) / (ih)
    Zaxpby_o(mn_loc, 1.0, cdtmp, 1.0 / (iu_d * h), Om34, om_oih); // om_oih = (Om1 + Om2 + Om34) / (ih)
    MPI_Barrier(group_comm);

    delete[] B0; delete[] B1; delete[] B2;
    delete[] Om1; delete[] Om2; delete[] Om34;
    delete[] cdtmp; delete[] cdtmp2;
    
    return;
}

void ReadDiag(const char *filename, const int itime,
              vector<int> &allispns, vector<int> &allikpts, 
              const int *ibndstart, const int nbnds,
              const int totnspns, const int totnkpts, const int totnbnds,
              const complex<double> alpha, complex<double> *locmat) {
    /* 
        fulldiag += alpha * read_in_data
        [bndstart, bndstart + nbnds) is included in [0, totnbnds)
    */
    ifstream inf(filename, ios::in|ios::binary);
    if(!inf.is_open()) { cerr << "ERROR: " << filename << " can't open in tdcft.cpp::ReadDiag" << endl; exit(1); }
    double *diagdata = new double[nbnds];
    
    const int nspns = allispns.size();
    const int nkpts = allikpts.size();
    const int dim = nspns * nkpts * nbnds;
    const int ndim_loc_row = Numroc(dim, MB_ROW, myprow_group, nprow_group);
    int ispn, ikpt;
    int ii_loc_row, jj_loc_col;
    int iprow, jpcol;
    for(int is = 0; is < nspns; is++) {
        ispn = allispns[is];
        for(int ik = 0; ik < nkpts; ik++) {
            ikpt = allikpts[ik];
            inf.seekg(sizeof(double) * ( (size_t)itime * totnspns * totnkpts * totnbnds +
                                         ispn * totnkpts * totnbnds + ikpt * totnbnds + ibndstart[ispn] ), ios::beg);
            inf.read((char*)diagdata, sizeof(double) * nbnds);
            
            #pragma omp parallel for private(ii_loc_row, jj_loc_col, iprow, jpcol)
            for(int ib = 0; ib < nbnds; ib++) {
                BlacsIdxglb2loc(is * nkpts * nbnds + ik * nbnds + ib,
                                iprow, ii_loc_row, 0, dim, MB_ROW, nprow_group);
                BlacsIdxglb2loc(is * nkpts * nbnds + ik * nbnds + ib,
                                jpcol, jj_loc_col, 0, dim, NB_COL, npcol_group);
                if(myprow_group == iprow && mypcol_group == jpcol)
                locmat[ii_loc_row + jj_loc_col * ndim_loc_row] += alpha * diagdata[ib];
            }
        }
    }
    
    inf.close();
    delete[] diagdata;
    return;
}

void ReadNAC(const char *filename, const int itime,
             vector<int> &allispns, vector<int> &allikpts, 
             const int *ibndstart, const int nbnds,
             const int totnspns, const int totnkpts, const int totnbnds,
             const complex<double> alpha, complex<double> *locmat, const bool isConj) {
/*
    for single particle, electron or hole
    [bndstart, bndstart + nbnds) is included in [0, totnbnds)
*/
    ifstream inf(filename, ios::in|ios::binary);
    if(!inf.is_open()) { cerr << "ERROR: " << filename << " can't open in tdcft.cpp::ReadNAC" << endl; exit(1); }
    
    const int nspns = allispns.size();
    const int nkpts = allikpts.size();
    const int dim = nspns * nkpts * nbnds;
    const int ndim_loc_row = Numroc(dim, MB_ROW, myprow_group, nprow_group);
    int ispn, ikpt;
    int iprow, jpcol;
    int ii, jj, ii_loc_row, jj_loc_col;
    complex<double> cdtmp;
    for(int is = 0; is < nspns; is++) {
        ispn = allispns[is];
        for(int ik = 0; ik < nkpts; ik++) {
            ikpt = allikpts[ik];
            for(int iijj = 0; iijj < nbnds * nbnds; iijj++) {
                ii = iijj % nbnds; // row
                jj = iijj / nbnds; // col
                BlacsIdxglb2loc(is * nkpts * nbnds + ik * nbnds + ii,
                                iprow, ii_loc_row, 0, dim, MB_ROW, nprow_group);
                BlacsIdxglb2loc(is * nkpts * nbnds + ik * nbnds + jj,
                                jpcol, jj_loc_col, 0, dim, NB_COL, npcol_group);
                if(myprow_group != iprow || mypcol_group != jpcol) continue;
                inf.seekg(sizeof(complex<double>) * ( (size_t)itime * totnspns * totnkpts * totnbnds * totnbnds
                                                    + (size_t)ispn * totnkpts * totnbnds * totnbnds
                                                                       + ikpt * totnbnds * totnbnds
                                                    + ((ii + ibndstart[ispn]) + 
                                                       (jj + ibndstart[ispn]) * totnbnds) ), ios::beg);
                inf.read((char*)&cdtmp, sizeof(complex<double>));
                locmat[ii_loc_row + jj_loc_col * ndim_loc_row] += alpha * (isConj ? conj(cdtmp) : cdtmp);
            }
        }
    }

    inf.close();
    MPI_Barrier(group_comm);
    return;
}

void ReadNACbySK(const char *ccfile, const char *vvfile,
                 const int itime, const int start_idx,
                 const int nspns, const int nkpts, const int dimC, const int dimV,
                 const complex<double> alpha, complex<double> *locmat) {
/* 
    (dimC x dimV)^2 for each spn-kpt (total nspns x nkpts matrix block)
    (nspns x nkpts x dimC x dimV)^2 is the dimension of fullmat, where locmat is the distributed storage of fullmat
*/
    ifstream infcc(ccfile, ios::in|ios::binary);
    if(!infcc.is_open()) { cerr << "ERROR: " << ccfile << " can't open in tdcft.cpp::ReadNACbySK" << endl; exit(1); }
    ifstream infvv(vvfile, ios::in|ios::binary);
    if(!infvv.is_open()) { cerr << "ERROR: " << vvfile << " can't open in tdcft.cpp::ReadNACbySK" << endl; exit(1); }
    const int dimCC = dimC * dimC;
    const int dimVV = dimV * dimV;
    const int dimCV = dimC * dimV;
    const int dim = nspns * nkpts * dimCV + start_idx;
    const int dim_loc = Numroc(dim, MB_ROW, myprow_group, nprow_group);
    int icv, jcv, iprow, jpcol, ii_loc_row, jj_loc_col;
    int ic, iv, jc, jv;
    complex<double> cdtmp;
    for(int is = 0; is < nspns; is++)
    for(int ik = 0; ik < nkpts; ik++) {
        for(int ijcv = 0; ijcv < dimCV * dimCV; ijcv++) {
            icv = ijcv % dimCV; // row
            jcv = ijcv / dimCV; // col
            BlacsIdxglb2loc((is * nkpts + ik) * dimCV + icv + start_idx, iprow, ii_loc_row, 0, dim, MB_ROW, nprow_group);
            BlacsIdxglb2loc((is * nkpts + ik) * dimCV + jcv + start_idx, jpcol, jj_loc_col, 0, dim, NB_COL, npcol_group);
            if(myprow_group != iprow || mypcol_group != jpcol) continue;
            ic = icv / dimV; iv = icv % dimV; jc = jcv / dimV; jv = jcv % dimV;
            if(ic == jc && iv != jv) {
                infvv.seekg(sizeof(complex<double>) * ( (size_t)itime * totdiffspns * nkpts * dimVV
                                                      + (size_t)(min(is, totdiffspns - 1) * nkpts + ik) * dimVV 
                                                      + (iv + jv * dimV) ), ios::beg);
                infvv.read((char*)&cdtmp, sizeof(complex<double>));
                locmat[ii_loc_row + jj_loc_col * dim_loc] += alpha * conj(cdtmp);
            }
            if(iv == jv && ic != jc) {
                infcc.seekg(sizeof(complex<double>) * ( (size_t)itime * totdiffspns * nkpts * dimCC
                                                      + (size_t)(min(is, totdiffspns - 1) * nkpts + ik) * dimCC 
                                                      + (ic + jc * dimC) ), ios::beg);
                infcc.read((char*)&cdtmp, sizeof(complex<double>));
                locmat[ii_loc_row + jj_loc_col * dim_loc] += alpha * cdtmp;
            }
        }
    }
    infcc.close(); infvv.close();
    MPI_Barrier(group_comm);
    return;
}

void ReadCtoVNAC(const char *filename, const int itime,
                 const int nspns, const int nkpts, const int dimC, const int dimV,
                 const complex<double> alpha, complex<double> *locmat) {
    ifstream inf(filename, ios::in|ios::binary);
    if(!inf.is_open()) { cerr << "ERROR: " << filename << " can't open in tdcft.cpp::ReadCtoVNAC" << endl; exit(1); }
    const int dimCV = dimC * dimV;
    const int start_idx = 1; // MUST "1"
    const int dim = nspns * nkpts * dimCV + start_idx;
    const int dim_loc = Numroc(dim, MB_ROW, myprow_group, nprow_group);
    int iprow, jpcol, ii_loc_row, jj_loc_col;
    int ic, iv;
    complex<double> cdtmp;
    for(int is = 0; is < nspns; is++)
    for(int ik = 0; ik < nkpts; ik++) {
        for(int icv = 0; icv < dimCV; icv++) {
            BlacsIdxglb2loc((is * nkpts + ik) * dimCV + icv + start_idx, iprow, ii_loc_row, 0, dim, MB_ROW, nprow_group);
            BlacsIdxglb2loc((is * nkpts + ik) * dimCV + icv + start_idx, jpcol, jj_loc_col, 0, dim, NB_COL, npcol_group);
            if(myprow_group != 0 && mypcol_group != 0) continue;
            ic = icv / dimV; iv = icv % dimV;
            inf.seekg(sizeof(complex<double>) * ( (size_t)itime * totdiffspns * nkpts * dimCV
                                                + (size_t)(min(is, totdiffspns - 1) * nkpts + ik) * dimCV 
                                                + (ic + iv * dimC) ), ios::beg);
            inf.read((char*)&cdtmp, sizeof(complex<double>));
            if(myprow_group == iprow && mypcol_group == 0) { // first column must in first process column
                locmat[ii_loc_row] += alpha * cdtmp;
            }
            if(mypcol_group == jpcol && myprow_group == 0) { // first row must in first process row
                locmat[jj_loc_col * dim_loc] += alpha * ( - conj(cdtmp) );
            }
        }
    }
    inf.close();
    MPI_Barrier(group_comm);
    return;
}

void ReadOnsiteC(const int t_ion, const int matsize,
                 const int nspns, const int nkpts, const int dimC, const int dimV,
                 vector<int> &allispns, vector<int> &allikpts, const int *ibndstart,
                 const int totnspns, const int totnkpts, const int totnbnds,
                 complex<double> **c_onsite) { // output
/*  t_ion: time of ion step, at least 2 [2, totstru - 1], NOT include "1" and "totstru" */
/*  update: t_ion belongs to [1, +oo), when > totstru, need do bounce-back process*/
    int t_stru = (t_ion - 1) % (2 * (totstru - 1)) + 1;          // loop order: {1 ~ (totstru - 1), (totstru - 1) ~ 1}
    if(t_stru > totstru - 1) t_stru = 2 * totstru - 1 - t_stru ; // period: 2(totstru - 1)
    const double h = iontime / neleint;
    const int nrows_loc = Numroc(matsize, MB_ROW, myprow_group, nprow_group);
    const int ncols_loc = Numroc(matsize, NB_COL, mypcol_group, npcol_group);
    const size_t mn_loc = (size_t)nrows_loc * ncols_loc;
    for(int i = 0; i < 4; i++) {
        #pragma omp parallel for
        for(size_t ij = 0; ij < mn_loc; ij++) c_onsite[i][ij] = 0.0;
    }
    if(carrier == "electron" || carrier == "hole") {
        // electron: C_ki =  energy / (ihbar) - <k|d/dt|i>
        // hole:     C_ki = -energy / (ihbar) - <k|d/dt|i>^*
        // energy is onsite, <k|d/dt|i> is midsite
        const int nbnds = dimC + dimV;
        for(int i = 0; i < 4; i++) {
            ReadDiag((namddir + "/tmpEnergy/c0123").c_str(), 4 * (t_stru - 1) + i,
                     allispns, allikpts, ibndstart, nbnds, totnspns, totnkpts, totnbnds, 
                     (carrier == "electron" ? 1.0 : -1.0) / (iu_d * hbar), c_onsite[i]);
            if(hopmech == "nacsoc"); // need read extra soc matrix
        }
    }
    else if(carrier == "exciton") {
        // C_ki = (DiagEnDiff + a_w * W + a_v * v) / (ihbar) - a_eph * NAC, 
        // [a_w, a_v, a_eph] = dynchan[0, 1, 2]
        // NAC is midsite, else onesite
        const int start_idx = ( lrecomb ? 1 : 0 ); 
        for(int i = 0; i < 4; i++) {
            Blacs_ReadDiag2Full((namddir + "/tmpDiagonal/c0123").c_str(), 4 * (t_stru - 1) + i,
                                matsize, 1.0 / (iu_d * hbar), c_onsite[i], start_idx);
            if(is_bse_calc) {
                if(abs(dynchan[0]) > 1e-8) // direct term
                Blacs_ReadFullMat((namddir + "/tmpDirect/c0123").c_str(), 4 * (t_stru - 1) + i,
                                  matsize, dynchan[0] / (iu_d * hbar), /*locbeg, glbbeg, bcklen,*/ c_onsite[i], start_idx);
                if(abs(dynchan[1]) > 1e-8) // exchange term
                Blacs_ReadFullMat((namddir + "/tmpExchange/c0123").c_str(), 4 * (t_stru - 1) + i,
                                  matsize, dynchan[1] / (iu_d * hbar), /*locbeg, glbbeg, bcklen,*/ c_onsite[i], start_idx);
            }
            if((lrecomb == 2 || lrecomb == 3) && abs(dynchan[3]) > 1e-8) // radiative recombination
            ;
        }
    }
    return;
}

void ReadMidsiteC(const int t_ion, const int matsize,
                  const int nspns, const int nkpts, const int dimC, const int dimV,
                  vector<int> &allispns, vector<int> &allikpts, const int *ibndstart,
                  const int totnspns, const int totnkpts, const int totnbnds,
                  complex<double> **c_midsite) { // output
/*  t_ion: time of ion step: [2, totstru - 1], NOT include "1" and "totstru" */
/*  update: t_ion belongs to [1, +oo), when > totstru, need do bounce-back process*/
    int t_stru = t_ion % (2 * (totstru - 1)); // loop order: {1 ~ totstru - 1, "-"(totstru - 2 ~ 1), 0}
    double oddeven_scale = 1.0;               // period: 2(totstru - 1)
    if(t_stru > totstru - 1) { t_stru = 2 * (totstru - 1) - t_stru; oddeven_scale = -1.0; } 
    const double h = iontime / neleint;
    const int nrows_loc = Numroc(matsize, MB_ROW, myprow_group, nprow_group);
    const int ncols_loc = Numroc(matsize, NB_COL, mypcol_group, npcol_group);
    const size_t mn_loc = (size_t)nrows_loc * ncols_loc;
    for(int i = 0; i < 4; i++) {
        #pragma omp parallel for
        for(size_t ij = 0; ij < mn_loc; ij++) c_midsite[i][ij] = 0.0;
    }
    if(carrier == "electron" || carrier == "hole") {
        // electron: a_ki =  energy / (ihbar) - <k|d/dt|i>
        // hole:     a_ki = -energy / (ihbar) - <k|d/dt|i>^*
        // energy is onsite, <k|d/dt|i> is midsite
        const int nbnds = dimC + dimV;
        for(int i = 0; i < 4; i++) {
            ReadNAC((namddir + "/tmpNAC/c0123").c_str(), 4 * t_stru + i,
                    allispns, allikpts, ibndstart, nbnds, totnspns, totnkpts, totnbnds, 
                    -1.0 * oddeven_scale, c_midsite[i], carrier == "electron" ? false : true);
        }
    }
    else if(carrier == "exciton") {
        // a_ki = (DiagEnDiff + a_w * W + a_v * v) / (ihbar) - a_eph * NAC, 
        // [a_w, a_v, a_eph] = dynchan[0, 1, 2]
        // NAC is midsite, else onesite
        const int start_idx = ( lrecomb ? 1 : 0 ); 
        for(int i = 0; i < 4; i++) {
            if(abs(dynchan[2]) > 1e-8) // e-ph
            ReadNACbySK((namddir + "/tmpCBNAC/c0123").c_str(),
                        (namddir + "/tmpVBNAC/c0123").c_str(), 4 * t_stru + i,
                        start_idx, nspns, nkpts, dimC, dimV, - 1.0 * oddeven_scale * dynchan[2], c_midsite[i]);
            if((lrecomb == 1 || lrecomb == 3) && abs(dynchan[2]) > 1e-8) // nonradiative recombination
            ReadCtoVNAC((namddir + "/tmpC2VNAC/c0123").c_str(), 4 * t_stru + i,
                        nspns, nkpts, dimC, dimV, - 1.0 * oddeven_scale * dynchan[2], c_midsite[i]);
            ;
        }
    }

    return;
}

void Mat_CtoA(complex<double> **c_onsite, complex<double> **c_midsite, // input
              const int mirror_onsitec, const int mirror_midsitec,
              const int t_ref, const int t_ele, const double h, const int Nh,
              const int matsize,
              complex<double> *a0, complex<double> *a1,
              complex<double> *a2, complex<double> *a3) {
/*
    t_ref: time of reference, used for midsite, (-/+)neleint / 2
    t_ele: time of electron step, [0, neleint)
    mirror_c = {1, -1}, if mirror_c = -1, then x -> Nh * h - x and nagetive somewhere c
*/
    const int nrows_loc = Numroc(matsize, MB_ROW, myprow_group, nprow_group);
    const int ncols_loc = Numroc(matsize, NB_COL, mypcol_group, npcol_group);
    const size_t mn_loc = (size_t)nrows_loc * ncols_loc;
    const double dt_onsite   = h * ( 0.5 * (1 - mirror_onsitec) * Nh + mirror_onsitec * (0.5 + t_ele) );
    const double dt_onsite2  = dt_onsite  * dt_onsite;
    const double dt_onsite3  = dt_onsite2 * dt_onsite;
    const double dt_midsite  = h * ( 0.5 * (1 - mirror_midsitec) * Nh + mirror_midsitec * (0.5 + t_ele - t_ref) );
    const double dt_midsite2 = dt_midsite  * dt_midsite;
    const double dt_midsite3 = dt_midsite2 * dt_midsite;
    #pragma omp parallel for
    for(size_t ij = 0; ij < mn_loc; ij++) {
        a0[ij] = c_onsite[0][ij] + c_onsite[1][ij] * dt_onsite 
                                 + c_onsite[2][ij] * dt_onsite2 + c_onsite[3][ij] * dt_onsite3
               + c_midsite[0][ij] + c_midsite[1][ij] * dt_midsite 
                                  + c_midsite[2][ij] * dt_midsite2 + c_midsite[3][ij] * dt_midsite3;
        a1[ij] = 1.0 * mirror_onsitec * (c_onsite[1][ij] + 2.0 * c_onsite[2][ij] * dt_onsite 
                                                         + 3.0 * c_onsite[3][ij] * dt_onsite2)
               + 1.0 * mirror_midsitec * (c_midsite[1][ij] + 2.0 * c_midsite[2][ij] * dt_midsite
                                                           + 3.0 * c_midsite[3][ij] * dt_midsite2);
        a2[ij] = c_onsite[2][ij]  + 3.0 * c_onsite[3][ij]  * dt_onsite
               + c_midsite[2][ij] + 3.0 * c_midsite[3][ij] * dt_midsite;
        a3[ij] = 1.0 * mirror_onsitec * c_onsite[3][ij] + 1.0 * mirror_midsitec * c_midsite[3][ij];
    }
    
    return;
}

void A0Coeff(const int nstates, const double h, const complex<double> *a0,
             complex<double> *coeff) {
    const int ndim_loc_row = Numroc(nstates, MB_ROW, myprow_group, nprow_group);
    complex<double> *cdvectmp = new complex<double>[mypcol_group == 0 ? max(ndim_loc_row, 1) : 1]();
    if(mypcol_group == 0) Zcopy(ndim_loc_row, coeff, 1, cdvectmp, 1);
    MPI_Barrier(group_comm);
    
    Pzgemv("N", nstates, nstates, 1.0, a0,       ndim_loc_row,
                                       coeff,    ndim_loc_row,
                                  0.0, cdvectmp, ndim_loc_row);
    
    if(mypcol_group == 0) Zaxpy(ndim_loc_row, h, cdvectmp, coeff);
    MPI_Barrier(group_comm);
    double norm2 = Pdznrm2(nstates, coeff, ndim_loc_row, 1, 0, 0, nstates, 1);
    ZDscal(ndim_loc_row, 1.0 / norm2, coeff);
    MPI_Barrier(group_comm);
    delete[] cdvectmp;
    return;
}

void EomegaCoeff(const int nstates, const double h,
                 complex<double> *om_oih, complex<double> *coeff, const int ntrajs) {
/*
    calculate coeff = e^(i x h x om_oih) x coeff 
    read om_oih and coeff update in this routine
    matrix: om_oih[nstates x nstates]
    vector: coeff[nstates x 1]

    e^(i x h x om_oih) = u e^(i x h x lambda) u^+
    
    u is the eigenvector matrix of om_oih,
    lambda is a diagonal matrix with eigenvalues of om_oih
*/
    const int ndim_loc_row = Numroc(nstates, MB_ROW, myprow_group, nprow_group);
    const int ndim_loc_col = Numroc(nstates, NB_COL, mypcol_group, npcol_group);
    const size_t mn_loc = (size_t)ndim_loc_row * ndim_loc_col;
    complex<double> *eigenvecs = new complex<double>[max(mn_loc, (size_t)1)]();
    double *eigenvals = new double[nstates];
    Pzheev("V", nstates, om_oih, ndim_loc_row, eigenvals, eigenvecs, ndim_loc_row);
    
    if(ntrajs == 1) {
        complex<double> *cdvectmp = new complex<double>[mypcol_group == 0 ? max(ndim_loc_row, 1) : 1]();
        Pzgemv("C", nstates, nstates, 1.0, eigenvecs, ndim_loc_row,
                                           coeff,     ndim_loc_row,
                                      0.0, cdvectmp,  ndim_loc_row);  // coeff = u^+ x coeff 
        if(mypcol_group == 0) {
            #pragma omp parallel for
            for(int ii = 0; ii < ndim_loc_row; ii++) {                // coeff = e^(i x h x lambda) x coeff
                cdvectmp[ii] *= exp( iu_d * h * eigenvals[BlacsIdxloc2glb(ii, nstates, MB_ROW, myprow_group, nprow_group)] );
            }
        }
        MPI_Barrier(group_comm);
        Pzgemv("N", nstates, nstates, 1.0, eigenvecs, ndim_loc_row,
                                           cdvectmp,  ndim_loc_row,
                                      0.0, coeff,     ndim_loc_row);  // coeff = u x coeff
        
        /*complex<double> *fullmat = NULL;
        if(is_sub_root) fullmat = new complex<double>[(size_t)nstates * nstates];
        Blacs_MatrixZGather(nstates, nstates, eigenvecs, ndim_loc_row, fullmat, nstates);
        if(is_sub_root) {
            for(int ii = 0; ii < 10; ii++) {
                for(int jj = 0; jj < 10; jj++) { cout << fullmat[ii + jj * nstates]; }
                cout << endl;
            }
            delete[] fullmat;
        }
        MPI_Barrier(group_comm);
        while(1);*/
        delete[] cdvectmp;
    }
    else {
        const int ntrajs_loc_col = Numroc(ntrajs, NB_COL, mypcol_group, npcol_group);
        complex<double> *tmpmat = new complex<double>[max((size_t)ndim_loc_row * ntrajs_loc_col, (size_t)1)]();
        Pzgemm("C", "N", nstates, ntrajs, nstates,   // coeff = u^+ x coeff
               1.0, eigenvecs, ndim_loc_row, coeff, ndim_loc_row,
               0.0, tmpmat, ndim_loc_row);           
        #pragma omp parallel for
        for(int ii = 0; ii < ndim_loc_row; ii++) {   // coeff = e^(i x h x lambda) x coeff
            Zscal(ntrajs_loc_col,
                  exp( iu_d * h * eigenvals[BlacsIdxloc2glb(ii, nstates, MB_ROW, myprow_group, nprow_group)] ),
                  tmpmat + ii, ndim_loc_row);
        }
        MPI_Barrier(group_comm);
        Pzgemm("N", "N", nstates, ntrajs, nstates,   // coeff = u x coeff
               1.0, eigenvecs, ndim_loc_row, tmpmat, ndim_loc_row,
               0.0, coeff, ndim_loc_row);
        delete[] tmpmat;
    }
        
    delete[] eigenvecs; delete[] eigenvals;
    MPI_Barrier(group_comm);
    return;
}

void CoeffUpdate(const double h, const int t_ion, const int nstates, 
                 const int nspns, const int nkpts, const int dimC, const int dimV,
                 vector<int> &allispns, vector<int> &allikpts, const int *ibndstart,
                 const int totnspns, const int totnkpts, const int totnbnds,
                 complex<double> **c_onsite, complex<double> **c_midsite,
                 complex<double> *coeff, const int ntrajs,
                 complex<double> *a0, complex<double> *a1, complex<double> *a2, complex<double> *a3,
                 complex<double> *om_oih) {
/*
    calculate the coefficients evolution from t_ion to t_ion + 1, t_ion belongs to [1, +oo)
    the range [t_ion, t_ion + 1] is divided into "neleint" intervals with length of h = iontime / neleint 
    each integral in [t_ele x h, (t_ele + 1) x h] is calculate by

        Coeff[t_ele + 1] = e^Omega x Coeff[t_ele]

    where e^Omega is given in Mat_AtoOmega routine
    Coeff with dimension of "nstates" will update in this routine
    Also, c_midsite should be prepare before and update in this routine
*/
    const int mirror_onsitec  = ( (t_ion - 1) % (2 * (totstru - 1)) + 1 > totstru - 1 ? -1 : 1 );
    const int mirror_midsitec = ( t_ion % (2 * (totstru - 1)) > totstru - 1 ? -1 : 1 );
    ReadOnsiteC(t_ion, nstates, nspns, nkpts, dimC, dimV, 
                allispns, allikpts, ibndstart,
                totnspns, totnkpts, totnbnds, 
                c_onsite);
    for(int t_ele = 0; t_ele < neleint / 2; t_ele++) {
        Mat_CtoA(c_onsite, c_midsite, mirror_onsitec, mirror_midsitec,
                 - neleint / 2, t_ele, h, neleint, nstates, a0, a1, a2, a3);
        if(intalgo == "Euler") A0Coeff(nstates, h, a0, coeff); // usually for test
        else if(intalgo == "Magnus") {
            Mat_AtoOmega(nstates, h, a0, a1, a2, a3, om_oih);
            EomegaCoeff(nstates, h, om_oih, coeff, ntrajs);
        }
    }
    ReadMidsiteC(t_ion, nstates, nspns, nkpts, dimC, dimV,
                 allispns, allikpts, ibndstart,
                 totnspns, totnkpts, totnbnds, 
                 c_midsite);
    for(int t_ele = neleint / 2; t_ele < neleint; t_ele++) {
        Mat_CtoA(c_onsite, c_midsite, mirror_onsitec, mirror_midsitec, 
                 + neleint / 2, t_ele, h, neleint, nstates, a0, a1, a2, a3);
        if(intalgo == "Euler") A0Coeff(nstates, h, a0, coeff);
        else if(intalgo == "Magnus") {
            Mat_AtoOmega(nstates, h, a0, a1, a2, a3, om_oih);
            EomegaCoeff(nstates, h, om_oih, coeff, ntrajs);
        }
    }
    MPI_Barrier(group_comm);
    return;
}

void SetIniCoeff(complex<double> *coeff, double *population,
                 const int nstates, const int ismp, int &begtime,
                 vector<int> *&allbands, const int ntrajs, const char *inicon) {
    ifstream inf(inicon, ios::in);
    if(!inf.is_open()) { Cout << "\"" << inicon << "\" file doesn't find, please check."; exit(1); }
    string line;
    vector<string> vecstrtmp;
    for(int ii = 0; ii < ismp + 1; ii++) getline(inf, line);
    inf.close();

    vecstrtmp = StringSplitByBlank(line);
    begtime = stoi(vecstrtmp[0]);
    vector<int> inistates;
    vector<double> iniweight;
    int  spn,  kpt,  cbd,  vbd,  bnd;
    int ispn, ikpt, icbd, ivbd, ibnd;
    
    int rndifspns, nkpts, dimC, dimV;
    if(carrier == "exciton") ReadInfoTmp(rndifspns, nkpts, dimC, dimV);
    const int nnbnds = allbands[0].size();
    for(int inv = 0; inv < nvinidc; inv++) {
        if(carrier == "electron" || carrier == "hole") {
            spn = stoi(vecstrtmp[1 + 4 * inv]);
            kpt = stoi(vecstrtmp[1 + 4 * inv + 1]); 
            bnd = stoi(vecstrtmp[1 + 4 * inv + 2]); 
            if(nvinidc > 1) iniweight.push_back(stod(vecstrtmp[1 + 4 * inv + 3]));
            else iniweight.push_back(1.0);
            ispn = spn;
            ikpt = FindIndex(Kpoints, kpt);        // start from "0"
            ibnd = FindIndex(allbands[ispn], bnd); // start from "0"
            inistates.push_back(ispn * numkpts * nnbnds + ikpt * nnbnds + ibnd);
        }
        else if(carrier == "exciton") {
            spn = stoi(vecstrtmp[1 + 5 * inv]);
            kpt = stoi(vecstrtmp[1 + 5 * inv + 1]); 
            cbd = stoi(vecstrtmp[1 + 5 * inv + 2]); 
            vbd = stoi(vecstrtmp[1 + 5 * inv + 3]); 
            if(nvinidc > 1) iniweight.push_back(stod(vecstrtmp[1 + 4 * inv + 3]));
            else iniweight.push_back(1.0);
            ispn = spn;
            ikpt = FindIndex(Kpoints, kpt);        // start from "0"
            icbd = FindIndex(allbands[ispn], cbd); // start from "0"
            ivbd = FindIndex(allbands[ispn], vbd) - dimC; // start from "0"
            const int iground = ( lrecomb ? 1 : 0 ); 
            inistates.push_back(ispn * nkpts * dimC * dimV + ikpt * dimC * dimV + icbd * dimV + ivbd + iground);
        }
    }

    const double sumwt = Dasum(nvinidc, (double*)&(iniweight[0]));
    Dscal(nvinidc, 1.0 / sumwt, (double*)&(iniweight[0]));

    const int nstates_loc = Numroc(nstates, MB_ROW, myprow_group, nprow_group);
    const int ntrajs_loc_col = Numroc(ntrajs, NB_COL, mypcol_group, npcol_group);
    for(int itraj = 0; itraj < ntrajs_loc_col; itraj++) {
        #pragma omp parallel for
        for(int ist = 0; ist < nstates_loc; ist++) {
            coeff[ist + itraj * nstates_loc] = 0.0;
            population[ist + itraj * nstates_loc] = 0.0;
        }
        int ist_loc, iprow;
        for(int inv = 0; inv < nvinidc; inv++) {
            BlacsIdxglb2loc(inistates[inv], iprow, ist_loc, 0, nstates, MB_ROW, nprow_group);
            if(myprow_group == iprow) {
                coeff[ist_loc + itraj * nstates_loc] = sqrt(iniweight[inv]);
                population[ist_loc + itraj * nstates_loc] = iniweight[inv];
            }
        }
    }
    MPI_Barrier(group_comm);
    return;
}
