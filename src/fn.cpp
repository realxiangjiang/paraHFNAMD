#include "fn.h"

bool FFTch1(int n) {
/*
    Check if n can be factorized into products of 2, 3 and 5.
    From VASP fft3dcray.F, FFTCH1
*/
    if(n % 2 == 1) return false;
    for(int i = 0; i < round(log(n) / log(5)) + 1; i++)
    for(int j = 0; j < round(log(n) / log(3)) + 1; j++)
    for(int k = 1; k < round(log(n) / log(2)) + 1; k++) { // exclude odd number
        if(round(pow(5, i)) * round(pow(3, j)) * round(pow(2, k)) == n) return true;
    }
    return false;
}

void FFTch3(int ngf[]) {
    for(int s = 0; s < 3; s++)
    while(!FFTch1(ngf[s])) ngf[s]++;
    return;
}

void DeterminCommNumClr(const int nbinding, const int rk, const int sz, int &num, int &clr) {
    num = sz / nbinding; 
    // sz % num segments have extra one process
    if( rk < (sz / num + 1) * (sz % num) ) clr = rk / (sz / num + 1);
    else clr = sz % num + (rk - (sz / num + 1) * (sz % num)) / (sz / num);
    return;
}

void RowColSquareSet(const int area, int &nrow, int &ncol) {
    for(nrow = 1; nrow <= area; nrow++) {
        if(area % nrow == 0) {
            ncol = area / nrow;
            if(nrow >= ncol) break;
        }
    }
    return;
}

void ColRowSquareSet(const int area, int &nrow, int &ncol) {
    for(nrow = area; nrow >= 1; nrow--) {
        if(area % nrow == 0) {
            ncol = area / nrow;
            if(nrow <= ncol) break;
        }
    }
    return;
}

int BlacsIdxloc2glb(const int locidx, const int n, const int nb, const int iproc, const int nproc, const int psrc) {
    return nproc * nb * (locidx / nb) + (iproc - psrc + nproc) % nproc * nb + locidx % nb;
}

void BlacsIdxglb2loc(const int glbidx, int &iproc, int &locidx,
                     const int psrc, const int n, const int nb, const int nproc) {
    iproc = (glbidx / nb + psrc) % nproc;
    locidx = glbidx / (nproc * nb) * nb + glbidx % nb;
    
    return;
}

void BlacsBegidxBlocklen(const int iproc, const int nproc, const int n, const int nb,
                         vector<int> &locbeg, vector<int> &glbbeg, vector<int> &bcklen, // output
                         const int psrc) {
    // i/nproc: the ith/total process; n: dimension size; nb: block size
    const int diproc = (iproc - psrc + nproc) % nproc;
    const int pl = nproc * nb; // period length
    const int remain =  (n - diproc * nb) % pl;  // { 0 <= diproc < nproc and n >= 1 } ---> { n - diproc * nb > - pl + 1 }
    const int nperiod = (n - diproc * nb) / pl + (remain > 0 ? 1 : 0); // include the last remained part
    locbeg.clear(); glbbeg.clear(); bcklen.clear();
    
    if(nperiod == 0) return;
    
    for(int ii = 0; ii < nperiod - 1; ii++) {
        glbbeg.push_back(diproc * nb + ii * pl);
        bcklen.push_back(nb);
        if(locbeg.empty()) locbeg.push_back(0);
        else locbeg.push_back(locbeg.back() + nb);
    }
    if(remain == 0) bcklen.push_back(nb);
    else if(remain > 0) bcklen.push_back(remain / nb ? nb : remain % nb);
    glbbeg.push_back(diproc * nb + (nperiod - 1) * pl);
    if(locbeg.empty()) locbeg.push_back(0);
    else locbeg.push_back(locbeg.back() + nb);

    return;
}

string Int2Str(const int input, const int strsize, const char fillc) { // integer to strsize length string fill with fillc
    ostringstream sstr;
    sstr << setw(strsize) << setfill(fillc) << input;
    return sstr.str();
}

string WholeFile2String(ifstream &inf) {
    ostringstream sstr;
    sstr << inf.rdbuf();
    return sstr.str();
}

long StringSplit(string &ins, string &ots, string words, bool is_contain) { // in string: ins; out string: ots
    size_t idx = ins.find(words);
    if(idx != string::npos) {
        if(is_contain) {
            ots = ins.substr(0, idx) + words;
            return idx + words.length();
        }
        else {
            ots = ins.substr(0, idx);
            return idx;
        }
    }
    else {
        ots = "";
        return -1;
    }
}

vector<string> StringSplitByBlank(string &ins) {
    vector<string> vecstr;
    istringstream iss(ins);
    string stmp;
    while(iss >> stmp) vecstr.push_back(stmp);
    return vecstr;
}

int StringMatchCount(string &ins, string target) {
    int occur = 0;
    size_t pos = 0;
    while((pos = ins.find(target, pos)) != string::npos) { occur++; pos += target.length(); }
    return occur;
}


int IdxNat1toSym1(const int ii, const int NN) { return (ii < NN / 2 + 1 ? ii : (ii - NN)); }
int IdxSym1toNat1(const int ii, const int NN) { return (ii >= 0 ? ii : (ii + NN)); }
int IdxNat3toNat1(const int ii, const int jj, const int kk, const int NN0, const int NN1, const int NN2) { return ii * NN1 * NN2 + jj * NN2 + kk; }
int IdxSym3toNat1(const int ii, const int jj, const int kk, const int NN0, const int NN1, const int NN2) { return (ii >= 0 ? ii : ii + NN0) * NN1 * NN2 + (jj >= 0 ? jj : jj + NN1) * NN2 + (kk >= 0 ? kk : kk + NN2); }
void IdxNat1toNat3(const int in, int &out0, int &out1, int &out2, const int NN0, const int NN1, const int NN2) {
    out0 = in / (NN1 * NN2);
    out1 = in % (NN1 * NN2) / NN2;
    out2 = in % NN2;
    return;
}
void IdxNat1toSym3(const int in, int &out0, int &out1, int &out2, const int NN0, const int NN1, const int NN2) {
    IdxNat1toNat3(in, out0, out1, out2, NN0, NN1, NN2);
    out0 = IdxNat1toSym1(out0, NN0);
    out1 = IdxNat1toSym1(out1, NN1);
    out2 = IdxNat1toSym1(out2, NN2);
    return;
}

int MatchKptDiff(int ikpt1, int ikpt2, int NK[]) {
    int diffkp[3];
    int kp1[3], kp2[3];
    IdxNat1toSym3(ikpt1, kp1[2], kp1[1], kp1[0], NK[2], NK[1], NK[0]);
    IdxNat1toSym3(ikpt2, kp2[2], kp2[1], kp2[0], NK[2], NK[1], NK[0]);
    return IdxSym3toNat1(kp1[2] - kp2[2],  kp1[1] - kp2[1],  kp1[0] - kp2[0],
                         2 * NK[2] - 1, 2 * NK[1] - 1, 2 * NK[0] - 1);
}

void BlockCyclicToMpiFFTW3d(const complex<double> *in, const int totin, // compacted with global size totin
                            complex<double> *out,
                            const int nin[], const int nout[],
                            const int loc_nout0, const int loc_out0_start, 
                            const int *idxin) { // global index of in
/*
      In Scalapack/PBLAS, data are block-cyclic stored;
      while in mpi-fftw, data are divided along the first axis.
      This routine transform one block-cyclic column 3D data to fftw style.
      algorithm:
      ijk of in ==> (i, j, k) of (i x nin[1] x nin[2] + j x nin[2] + k) by idxin
                ==> update(i, j, k) ==> (i - loc_out0_start, j, k)
*/
    #pragma omp parallel for
    for(int ijk = 0; ijk < loc_nout0 * nout[1] * nout[2]; ijk++) out[ijk] = 0.0;
    MPI_Barrier(col_comm);

    int send_rank, recv_rank;
    int *i_am_recv_rank = new int[col_size];
    int ii, jj, kk;
    int ijk_loc;
    for(int ijk = 0; ijk < totin; ijk++) {
        BlacsIdxglb2loc(ijk, send_rank, ijk_loc, 0, totin, MB_ROW, nprow_group);
        IdxNat1toSym3(idxin[ijk], ii, jj, kk, nin[0], nin[1], nin[2]); //  - (nin + 1) / 2 < ii/jj/kk < nin / 2 + 1
        ii = IdxSym1toNat1(ii, nout[0]); 
        jj = IdxSym1toNat1(jj, nout[1]); 
        kk = IdxSym1toNat1(kk, nout[2]); // 0 <= ii/jj/kk < nout
        for(int irk = 0; irk < col_size; irk++) i_am_recv_rank[irk] = 0; // set all to zero
        if(loc_out0_start <= ii && ii < loc_out0_start + loc_nout0) {    // only one process contents
            i_am_recv_rank[col_rank] = 1;
        }
        MPI_Allreduce(MPI_IN_PLACE, i_am_recv_rank, col_size, MPI_INT, MPI_SUM, col_comm);
        for(int irk = 0; irk < col_size; irk++) if(i_am_recv_rank[irk]) { recv_rank = irk; break; }
        //if(col_rank == send_rank) cout << "ijk = " << ijk << '/' << totin << ": send/recv rank = " << send_rank << '/' << recv_rank << endl;
        
        if(recv_rank == send_rank) { // send and recv are in same process
            if(col_rank == send_rank)
            out[IdxNat3toNat1(ii - loc_out0_start, jj, kk, loc_nout0, nout[1], nout[2])] = in[ijk_loc];
        }
        else { // need communication
            if(col_rank == send_rank) 
            MPI_Send(in + ijk_loc, 1, MPI_CXX_DOUBLE_COMPLEX, recv_rank, ijk, col_comm);
            else if(col_rank == recv_rank) {
                MPI_Status mpi_status;
                MPI_Recv(out + IdxNat3toNat1(ii - loc_out0_start, jj, kk, loc_nout0, nout[1], nout[2]),
                         1, MPI_CXX_DOUBLE_COMPLEX, send_rank, ijk, col_comm, &mpi_status);
            }
        }
        MPI_Barrier(col_comm);
    }
    delete[] i_am_recv_rank;
    
    return;
}

void MpiFFTW3dToBlockCyclic(const complex<double> *in,
                            complex<double> *out, const int totout, // compacted with global size totout, 
                            const int nin[], const int nout[],
                            const int loc_nin0, const int loc_in0_start,
                            const int *idxout) {
/* reverse of BlockCyclicToMpiFFTW3d */
    int send_rank, recv_rank;
    int *i_am_send_rank = new int[col_size];
    int ii, jj, kk;
    int ijk_loc;
    for(int ijk = 0; ijk < totout; ijk++) {
        BlacsIdxglb2loc(ijk, recv_rank, ijk_loc, 0, totout, MB_ROW, nprow_group);
        IdxNat1toSym3(idxout[ijk], ii, jj, kk, nout[0], nout[1], nout[2]); // - (nout + 1) / 2 < ii/jj/kk < nout / 2 + 1
        ii = IdxSym1toNat1(ii, nin[0]); 
        jj = IdxSym1toNat1(jj, nin[1]); 
        kk = IdxSym1toNat1(kk, nin[2]); // 0 <= ii/jj/kk < nin
        for(int irk = 0; irk < col_size; irk++) i_am_send_rank[irk] = 0; // set all to zero
        if(loc_in0_start <= ii && ii < loc_in0_start + loc_nin0) {       // exist and only exist one process contents
            i_am_send_rank[col_rank] = 1;
        }
        MPI_Allreduce(MPI_IN_PLACE, i_am_send_rank, col_size, MPI_INT, MPI_SUM, col_comm);
        for(int irk = 0; irk < col_size; irk++) if(i_am_send_rank[irk]) { send_rank = irk; break; }
    
        if(recv_rank == send_rank) { // send and recv are in same process
            if(col_rank == send_rank)
            out[ijk_loc] = in[IdxNat3toNat1(ii - loc_in0_start, jj, kk, loc_nin0, nin[1], nin[2])];
        }
        else { // need communication
            if(col_rank == send_rank)
            MPI_Send(in + IdxNat3toNat1(ii - loc_in0_start, jj, kk, loc_nin0, nin[1], nin[2]),
                     1, MPI_CXX_DOUBLE_COMPLEX, recv_rank, ijk, col_comm);
            else if(col_rank == recv_rank) {
                MPI_Status mpi_status;
                MPI_Recv(out + ijk_loc, 1, MPI_CXX_DOUBLE_COMPLEX, send_rank, ijk, col_comm, &mpi_status);
            }
        }
        MPI_Barrier(col_comm);
    }
    delete[] i_am_send_rank;
 
    return;
}

void Compacted2Cubiod(const int totin, const int *idxin,
                      const int nin[], const complex<double> *in,
                      const int nout[],      complex<double> *out) {
    int ii, jj, kk;
    fill_n(out, (size_t)nout[0] * nout[1] * nout[2], complex<double>(0.0, 0.0));
    #pragma omp parallel for private(ii, jj, kk)
    for(int ijk = 0; ijk < totin; ijk++) {
        IdxNat1toSym3(idxin[ijk], ii, jj, kk, nin[0], nin[1], nin[2]); //  - (nin + 1) / 2 < ii/jj/kk < nin / 2 + 1
        out[IdxSym3toNat1(ii, jj, kk, nout[0], nout[1], nout[2])] = in[ijk];
    }

    return;
}

void Cubiod2Compacted(const int nin[], const complex<double> *in,
                      const int totout, const int *idxout, 
                      const int nout[], complex<double> *out) {
    int ii, jj, kk;
    #pragma omp parallel for private(ii, jj, kk)
    for(int ijk = 0; ijk < totout; ijk++) {
        IdxNat1toSym3(idxout[ijk], ii, jj, kk, nout[0], nout[1], nout[2]); //  - (nout + 1) / 2 < ii/jj/kk < nout / 2 + 1
        out[ijk] = in[IdxSym3toNat1(ii, jj, kk, nin[0], nin[1], nin[2])];
    }

    return;
}
