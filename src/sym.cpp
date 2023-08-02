#include "sym.h"

void AxisTheta2Rmat(const double *axis, const double theta, const double det,
                    double *a[], double *b[],
                    double *rmat, double *aRb, double *aRTb) {
/*
    axis with dimension of 3
    aRb: [3 x 3], column-major

    the rotation operator R rotate (kx, ky, kz) to (k'x, k'y, k'z)
    that is

               (kx)             (k'x)
    R(b0 b1 b2)(ky) = (b0 b1 b2)(k'y)
               (kz)             (k'z)

        (k'x)   (a0)              (kx)
    ==> (k'y) = (a1) R (b0 b1 b2) (ky)
        (k'z)   (a2)              (kz)
    

*/
    double ux = axis[0], uy = axis[1], uz = axis[2];
    double costa = cos(theta), sinta = sin(theta);
    double *rbmat = new double[9]();
    double *amat = new double[9]();
    double *bmat = new double[9]();
    
    rmat[0] = costa + ux * ux * ( 1 - costa );      // [0, 0]
    rmat[1] = uy * ux * ( 1 - costa ) + uz * sinta; // [0, 1]
    rmat[2] = uz * ux * ( 1 - costa ) - uy * sinta; // [0, 2]

    rmat[3] = ux * uy * ( 1 - costa ) - uz * sinta; // [1, 0]
    rmat[4] = costa + uy * uy * ( 1 - costa );      // [1, 1]
    rmat[5] = uz * uy * ( 1 - costa ) + ux * sinta; // [1, 2]

    rmat[6] = ux * uz * ( 1 - costa ) + uy * sinta; // [2, 0]
    rmat[7] = uy * uz * ( 1 - costa ) - ux * sinta; // [2, 1]
    rmat[8] = costa + uz * uz * (1 - costa);        // [2, 2]

    Dscal(9, det, rmat);

    for(int i = 0; i < 3; i++) {
        Dcopy(3, a[i], 1, amat + i,     3); // (a0 a1 a2)^T
        Dcopy(3, b[i], 1, bmat + i * 3, 1); // (b0 b1 b2)
    }

    Dgemm("CblasColMajor", "CblasNoTrans", "CblasNoTrans",  // Rb = R(b0 b1 b2)
          3, 3, 3, 1.0, rmat, 3, bmat, 3, 
                   0.0, rbmat, 3);
    Dgemm("CblasColMajor", "CblasNoTrans", "CblasNoTrans",  // aRb = (a0 a1 a2)^T Rb
          3, 3, 3, 1.0, amat, 3, rbmat, 3, 
                   0.0, aRb, 3);
    
    Dgemm("CblasColMajor", "CblasTrans", "CblasNoTrans",    // RTb = R^T(b0 b1 b2)
          3, 3, 3, 1.0, rmat, 3, bmat, 3, 
                   0.0, rbmat, 3);
    Dgemm("CblasColMajor", "CblasNoTrans", "CblasNoTrans",  // aRTb = (a0 a1 a2)^T RTb
          3, 3, 3, 1.0, amat, 3, rbmat, 3, 
                   0.0, aRTb, 3);

    delete[] rbmat; 
    delete[] amat; delete[] bmat;

    return;
}

operatorclass::operatorclass() {}
void operatorclass::Initial(const int in_no, const int in_det, const double in_theta,
                            const double in_ux, const double in_uy, const double in_uz,
                            const double in_taux, const double in_tauy, const double in_tauz,
                            double *a[], double *b[]) {
    isInitial = true;
    no = in_no; det = in_det; theta = in_theta;
    axis = new double[3];
    axis[0] = in_ux; axis[1] = in_uy; axis[2] = in_uz;
    tau = new double[3];
    tau[0] = in_taux; tau[1] = in_tauy; tau[2] = in_tauz;
    rmat = new double[9];
    aRb = new double[9];
    aRTb = new double[9];

    AxisTheta2Rmat(axis, theta, (double)det, a, b, rmat, aRb, aRTb);

    return;

}

operatorclass::~operatorclass() {
    if(isInitial) {
        delete[] axis; delete[] tau;
        delete[] rmat; delete[] aRb; delete[] aRTb;
    }
}

int FindOpNum(int *gijk, const int time_reve_opt, // output gijk[] = {-1, 0, 1}
              const double *q, const double *q1, operatorclass *optcs, const int numopt) {
/*
    this routine find the operator No. in optcs which satisfy 
                
            time_reve_opt x q1 = R(q) + G
    
    G is any interger 3-dimension vector
    q and q1 are fractional vectors with |q[i]| <= 0.5 and |q1[i]| < 1.0
    time_reve_opt = 1 or -1
*/
    double *rq = new double[3]();
    int i, j, k;
    for(int iopt = 0; iopt < numopt; iopt++) {
        Dgemv("CblasColMajor", "CblasNoTrans", 3, 3,
              1.0, optcs[iopt].aRb, 3, q, 1,
              0.0,                    rq, 1);
        for(int ijk = 0; ijk < 27; ijk++) { // loop for rq + G, G[ijk] = {-1, 0, 1}
            i = ijk / 9;
            j = ijk % 9 / 3;
            k = ijk % 3;
            i -= 1; j -= 1; k -= 1;
            /*cout << '(' << q1[0] << ' ' << q1[1] << ' ' << q1[2] << ")("
                 << q[0] << ' ' << q[1] << ' ' << q[2] << ") "<< iopt << ": "
                 << '(' << rq[0] + i - q1[0] << ',' << rq[1] + j - q1[1] << ',' << rq[2] + k - q1[2] << "): rq+G="
                 << '(' << rq[0] + i << ',' << rq[1] + j << ',' << rq[2] + k << ')' << endl;*/
            if( abs(rq[0] + i - q1[0] * time_reve_opt) < 0.000002 &&
                abs(rq[1] + j - q1[1] * time_reve_opt) < 0.000002 &&
                abs(rq[2] + k - q1[2] * time_reve_opt) < 0.000002 ) {
                gijk[0] = i; gijk[1] = j; gijk[2] = k;
                delete[] rq;
                return iopt;
            }
        } // ijk
    } // iopt

    delete[] rq;
    cerr << "FindOpNum: NOT find consistent q for q1 = (" << q1[0] << ',' << q1[1] << ',' << q1[2]
         << "), please check" << endl;
    exit(1);
    
    return -1;
}
