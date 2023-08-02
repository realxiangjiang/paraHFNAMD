#include <iostream>
#include <fstream>
#include <cassert>
#include <cmath>
#include <complex>
#include <iomanip>   // setiosflags
#include <algorithm> // min

using namespace std;

const int lmax = 3;
const complex<long double> iu_ld(0.0, 1.0);

long double Factorial(int n) {
    if(n == 0 || n == 1) return 1.0;
    long double res = 1.0;
    for(int i = 2; i <= n; i++) res *= (long double)i;
    return res;
}

long double deltaj1j2j3(const int j1, const int j2, const int j3) {
    return sqrt( Factorial(j1 + j2 - j3) * Factorial(j1 - j2 + j3) * Factorial(- j1 + j2 + j3) 
               / Factorial(j1 + j2 + j3 + 1) );
}

long double w3j(const int j1, const int j2, const int j3,
                const int m1, const int m2, const int m3) {
    // reference: https://dlmf.nist.gov/34.2
    long double res = 0.0;
    if(!(abs(m1) <= j1 && abs(m2) <= j2 && abs(m3) <= j3 &&
         abs(j1 - j2) <= j3 && j3 <= j1 + j2 && m1 + m2 + m3 == 0)) return res;
    int myints[3] = { j1 + j2 - j3, j1 - m1, j2 + m2 };
    for(int s = 0; s <= min(j1 + j2 - j3, min(j1 - m1, j2 + m2)); s++) {
        if(j3 - j2 + m1 + s >= 0 && j3 - j1 - m2 + s >= 0)
        res += pow(-1, s) / Factorial(s) / Factorial(j1 + j2 - j3 - s) 
                          / Factorial(j1 - m1 - s) / Factorial(j2 + m2 - s)
                          / Factorial(j3 - j2 + m1 + s) / Factorial(j3 - j1 - m2 + s);
    }
   
    return res * pow(-1, j1 - j2 - m3) * deltaj1j2j3(j1, j2, j3)
               * sqrt( Factorial(j1 + m1) * Factorial(j1 - m1) 
                     * Factorial(j2 + m2) * Factorial(j2 - m2)
                     * Factorial(j3 + m3) * Factorial(j3 - m3) );
}

long double Yl1m1Yl2m2YLM(const int l1, const int m1, 
                          const int l2, const int m2,
                          const int L,  const int M) {
    return sqrt( (2 * l1 + 1) * (2 * l2 + 1) * (2 * L + 1) / 4.0 / M_PI )
         * w3j(l1, l2, L, m1, m2, M) * w3j(l1, l2, L, 0, 0, 0);
}

complex<long double> MsubM2scale(const int M, const int subM) {
    if(M < 0 && subM == 0) return iu_ld / sqrt((long double)2.0);
    if(M < 0 && subM == 1) return iu_ld / sqrt((long double)2.0) * (- (long double)pow(-1, M));
    if(M == 0) return 1.0;
    if(M > 0 && subM == 0) return (long double)(1.0 / sqrt(2.0));
    if(M > 0 && subM == 1) return (long double)(1.0 / sqrt(2.0) * pow(-1, M));
}

int MsubM2newM(const int M, const int subM) {
    if(M < 0 && subM == 0) return M;
    if(M < 0 && subM == 1) return -M;
    if(M == 0) return 0;
    if(M > 0 && subM == 0) return -M;
    if(M > 0 && subM == 1) return M;
}

int main() {
    ofstream otf("w3j.h", ios::out); assert(otf.is_open());
    const int totlm12 = (1 + lmax) * (1 + lmax);
    const int totLM  = (1 + 2 * lmax) * (1 + 2 * lmax);
    const int totlm1lm2LM = totLM * totlm12 * totlm12;
    //( L * L + (M + L) ) * totlm12 * totlm12 + ( l1 * l1 + (m1 + l1) ) * totlm12 + ( l2 * l2 + (m2 + l2) ) 
    otf << "#ifndef w3j_h" << endl << "#define w3j_h" << endl << endl
        << "const int totlm12 = " << totlm12 << "; // lmax = " << lmax << endl
        << "const double YLMY1Y2[" << totlm1lm2LM
        << "] = { // integral( YLM x Yl1m1 x Yl2m2 ), all Y's are real spherical harmonics" << endl;
    int nn = 0;
    for(int L = 0; L <= 2 * lmax; L++)   // the order
    for(int M = -L; M <= L; M++)         // of the
    for(int l1 = 0; l1 <= lmax; l1++)    // 6
    for(int m1 = -l1; m1 <= l1; m1++)    // lines
    for(int l2 = 0; l2 <= lmax; l2++)    // CAN'T
    for(int m2 = -l2; m2 <= l2; m2++) {  // change
        nn++;
        if( abs(l1 - l2) <= L && L <= l1 + l2 && (
            m1 + m2 + M == 0 ||  m1 + m2 - M == 0 || 
            m1 - m2 + M == 0 ||  m1 - m2 - M == 0 ||
           -m1 + m2 + M == 0 || -m1 + m2 - M == 0 ||
           -m1 - m2 + M == 0 || -m1 - m2 - M == 0 )) {
            complex<long double> res = 0.0;
            for(int subm1 = 0; subm1 < (m1 ? 2 : 1); subm1++)
            for(int subm2 = 0; subm2 < (m2 ? 2 : 1); subm2++)
            for(int subM  = 0; subM  < (M  ? 2 : 1);  subM++)
            if(MsubM2newM(m1, subm1) + MsubM2newM(m2, subm2) + MsubM2newM(M, subM) == 0) {
                res += MsubM2scale(m1, subm1) * MsubM2scale(m2, subm2) * MsubM2scale(M, subM)
                     * Yl1m1Yl2m2YLM(l1, MsubM2newM(m1, subm1),
                                     l2, MsubM2newM(m2, subm2),
                                     L,  MsubM2newM(M,  subM ));
            }
            if( abs(res) > 1e-18 ) {
                if(real(res) > 0.0) otf << ' ';
                otf << setiosflags(ios::fixed) << setprecision(16) << real(res);
            }
            else otf << " 0.0";
            if(nn != totlm1lm2LM) otf << ',' << endl;

            /*if(abs(res) > 1e-18) { nn++;
            cout << '(' << l1 << ',' << l2 << ',' << L <<
                    ',' << m1 << ',' << m2 << ',' << M << "): " << w3j(l1, l2, L, m1, m2, M) << ' '
                                                                << Yl1m1Yl2m2YLM(l1, m1, l2, m2, L, M) << ' '
                                                                << res << endl; } */
        }
        else {
            otf << " 0.0";
            if(nn != totlm1lm2LM) otf << ',' << endl;
        }
        /*if( abs(l1 - l2) <= L && L <= l1 + l2 && m1 + m2 + M == 0 )
        cout << '(' << l1 << ',' << l2 << ',' << L <<
                ',' << m1 << ',' << m2 << ',' << M << "): " << w3j(l1, l2, L, m1, m2, M) << ' '
                                                            << Yl1m1Yl2m2YLM(l1, m1, l2, m2, L, M) << endl;*/
    }
    otf << "};" << endl << endl;

    otf << setiosflags(ios::fixed) << setprecision(20);
    // <phi_i|grad_x|phi_j>
    otf << "const double piDpj_x[256] = {" << endl;
    otf << "0.0, 0.0, 0.0, " << 1.0/(2.0*sqrt(3.0)) << ", 0.0, 0.0, 0.0, 0.0, 0.0, // (0, 0)" << endl; // (0, 0), l'=0~2
    otf << "0.0, 0.0, 0.0, 0.0, " << -(1.0/2.0)*sqrt(7.0/6.0) << ", 0.0, 0.0," << endl << endl;        // (0, 0), l'=3
    
    otf << "0.0, 0.0, " << 3.0*M_PI/8.0 << ", 0.0, " <<  1.0/(2.0*sqrt(5.0)) << ", 0.0, 0.0, 0.0, 0.0, // (1,-1)" << endl;       // (1,-1), l'=0~2
    otf << "0.0, 0.0, 0.0, " << 3.0*sqrt(21.0)*M_PI/64.0 << ", 0.0, " << -3.0*sqrt(35.0)*M_PI/128.0 << ", 0.0," << endl << endl; // (1,-1), l'=3
    
    otf << "0.0, " << -3.0*M_PI/8.0 << ", 0.0, 0.0, 0.0, 0.0, 0.0, " <<  1.0/(2.0*sqrt(5.0)) << ", 0.0, // (1, 0)" << endl; // (1, 0), l'=0~2
    otf << "0.0, 0.0, " << (-3.0/64.0)*sqrt(7.0/2.0)*M_PI << ", 0.0, 0.0, 0.0, 0.0," << endl << endl;                       // (1, 0), l'=3
    
    otf << "0.0, 0.0, 0.0, 0.0, 0.0, 0.0, " <<  -sqrt(3.0/5.0) << ", 0.0, " << 1.0/(2.0*sqrt(5.0)) << ", // (1, 1)" << endl; // (1, 1), l'=0~2
    otf << "0.0, " << 3.0*sqrt(35.0)*M_PI/128.0 << ", 0.0, 0.0, 0.0, 0.0, 0.0," << endl << endl;                             // (1, 1), l'=3
    
    otf << "0.0, " << 1.0/(4.0*sqrt(5.0)) << ", 0.0, 0.0, 0.0, 0.0, 0.0, " <<  15.0*M_PI/64.0 << ", 0.0, // (2,-2)" << endl; // (2,-2), l'=0~2
    otf << (1.0/2.0)*sqrt(3.0/14.0) << ", 0.0, " << -2.0*sqrt(2.0/35.0) << ", 0.0, 0.0, 0.0, 0.0," << endl << endl;          // (2,-2), l'=3
    
    otf << "0.0, 0.0, 0.0, 0.0, 0.0, 0.0, " <<  15.0*sqrt(3.0)*M_PI/32.0 << ", 0.0, " << 15.0*M_PI/64.0 << ", // (2,-1)" << endl; // (2,-1), l'=0~2
    otf << "0.0, " << 1.0/(2.0*sqrt(7.0)) << ", 0.0, 0.0, 0.0, 0.0, 0.0," << endl << endl;                                        // (2,-1), l'=3
    
    otf << "0.0, 0.0, 0.0, " << 1.0/sqrt(15.0) << ", 0.0, " << -15.0*sqrt(3.0)*M_PI/32.0 << ", 0.0, 0.0, 0.0, // (2, 0)" << endl; // (2, 0), l'=0~2
    otf << "0.0, 0.0, 0.0, 0.0, " << 13.0/(2.0*sqrt(210.0)) << ", 0.0, 0.0," << endl << endl;                                     // (2, 0), l'=3
    
    otf << "0.0, 0.0, " << -1.0/sqrt(5.0) << ", 0.0, " << -15.0*M_PI/64.0 << ", 0.0, 0.0, 0.0, 0.0, // (2, 1)" << endl;  // (2, 1), l'=0~2
    otf << "0.0, 0.0, 0.0, " << -4.0*sqrt(3.0/35.0) << ", 0.0, " << 1.0/(2.0*sqrt(7.0)) << ", 0.0," << endl << endl;     // (2, 1), l'=3
    
    otf << "0.0, 0.0, 0.0, " << 1.0/(4.0*sqrt(5.0)) << ", 0.0, " << -15.0*M_PI/64.0 << ", 0.0, 0.0, 0.0, // (2, 2)" << endl; // (2, 2), l'=0~2
    otf << "0.0, 0.0, 0.0, 0.0, " << -2.0*sqrt(2.0/35.0) << ", 0.0, " << (1.0/2.0)*sqrt(3.0/14.0) << ',' << endl << endl;    // (2, 2), l'=3
    
    otf << "0.0, 0.0, 0.0, 0.0, " << 1.0/sqrt(42.0) << ", 0.0, 0.0, 0.0, 0.0, // (3,-3)" << endl;        // (3,-3), l'=0~2
    otf << "0.0, 0.0, 0.0, 0.0, 0.0, " << (105.0/512.0)*sqrt(3.0/2.0)*M_PI << ", 0.0, " << endl << endl; // (3,-3), l'=3
    
    otf << "0.0, 0.0, 0.0, " <<  -3.0*sqrt(35.0)*M_PI/128.0 << ", 0.0, " << -1.0/(4.0*sqrt(7.0)) << ", 0.0, 0.0, 0.0, // (3,-2)" << endl;      // (3,-2), l'=0~2
    otf << "0.0, 0.0, 0.0, 0.0, " << (147.0/512.0)*sqrt(5.0/2.0)*M_PI << ", 0.0, " << (105.0/512.0)*sqrt(3.0/2.0)*M_PI << ',' << endl << endl; // (3,-2), l'=3
    
    otf << "0.0, 0.0, " <<  (3.0/64.0)*sqrt(7.0/2.0)*M_PI << ", 0.0, " << sqrt(2.0/35.0) << ", 0.0, 0.0, 0.0, 0.0, // (3,-1)" << endl;         // (3,-1), l'=0~2
    otf << "0.0, 0.0, 0.0, " << (273.0/256.0)*sqrt(3.0/2.0)*M_PI << ", 0.0, " << (147.0/512.0)*sqrt(5.0/2.0)*M_PI << ", 0.0," << endl << endl; // (3,-1), l'=3
    
    otf << "0.0, " << -3.0*sqrt(21.0)*M_PI/64.0 << ", 0.0, 0.0, 0.0, 0.0, 0.0, " << 2.0*sqrt(3.0/35.0) << ", 0.0, // (3, 0)" << endl; // (3, 0), l'=0~2
    otf << "0.0, 0.0, " << (-273.0/256.0)*sqrt(3.0/2.0)*M_PI << ", 0.0, 0.0, 0.0, 0.0," << endl << endl;                              // (3, 0), l'=3
    
    otf << "0.0, 0.0, 0.0, 0.0, 0.0, 0.0, " << -2.0*sqrt(6.0/35.0) << ", 0.0, " << sqrt(2.0/35.0) << ", // (3, 1)" << endl; // (3, 1), l'=0~2
    otf << "0.0, " << (-147.0/512.0)*sqrt(5.0/2.0)*M_PI << ", 0.0, 0.0, 0.0, 0.0, 0.0,"<< endl << endl;                    // (3, 1), l'=3
    
    otf << "0.0, " << 3.0*sqrt(35.0)*M_PI/128.0 << ", 0.0, 0.0, 0.0, 0.0, 0.0, " << -1.0/(4.0*sqrt(7.0)) << ", 0.0, // (3, 2)" << endl;    // (3, 2), l'=0~2
    otf << (-105.0/512.0)*sqrt(3.0/2.0)*M_PI << ", 0.0, " << (-147.0/512.0)*sqrt(5.0/2.0)*M_PI << ", 0.0, 0.0, 0.0, 0.0," << endl << endl; // (3, 2), l'=3
    
    otf << "0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, " << 1.0/sqrt(42.0) << ", // (3, 3)" << endl;         // (3, 3), l'=0~2
    otf << "0.0, " << (-105.0/512.0)*sqrt(3.0/2.0)*M_PI << ", 0.0, 0.0, 0.0, 0.0, 0.0};" << endl << endl; // (3, 3), l'=3
    
    // <phi_i|grad_y|phi_j>
    otf << "const double piDpj_y[256] = {" << endl;
    otf << "0.0, " << 1.0/(2.0*sqrt(3.0)) << ", 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, // (0, 0)" << endl; // (0, 0), l'=0~2
    otf << "0.0, 0.0, " << (-1.0/2.0)*sqrt(7.0/6.0) << ", 0.0, 0.0, 0.0, 0.0," << endl << endl;        // (0, 0), l'=3
    
    otf << "0.0, 0.0, 0.0, 0.0, 0.0, 0.0, " <<  -sqrt(3.0/5.0) << ", 0.0, " << -1.0/(2.0*sqrt(5.0)) << ", // (1,-1)" << endl; // (1,-1), l'=0~2
    otf << "0.0, " << -3.0*sqrt(35.0)*M_PI/128.0 << ", 0.0, 0.0, 0.0, 0.0, 0.0," << endl << endl;                             // (1,-1), l'=3
    
    otf << "0.0, 0.0, 0.0, " << 3.0*M_PI/8.0 << ", 0.0, " << 1.0/(2.0*sqrt(5.0)) << ", 0.0, 0.0, 0.0, // (1, 0)" << endl; // (1, 0), l'=0~2
    otf << "0.0, 0.0, 0.0, 0.0, " << (3.0/64.0)*sqrt(7.0/2.0)*M_PI << ", 0.0, 0.0," << endl << endl;                      // (1, 0), l'=3
    
    otf << "0.0, 0.0, " << -3.0*M_PI/8.0 << ", 0.0, " << 1.0/(2.0*sqrt(5.0)) << ", 0.0, 0.0, 0.0, 0.0, // (1, 1)" << endl;        // (1, 1), l'=0~2
    otf << "0.0, 0.0, 0.0, " << -3.0*sqrt(21.0)*M_PI/64.0 << ", 0.0, " << -3.0*sqrt(35.0)*M_PI/128.0 << ", 0.0," << endl << endl; // (1, 1), l'=3
    
    otf << "0.0, 0.0, 0.0, " << 1.0/(4.0*sqrt(5.0)) << ", 0.0, " << -15.0*M_PI/64.0 << ", 0.0, 0.0, 0.0, // (2,-2)" << endl; // (2,-2), l'=0~2
    otf << "0.0, 0.0, 0.0, 0.0, " << -2.0*sqrt(2.0/35.0) << ", 0.0, " << -(1.0/2.0)*sqrt(3.0/14.0) << ',' << endl << endl;   // (2,-2), l'=3
    
    otf << "0.0, 0.0, " << -1.0/sqrt(5.0) << ", 0.0, " << 15.0*M_PI/64.0 << ", 0.0, 0.0, 0.0, 0.0, // (2,-1)" << endl; // (2,-1), l'=0~2
    otf << "0.0, 0.0, 0.0, " << -4.0*sqrt(3.0/35.0) << ", 0.0, " << -1.0/(2.0*sqrt(7.0)) << ", 0.0," << endl << endl;  // (2,-1), l'=3
    
    otf << "0.0, " << 1.0/sqrt(15.0) << ", 0.0, 0.0, 0.0, 0.0, 0.0, " << 15.0*sqrt(3.0)*M_PI/32.0 << ", 0.0, // (2, 0)" << endl;  // (2, 0), l'=0~2
    otf << "0.0, 0.0, " << 13.0/(2.0*sqrt(210.0)) << ", 0.0, 0.0, 0.0, 0.0," << endl << endl;                                     // (2, 0), l'=3
    
    otf << "0.0, 0.0, 0.0, 0.0, 0.0, 0.0, " << -15.0*sqrt(3.0)*M_PI/32.0 << ", 0.0, " << 15.0*M_PI/64.0 << ", // (2, 1)" << endl; // (2, 1), l'=0~2
    otf << "0.0, " << 1.0/(2.0*sqrt(7.0)) << ", 0.0, 0.0, 0.0, 0.0, 0.0," << endl << endl;                                        // (2, 1), l'=3
    
    otf << "0.0, " << -1.0/(4*sqrt(5.0)) << ", 0.0, 0.0, 0.0, 0.0, 0.0, " << -15.0*M_PI/64.0 << ", 0.0, // (2, 2)" << endl; // (2, 2), l'=0~2
    otf << (1.0/2.0)*sqrt(3.0/14.0) << ", 0.0, " << 2.0*sqrt(2.0/35.0) << ", 0.0, 0.0, 0.0, 0.0," << endl << endl;          // (2, 2), l'=3
    
    otf << "0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, " << 1.0/sqrt(42.0) << ", // (3,-3)" << endl;        // (3,-3), l'=0~2
    otf << "0.0, " << (-105.0/512.0)*sqrt(3.0/2.0)*M_PI << ", 0.0, 0.0, 0.0, 0.0, 0.0," << endl << endl; // (3,-3), l'=3
    
    otf << "0.0, " << 3.0*sqrt(35.0)*M_PI/128.0 << ", 0.0, 0.0, 0.0, 0.0, 0.0, " << -1.0/(4.0*sqrt(7.0)) << ", 0.0, // (3,-2)" << endl;   // (3,-2), l'=0~2
    otf << (105.0/512.0)*sqrt(3.0/2.0)*M_PI << ", 0.0, " << (-147.0/512.0)*sqrt(5.0/2.0)*M_PI << ", 0.0, 0.0, 0.0, 0.0," << endl << endl; // (3,-2), l'=3
    
    otf << "0.0, 0.0, 0.0, 0.0, 0.0, 0.0, " << -2.0*sqrt(6.0/35.0) << ", 0.0, " << -sqrt(2.0/35.0) << ", // (3,-1)" << endl; // (3,-1), l'=0~2
    otf << "0.0, "<< (147.0/512.0)*sqrt(5.0/2.0)*M_PI << ", 0.0, 0.0, 0.0, 0.0, 0.0," << endl << endl;                       // (3,-1), l'=3
    
    otf << "0.0, 0.0, 0.0, " << 3.0*sqrt(21.0)*M_PI/64.0 << ", 0.0, " << 2.0*sqrt(3.0/35.0) << ", 0.0, 0.0, 0.0, // (3, 0)" << endl; // (3, 0), l'=0~2
    otf << "0.0, 0.0, 0.0, 0.0, " << (273.0/256.0)*sqrt(3.0/2.0)*M_PI << ", 0.0, 0.0," << endl << endl;                              // (3, 0), l'=3
    
    otf << "0.0, 0.0, " << -(3.0/64.0)*sqrt(7.0/2.0)*M_PI << ", 0.0, " << sqrt(2.0/35.0) << ", 0.0, 0.0, 0.0, 0.0, // (3, 1)" << endl;          // (3, 1), l'=0~2
    otf << "0.0, 0.0, 0.0, " << (-273.0/256.0)*sqrt(3.0/2.0)*M_PI << ", 0.0, " << (147.0/512.0)*sqrt(5.0/2.0)*M_PI << ", 0.0," << endl << endl; // (3, 1), l'=3
    
    otf << "0.0, 0.0, 0.0, " << 3.0*sqrt(35.0)*M_PI/128.0 << ", 0.0, " << 1.0/(4.0*sqrt(7.0)) << ", 0.0, 0.0, 0.0, // (3, 2)" << endl;          // (3, 2), l'=0~2
    otf << "0.0, 0.0, 0.0, 0.0, " << (-147.0/512.0)*sqrt(5.0/2.0)*M_PI << ", 0.0, " << (105.0/512.0)*sqrt(3.0/2.0)*M_PI << ',' << endl << endl; // (3, 2), l'=3
    
    otf << "0.0, 0.0, 0.0, 0.0, " << -1.0/sqrt(42.0) << ", 0.0, 0.0, 0.0, 0.0, // (3, 3)" << endl;         // (3, 3), l'=0~2
    otf << "0.0, 0.0, 0.0, 0.0, 0.0, " << (-105.0/512.0)*sqrt(3.0/2.0)*M_PI << ", 0.0};" << endl << endl;  // (3, 3), l'=3
    
    // <phi_i|grad_z|phi_j>
    otf << "const double piDpj_z[64] = { // non-vanishing necessary condition is m'=m" << endl;
    otf << "0.0," << -2.0/sqrt(3.0) << ", 0.0, 0.0, // (0, 0), l'=1" << endl;      // (0, 0), l'=1
    otf << "0.0, 0.0," << -3.0/sqrt(5.0) << ", 0.0, // (1,-1), l'=2" << endl;      // (1,-1), l'=2
    otf << "0.0, 0.0," << -2.0*sqrt(3.0/5.0) << ", 0.0, // (1, 0), l'=2" << endl;  // (1, 0), l'=2
    otf << "0.0, 0.0," << -3.0/sqrt(5.0) << ", 0.0, // (1, 1), l'=2" << endl;      // (1, 1), l'=2
    otf << "0.0, 0.0, 0.0," << -4.0/sqrt(7.0) << ", // (2,-2), l'=3" << endl;      // (2,-2), l'=3
    otf << "0.0, " << 1.0/sqrt(5.0) << ", 0.0," << -8.0*sqrt(2.0/35.0) << ", // (2,-1), l'=1,3" << endl; // (2,-1), l'=1,3
    otf << "0.0, " << 2.0/sqrt(15.0) << ", 0.0," << -12.0/sqrt(35.0) << ", // (2, 0), l'=1,3" << endl;   // (2, 0), l'=1,3
    otf << "0.0, " << 1.0/sqrt(5.0) << ", 0.0," << -8.0*sqrt(2.0/35.0) << ", // (2, 1), l'=1,3" << endl; // (2, 1), l'=1,3
    otf << "0.0, 0.0, 0.0," << -4.0/sqrt(7.0) << ", // (2, 2), l'=3" << endl;      // (2, 2), l'=3
    otf << "0.0, 0.0, 0.0, 0.0,  // (3,-3), all are zero" << endl;                 // (3,-3)
    otf << "0.0, 0.0, " << 2.0/sqrt(7.0) << ", 0.0, // (3,-2), l'=2" << endl;      // (3,-2), l'=2
    otf << "0.0, 0.0, " << 4.0*sqrt(2.0/35.0) << ", 0.0, // (3,-1), l'=2" << endl; // (3,-1), l'=2
    otf << "0.0, 0.0, " << 6.0/sqrt(35.0) << ", 0.0, // (3, 0), l'=2" << endl;     // (3, 0), l'=2
    otf << "0.0, 0.0, " << 4.0*sqrt(2.0/35.0) << ", 0.0, // (3, 1), l'=2" << endl; // (3, 1), l'=2
    otf << "0.0, 0.0, " << 2.0/sqrt(7.0) << ", 0.0, // (3, 2), l'=2" << endl;      // (3, 2), l'=2
    otf << "0.0, 0.0, 0.0, 0.0}; // (3, 3), all are zero" << endl;                 // (3, 3)

    otf << endl << "#endif";
    otf.close();
    //cout << "nn = " << nn << '/' << totlm1lm2LM << endl;
    return 0;
}
