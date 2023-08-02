#ifndef sym_h
#define sym_h

#include <mpi.h>
#include <omp.h>
#include <cmath>

#include "const.h"
#include "fn.h"
#include "math.h"

using namespace std;

void AxisTheta2Rmat(const double *axis, const double theta, const double det,
                    double *a[], double *b[],
                    double *rmat, double *aRb);

class operatorclass {
    public:
    // field
    int no;               // #
    int det;              // determinant 1 or -1
    double theta;         // rotation angle
    double *axis = NULL;  // rotation axis, dimension of 3, normalized
    double *tau  = NULL;   // fractional transition, dimension of 3

    double *rmat = NULL;  // rotation matrix, dimension of 3 x 3
    double *aRb  = NULL;  // (a0 a1 a2)^T R (b0 b1 b2), dimension of 3 x 3, see details in sym.cpp
    double *aRTb = NULL;  // (a0 a1 a2^T) R^T (b0 b1 b2), R^T = R^-1

    //method
    //constructor
    operatorclass(); // dummy
    void Initial(const int in_no, const int in_det, const double in_theta,
                 const double in_ux, const double in_uy, const double in_uz,
                 const double in_taux, const double in_tauy, const double in_tauz,
                 double *a[], double *b[]); bool isInitial = false;
    //destructor
    ~operatorclass();
};

int FindOpNum(int *gijk, const int time_reve_opt,
              const double *q, const double *q1, operatorclass *optcs, const int numopt);

#endif
