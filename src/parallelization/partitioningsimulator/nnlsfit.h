#include <libgeodecomp/config.h>
#ifdef LIBGEODECOMP_FEATURE_MPI
#ifndef _libgeodecomp_parallelization_partitioningsimulator_nnlsfit_h_
#define _libgeodecomp_parallelization_partitioningsimulator_nnlsfit_h_

#include <libgeodecomp/misc/commontypedefs.h>
#include <libgeodecomp/misc/supervector.h>
#include <libgeodecomp/parallelization/partitioningsimulator/nnls.h>

namespace LibGeoDecomp {

/**     
 * given an m by n matrix, X, and an m-vector, y,  compute an 
 * n-vector, c, that solves the least squares problem 
 * X * c = y  subject with c[i] >= 0 for i = 0..n-1
 *
 * m is the number of observations
 * n is the number of parameters
 */
class NNLSFit 
{

public:
    typedef SuperVector<DVec> DataPoints;

    NNLSFit(const DataPoints& dataPoints);
    ~NNLSFit();

    /**
     * DataPoints are represented as Vector of Vectors for user convenience
     */
    void refit(const DataPoints& dataPoints);


    DVec coefficients() const;


    double solve(const DVec& inputs) const;

private:
    unsigned _params;
    unsigned _observations;

    double* _X;
    double* _y;
    double* _c;
    double* _working_space_w;
    double* _working_space_zz;
    int* _working_space_index;

    /**
     * initializes _numInputs and _numDataPoints.
     */
    void validateShape(const DataPoints& dataPoints);

};

};

#endif
#endif
