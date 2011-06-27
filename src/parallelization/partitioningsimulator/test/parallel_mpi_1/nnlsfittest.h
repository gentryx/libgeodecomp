#include <boost/assign/std/vector.hpp>
#include <cxxtest/TestSuite.h>
#include "../../../../misc/testhelper.h"
#include "../../nnlsfit.h"

using namespace boost::assign;
using namespace LibGeoDecomp; 

namespace LibGeoDecomp {

class NNLSFitTest : public CxxTest::TestSuite
{
private:
    NNLSFit* _fit;
    NNLSFit::DataPoints* _dataPoints;

public:

    void setUp()
    {
        DVec v1, v2;
        v1 += 1, 0, 1;
        v2 += 1, 2, 2;
        _dataPoints = new NNLSFit::DataPoints();
        _dataPoints->push_back(v1);
        _dataPoints->push_back(v2);
        _fit = new NNLSFit(*_dataPoints);
    }


    void tearDown()
    {
        delete _fit;
        delete _dataPoints;
    }


    void testInvalidShapeThrows()
    {
        TS_ASSERT_THROWS(NNLSFit(NNLSFit::DataPoints(0)), std::invalid_argument);
        TS_ASSERT_THROWS(NNLSFit(NNLSFit::DataPoints(2, DVec())), std::invalid_argument);
        TS_ASSERT_THROWS(NNLSFit(NNLSFit::DataPoints(3, DVec(5))), std::invalid_argument);
    }


    void testNNLSFitThrows()
    {
        TS_ASSERT_THROWS(_fit->refit(NNLSFit::DataPoints(3)), std::invalid_argument);
        TS_ASSERT_THROWS(_fit->refit(NNLSFit::DataPoints(2, DVec(4))), 
                std::invalid_argument);
    }


    void testSolve()
    {
        DVec v;
        v += 1, 4;
        TS_ASSERT_THROWS(_fit->solve(DVec(3)), std::invalid_argument);
        TS_ASSERT_EQUALS_DOUBLE(_fit->solve(v), 3);
    }


    void testCoefficients()
    {
        DVec v;
        v += 1, 0.5;
        TS_ASSERT_EQUALS_DVEC(_fit->coefficients(), v);
    }
    
};

};
