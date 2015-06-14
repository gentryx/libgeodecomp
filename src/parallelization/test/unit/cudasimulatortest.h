#include <cuda.h>
#include <cxxtest/TestSuite.h>
#include <libgeodecomp/parallelization/cudasimulator.h>

using namespace LibGeoDecomp;

namespace LibGeoDecomp {

class CudaSimulatorTest : public CxxTest::TestSuite
{
public:
    void testBasic()
    {
        std::cout << "----------------------------------------------\n";
    }
};

}
