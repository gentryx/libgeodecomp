#include <cxxtest/TestSuite.h>
#include <libgeodecomp/geometry/cudaregion.h>

#include <cuda.h>

using namespace LibGeoDecomp;

namespace LibGeoDecomp {

class CUDARegionTest : public CxxTest::TestSuite
{
public:

    void test2D()
    {
        std::cout << "&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&\n";
        // fixme:
        // - fill Region
        // - load CUDARegion
        // - iterate through CUDARegion and set elements of CUDAGrid
        // - copy CUDAGrid to DisplacedGrid
        // - check contents
    }

    void test3D()
    {
        // fixme: equivalent test to test2D required
    }

};

}
