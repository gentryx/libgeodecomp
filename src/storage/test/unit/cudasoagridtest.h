#include <cxxtest/TestSuite.h>
#include <libgeodecomp/config.h>

#ifdef LIBGEODECOMP_WITH_CUDA
#include <cuda.h>
#endif

#include <libgeodecomp/storage/cudasoagrid.h>

using namespace LibGeoDecomp;

namespace LibGeoDecomp {

class CUDASoAGridTest : public CxxTest::TestSuite
{
public:
    void testScalarGetSet()
    {
        Coord<3> dim(100, 50, 30);
        Coord<3> origin(200, 210, 220);
        CoordBox<3> box(origin, dim);

        CUDASoAGrid<TestCellSoA, Topologies::Cube<3>::Topology> grid(box);
    }

};

}
