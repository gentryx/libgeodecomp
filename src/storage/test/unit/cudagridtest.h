#include <cxxtest/TestSuite.h>
#include <libgeodecomp/storage/cudagrid.h>
#include <libgeodecomp/storage/displacedgrid.h>

#include <cuda.h>

using namespace LibGeoDecomp;

namespace LibGeoDecomp {

class CUDAGridTest : public CxxTest::TestSuite
{
public:

    void testBasic()
    {
#ifdef LIBGEODECOMP_FEATURE_CUDA

        Coord<2> dim(30, 10);
        CoordBox<2> box(Coord<2>(), dim);

        DisplacedGrid<int> source(box, -1);
        DisplacedGrid<int> target(box, -2);
        CUDAGrid<int> buffer(box);

        Region<2> region;
        for (int y = 0; y < 10; ++y) {
            region << Streak<2>(Coord<2>(10 + y, y), 25 + y / 2);
        }

        int counter = 0;
        for (Region<2>::Iterator i = region.begin(); i != region.end(); ++i) {
            source[*i] = counter++;
        }

        buffer.loadRegion(source,  region);
        buffer.saveRegion(&target, region);

        counter = 0;
        for (CoordBox<2>::Iterator i = box.begin(); i != box.end(); ++i) {
            int expected = -2;
            if (region.count(*i)) {
                expected = counter++;
            }

            TS_ASSERT_EQUALS(target[*i], expected);
        }

#endif
    }

    // fixme: 3d test, test topological correctness
};

}
