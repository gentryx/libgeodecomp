#include <cxxtest/TestSuite.h>
#include <libgeodecomp/io/testinitializer.h>
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

    void test3d()
    {
#ifdef LIBGEODECOMP_FEATURE_CUDA
        typedef TestInitializer<TestCell<3> > TestCellInitializer;
        typedef TestCellInitializer::Topology Topology;

        Coord<3> dim(35, 20, 20);
        CoordBox<3> box(Coord<3>(), dim);
        TestCellInitializer init;

        DisplacedGrid<TestCell<3>, Topology> source(box);
        DisplacedGrid<TestCell<3>, Topology> target(box);
        init.grid(&source);
        CUDAGrid<TestCell<3>, Topology> buffer(box);

        Region<3> region;
        for (int y = 0; y < 10; ++y) {
            region << Streak<3>(Coord<3>(y, y + 10, 2 * y), 25 + y / 2);
        }

        for (CoordBox<3>::Iterator i = box.begin(); i != box.end(); ++i) {
            TS_ASSERT_DIFFERS(source[*i], target[*i]);
        }

        buffer.loadRegion(source,  region);
        buffer.saveRegion(&target, region);

        for (CoordBox<3>::Iterator i = box.begin(); i != box.end(); ++i) {
            TestCell<3> expected ;
            if (region.count(*i)) {
                expected = source[*i];
            }

            TS_ASSERT_EQUALS(expected, target[*i]);
        }

#endif
    }

    // fixme: test topological correctness
};

}
