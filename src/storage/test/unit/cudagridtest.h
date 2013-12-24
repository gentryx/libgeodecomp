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

        Coord<3> dim1(35, 20, 20);
        Coord<3> dim2(75, 40, 40);
        CoordBox<3> box1(Coord<3>(), dim1);
        CoordBox<3> box2(Coord<3>(-10, -10, -10), dim2);
        TestCellInitializer init;

        DisplacedGrid<TestCell<3>, Topology> source(box1);
        DisplacedGrid<TestCell<3>, Topology> target(box2);
        init.grid(&source);
        CUDAGrid<TestCell<3>, Topology> buffer(box1);

        Region<3> region;
        for (int y = 0; y < 10; ++y) {
            region << Streak<3>(Coord<3>(y, y + 10, 2 * y), 25 + y / 2);
        }

        for (CoordBox<3>::Iterator i = box2.begin(); i != box2.end(); ++i) {
            TS_ASSERT_DIFFERS(source[*i], target[*i]);
        }

        buffer.loadRegion(source,  region);
        buffer.saveRegion(&target, region);

        for (CoordBox<3>::Iterator i = box2.begin(); i != box2.end(); ++i) {
            TestCell<3> expected ;
            if (region.count(*i)) {
                expected = source[*i];
            }

            TS_ASSERT_EQUALS(expected, target[*i]);
        }

#endif
    }

    void testTopologicalCorrectness()
    {
#ifdef LIBGEODECOMP_FEATURE_CUDA

        // simulation space: (0,0), (100, 100),
        //
        // coordinates covered by topologically correct grids are
        // marked with "x" below. North-west corner marked with "A",
        // south-west as "B", north-east as "C", and south-east as
        // "D":
        //
        //     0123456789 ...               90 ...  99
        //  0: xxxxxxxxxxxxxxxxxxxx   ...   xxxxxxxxxx
        //  1: xxxxxxxxxxxxxxxxxxxx   ...   xxxxxxxxxx
        //  2: xxxxxxxxxxxxxxxxxxxx   ...   xxxxxxxxxx
        //  3: xxxxxxxxxxxxxxxxxxxx   ...   xxxxxxxxxx
        //  4: xxxxxxxxxxxxxxxxxxxx   ...   xxxxxxxxxx
        //  5: xxxxxxxxxxxxxxxxxxxx   ...   xxxxxxxxxx
        //  6: xxxxxxxxxxxxxxxxxxxD   ...   Bxxxxxxxxx
        //  7:
        //  8:
        //
        // ...
        //
        // 95:
        // 96:
        // 97: xxxxxxxxxxxxxxxxxxxC   ...   Axxxxxxxxx
        // 98: xxxxxxxxxxxxxxxxxxxx   ...   xxxxxxxxxx
        // 99: xxxxxxxxxxxxxxxxxxxx   ...   xxxxxxxxxx


        Coord<2> dim(30, 10);
        Coord<2> offset(90, 97);
        Coord<2> topologicalDim(100, 100);
        CoordBox<2> box(offset, dim);

        CoordBox<2> globalBox(Coord<2>(), topologicalDim);

        DisplacedGrid<double> source(globalBox, -1);
        DisplacedGrid<double, Topologies::Torus<2>::Topology, true> target(box, -2, 0, topologicalDim);
        CUDAGrid<double, Topologies::Torus<2>::Topology, true> buffer(box, topologicalDim);

        Region<2> region;
        // cover all 4 sectors sketched out above:
        region << Streak<2>(Coord<2>( 0,  0),  20);
        region << Streak<2>(Coord<2>(90,  6), 100);
        region << Streak<2>(Coord<2>( 5, 97),  10);
        region << Streak<2>(Coord<2>(95, 99),  97);

        for (Region<2>::Iterator i = region.begin(); i != region.end(); ++i) {
            Coord<2> c = Topologies::Torus<2>::Topology::normalize(*i, topologicalDim);

            source[*i] = 1000.0 + c.x() * 100.0 + c.y() / 100.0;
        }

        buffer.loadRegion(source,  region);
        buffer.saveRegion(&target, region);

        for (CoordBox<2>::Iterator i = box.begin(); i != box.end(); ++i) {
            Coord<2> c = Topologies::Torus<2>::Topology::normalize(*i, topologicalDim);

            double expected = -2;
            if (region.count(c)) {
                expected =  1000.0 + c.x() * 100.0 + c.y() / 100.0;
            }

            TS_ASSERT_EQUALS(target[c], expected);
        }

#endif
    }
};

}
