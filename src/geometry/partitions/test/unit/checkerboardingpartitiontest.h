#include <libgeodecomp/geometry/coordbox.h>
#include <libgeodecomp/geometry/partitions/checkerboardingpartition.h>

#include <boost/assign/std/vector.hpp>
#include <cxxtest/TestSuite.h>

using namespace boost::assign;
using namespace LibGeoDecomp;

namespace LibGeoDecomp {

class CheckerboardingPartitionTest : public CxxTest::TestSuite
{
public:
    void test3D()
    {
        Coord<3> origin(0, 0, 0);
        Coord<3> dimensions(20, 20, 20);
        std::vector<std::size_t> weights;
        weights << 100 << 100 << 100 << 100 << 100 << 100 << 100 << 100;
        CheckerboardingPartition<3> p(origin, dimensions, 0, weights);

        Region<3> expected0;
        Region<3> expected1;
        Region<3> expected2;
        Region<3> expected3;
        Region<3> expected4;
        Region<3> expected5;
        Region<3> expected6;
        Region<3> expected7;


        expected0 << CoordBox<3>(Coord<3>( 0, 0, 0), Coord<3>(10,10,10));
        expected1 << CoordBox<3>(Coord<3>(10, 0, 0), Coord<3>(10,10,10));
        expected2 << CoordBox<3>(Coord<3>( 0,10, 0), Coord<3>(10,10,10));
        expected3 << CoordBox<3>(Coord<3>(10,10, 0), Coord<3>(10,10,10));
        expected4 << CoordBox<3>(Coord<3>( 0, 0,10), Coord<3>(10,10,10));
        expected5 << CoordBox<3>(Coord<3>(10, 0,10), Coord<3>(10,10,10));
        expected6 << CoordBox<3>(Coord<3>( 0,10,10), Coord<3>(10,10,10));
        expected7 << CoordBox<3>(Coord<3>(10,10,10), Coord<3>(10,10,10));


        TS_ASSERT_EQUALS(expected0, p.getRegion(0));
        TS_ASSERT_EQUALS(expected1, p.getRegion(1));
        TS_ASSERT_EQUALS(expected2, p.getRegion(2));
        TS_ASSERT_EQUALS(expected3, p.getRegion(3));
        TS_ASSERT_EQUALS(expected4, p.getRegion(4));
        TS_ASSERT_EQUALS(expected5, p.getRegion(5));
        TS_ASSERT_EQUALS(expected6, p.getRegion(6));
        TS_ASSERT_EQUALS(expected7, p.getRegion(7));
    }

    void test2D()
    {
        Coord<2> origin(0, 0);
        Coord<2> dimensions(20, 20);
        std::vector<std::size_t> weights;
        weights << 100 << 100 << 100 << 100;
        CheckerboardingPartition<2> p(origin, dimensions, 0, weights);

        Region<2> expected0;
        Region<2> expected1;
        Region<2> expected2;
        Region<2> expected3;


        expected0 << CoordBox<2>(Coord<2>( 0, 0), Coord<2>(10,10));
        expected1 << CoordBox<2>(Coord<2>(10, 0), Coord<2>(10,10));
        expected2 << CoordBox<2>(Coord<2>( 0,10), Coord<2>(10,10));
        expected3 << CoordBox<2>(Coord<2>(10,10), Coord<2>(10,10));


        TS_ASSERT_EQUALS(expected0, p.getRegion(0));
        TS_ASSERT_EQUALS(expected1, p.getRegion(1));
        TS_ASSERT_EQUALS(expected2, p.getRegion(2));
        TS_ASSERT_EQUALS(expected3, p.getRegion(3));
    }

    void test2DwithMoreNodesAndOffset()
    {
        Coord<2> origin(10, 20);
        Coord<2> dimensions(32, 25);
        std::vector<std::size_t> weights;
        for (int i = 0; i < 20; ++i) {
            weights << 1;
        }

        CheckerboardingPartition<2> p(origin, dimensions, 0, weights);

        for (int y = 0; y < 5; ++y) {
            for (int x = 0; x < 4; ++x) {
                Region<2> expected;
                expected << CoordBox<2>(origin + Coord<2>(x * 8, y * 5), Coord<2>(8, 5));

                TS_ASSERT_EQUALS(expected, p.getRegion(y * 4 + x));
            }
        }
    }

    void test3DwithNonEvenDivisions()
    {
        Coord<3> origin(10, 20, 30);
        Coord<3> dimensions(29, 31, 37);
        std::vector<std::size_t> weights;
        for (int i = 0; i < 400; ++i) {
            weights << 1;
        }

        CheckerboardingPartition<3> p(origin, dimensions, 0, weights);

        for (int z = 0; z < 10; ++z) {
            for (int y = 0; y < 8; ++y) {
                for (int x = 0; x < 5; ++x) {
                    Region<3> expected;
                    int xStart = (x + 0) * dimensions.x() / 5;
                    int xEnd   = (x + 1) * dimensions.x() / 5;
                    int yStart = (y + 0) * dimensions.y() / 8;
                    int yEnd   = (y + 1) * dimensions.y() / 8;
                    int zStart = (z + 0) * dimensions.z() / 10;
                    int zEnd   = (z + 1) * dimensions.z() / 10;
                    Coord<3> upperLeftFrontCorner(xStart, yStart, zStart);
                    Coord<3> lowerRightRearCorner(xEnd,   yEnd,   zEnd);

                    expected << CoordBox<3>(
                        origin + upperLeftFrontCorner,
                        lowerRightRearCorner - upperLeftFrontCorner);

                    TS_ASSERT_EQUALS(expected, p.getRegion(z * 5 * 8 + y * 5 + x));
                }
            }
        }
    }
};

}
