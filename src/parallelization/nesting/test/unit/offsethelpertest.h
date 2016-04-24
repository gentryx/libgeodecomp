#include <cxxtest/TestSuite.h>
#include <libgeodecomp/geometry/region.h>
#include <libgeodecomp/geometry/topologies.h>
#include <libgeodecomp/parallelization/nesting/offsethelper.h>

using namespace LibGeoDecomp;

namespace LibGeoDecomp {

class OffsetHelperTest : public CxxTest::TestSuite
{
public:
    void testTorus2D()
    {
        Coord<2> offset;
        Coord<2> dimensions;

        Region<2> region;
        region << CoordBox<2>(Coord<2>(1, 1),
                              Coord<2>(5, 3));
        region = region.expand(2);

        OffsetHelper<1, 2, Topologies::Torus<2>::Topology>()(
            &offset,
            &dimensions,
            region,
            CoordBox<2>(Coord<2>(0, 0),
                        Coord<2>(10, 8)));

        TS_ASSERT_EQUALS(Coord<2>(-1, -1), offset);
        TS_ASSERT_EQUALS(Coord<2>(9, 7), dimensions);
    }

    void testTorus3DWithRegionNearOrigin()
    {
        Region<3> region;
        region << CoordBox<3>(Coord<3>(  0,   0,   0), Coord<3>(10, 10, 10));

        const CoordBox<3> gridBox = CoordBox<3>(Coord<3>(0, 0, 0), Coord<3>(200, 200, 200));
        const int ghostZoneWidth = 3;
        region = region.expand(ghostZoneWidth);

        Coord<3> offset;
        Coord<3> dimensions;

        OffsetHelper<3 - 1, 3, Topologies::Torus<3>::Topology>()(
            &offset,
            &dimensions,
            region,
            gridBox);

        TS_ASSERT_EQUALS(Coord<3>(-3, -3, -3), offset);
        TS_ASSERT_EQUALS(Coord<3>(16, 16, 16), dimensions);
    }

    void testTorus3DWithRegionCloseToFarCorner()
    {
        Region<3> region;
        region << CoordBox<3>(Coord<3>(190, 190, 190), Coord<3>(10, 10, 10));

        const CoordBox<3> gridBox = CoordBox<3>(Coord<3>(0, 0, 0), Coord<3>(200, 200, 200));
        const int ghostZoneWidth = 4;
        region = region.expand(ghostZoneWidth);

        Coord<3> offset;
        Coord<3> dimensions;

        OffsetHelper<3 - 1, 3, Topologies::Torus<3>::Topology>()(
            &offset,
            &dimensions,
            region,
            gridBox);

        TS_ASSERT_EQUALS(Coord<3>(186, 186, 186), offset);
        TS_ASSERT_EQUALS(Coord<3>( 18,  18,  18), dimensions);
    }

    void testCube2D()
    {
        Coord<2> offset;
        Coord<2> dimensions;

        Region<2> region;
        region << CoordBox<2>(Coord<2>(1, 1),
                              Coord<2>(6, 3));
        region = region.expand(2);

        OffsetHelper<1, 2, Topologies::Cube<2>::Topology>()(
            &offset,
            &dimensions,
            region,
            CoordBox<2>(Coord<2>(0, 0),
                        Coord<2>(8, 8)));

        TS_ASSERT_EQUALS(Coord<2>(0, 0), offset);
        TS_ASSERT_EQUALS(Coord<2>(8, 6), dimensions);
    }
};

}
