#include <cxxtest/TestSuite.h>
#include <libgeodecomp/geometry/region.h>
#include <libgeodecomp/geometry/topologies.h>
#include <libgeodecomp/parallelization/nesting/offsethelper.h>

using namespace LibGeoDecomp;

namespace LibGeoDecomp {

class OffsetHelperTest : public CxxTest::TestSuite
{
public:
    void testTorusWithRegionNearOrigin()
    {
        Region<3> region;
        region << CoordBox<3>(Coord<3>(  0,   0,   0), Coord<3>(10, 10, 10));

        const CoordBox<3> boundingBox = region.boundingBox();
        const CoordBox<3> gridBox = CoordBox<3>(Coord<3>(0, 0, 0), Coord<3>(200, 200, 200));
        const int ghostZoneWidth = 3;

        Coord<3> offset;
        Coord<3> dimensions;

        OffsetHelper<3 - 1, 3, Topologies::Torus<3>::Topology>()(
            &offset,
            &dimensions,
            boundingBox,
            gridBox,
            ghostZoneWidth);

        TS_ASSERT_EQUALS(Coord<3>(-3, -3, -3), offset);
        TS_ASSERT_EQUALS(Coord<3>(16, 16, 16), dimensions);
    }

    void testTorusWithRegionCloseToFarCorner()
    {
        Region<3> region;
        region << CoordBox<3>(Coord<3>(190, 190, 190), Coord<3>(10, 10, 10));

        const CoordBox<3> boundingBox = region.boundingBox();
        const CoordBox<3> gridBox = CoordBox<3>(Coord<3>(0, 0, 0), Coord<3>(200, 200, 200));
        const int ghostZoneWidth = 4;

        Coord<3> offset;
        Coord<3> dimensions;

        OffsetHelper<3 - 1, 3, Topologies::Torus<3>::Topology>()(
            &offset,
            &dimensions,
            boundingBox,
            gridBox,
            ghostZoneWidth);

        TS_ASSERT_EQUALS(Coord<3>(186, 186, 186), offset);
        TS_ASSERT_EQUALS(Coord<3>( 18,  18,  18), dimensions);
    }
};

}
