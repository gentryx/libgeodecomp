#include <libgeodecomp/parallelization/nesting/offsethelper.h>

#include <boost/assign/std/vector.hpp>
#include <cxxtest/TestSuite.h>

using namespace boost::assign;
using namespace LibGeoDecomp;

namespace LibGeoDecomp {

class OffsetHelperTest : public CxxTest::TestSuite
{
public:
    void testTorus()
    {
        Coord<2> offset;
        Coord<2> dimensions;
        OffsetHelper<1, 2, Topologies::Torus<2>::Topology>()(
            &offset,
            &dimensions,
            CoordBox<2>(Coord<2>(1, 1),
                        Coord<2>(5, 3)),
            CoordBox<2>(Coord<2>(0, 0),
                        Coord<2>(10, 8)),
            2);
        TS_ASSERT_EQUALS(Coord<2>(-1, -1), offset);
        TS_ASSERT_EQUALS(Coord<2>(9, 7), dimensions);
    }

    void testCube()
    {
        Coord<2> offset;
        Coord<2> dimensions;
        OffsetHelper<1, 2, Topologies::Cube<2>::Topology>()(
            &offset,
            &dimensions,
            CoordBox<2>(Coord<2>(1, 1),
                        Coord<2>(6, 3)),
            CoordBox<2>(Coord<2>(0, 0),
                        Coord<2>(8, 8)),
            2);
        TS_ASSERT_EQUALS(Coord<2>(0, 0), offset);
        TS_ASSERT_EQUALS(Coord<2>(8, 6), dimensions);
    }
};

}
