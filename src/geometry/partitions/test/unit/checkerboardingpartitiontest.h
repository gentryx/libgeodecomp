#include <boost/assign/std/vector.hpp>
#include <cxxtest/TestSuite.h>

#include <libgeodecomp/geometry/coordbox.h>
#include <libgeodecomp/geometry/partitions/checkerboardingpartition.h>

using namespace boost::assign;
using namespace LibGeoDecomp;

namespace LibGeoDecomp {

class CheckerboardingPartitionTest : public CxxTest::TestSuite
{
public:
    void testBasic()
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

        expected0 << CoordBox<2>(Coord<2>( 0,  0), Coord<2>(10, 10));
        expected1 << CoordBox<2>(Coord<2>(10,  0), Coord<2>(10, 10));
        expected2 << CoordBox<2>(Coord<2>( 0, 10), Coord<2>(10, 10));
        expected3 << CoordBox<2>(Coord<2>(10, 10), Coord<2>(10, 10));

        // TS_ASSERT_EQUALS(expected0, p.getRegion(0));
        // TS_ASSERT_EQUALS(expected1, p.getRegion(1));
        // TS_ASSERT_EQUALS(expected2, p.getRegion(2));
        // TS_ASSERT_EQUALS(expected3, p.getRegion(3));
    }

};

}
