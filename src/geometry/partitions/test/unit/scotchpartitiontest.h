#include <boost/assign/std/vector.hpp>
#include <cxxtest/TestSuite.h>

#include <libgeodecomp/geometry/coordbox.h>
#include <libgeodecomp/geometry/partitions/scotchpartition.h>

using namespace boost::assign;
using namespace LibGeoDecomp;

namespace LibGeoDecomp {

class ScotchPartitionTest : public CxxTest::TestSuite
{
public:
    void test2D()
    {
        Coord<2> origin(0, 0);
        Coord<2> dimensions(10000, 10000);
        std::vector<std::size_t> weights;
        weights << 100 << 100 << 100 << 100;
        ScotchPartition<2> p(origin, dimensions, 0, weights);

        Region<2> expected0;
        Region<2> expected1;
        Region<2> expected2;
        Region<2> expected3;


        expected0 << CoordBox<2>(Coord<2>( 0, 0), Coord<2>(500,500));
        expected1 << CoordBox<2>(Coord<2>( 500, 0), Coord<2>(500,500));
        expected2 << CoordBox<2>(Coord<2>( 0,500), Coord<2>(500,500));
        expected3 << CoordBox<2>(Coord<2>( 500,500), Coord<2>(500,500));


        /*TS_ASSERT_EQUALS(expected0, p.getRegion(1));
        TS_ASSERT_EQUALS(expected1, p.getRegion(3));
        TS_ASSERT_EQUALS(expected2, p.getRegion(0));
        TS_ASSERT_EQUALS(expected3, p.getRegion(2));*/


    }
};

}
