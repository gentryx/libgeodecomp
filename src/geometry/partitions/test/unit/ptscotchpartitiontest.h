#include <boost/assign/std/vector.hpp>
#include <cxxtest/TestSuite.h>

#include <libgeodecomp/geometry/coordbox.h>
#include <libgeodecomp/geometry/partitions/ptscotchpartition.h>

using namespace boost::assign;
using namespace LibGeoDecomp;

namespace LibGeoDecomp {

class PtscotchPartitionTest : public CxxTest::TestSuite
{
public:
    void test2D()
    {
        Coord<2> origin(0, 0);
        Coord<2> dimensions(10, 10);
        std::vector<std::size_t> weights;
        weights << 100 << 100 << 100 << 100;
        PtscotchPartition<2> p(origin, dimensions, 0, weights);

        Region<2> expected0;
        Region<2> expected1;
        Region<2> expected2;
        Region<2> expected3;


        expected0 << CoordBox<2>(Coord<2>( 0, 0), Coord<2>(5,5));
        expected1 << CoordBox<2>(Coord<2>( 5, 0), Coord<2>(5,5));
        expected2 << CoordBox<2>(Coord<2>( 0, 5), Coord<2>(5,5));
        expected3 << CoordBox<2>(Coord<2>( 5, 5), Coord<2>(5,5));


        TS_ASSERT_EQUALS(expected0, p.getRegion(1));
        TS_ASSERT_EQUALS(expected1, p.getRegion(3));
        TS_ASSERT_EQUALS(expected2, p.getRegion(0));
        TS_ASSERT_EQUALS(expected3, p.getRegion(2));
    }

    void testComplete(){
        Coord<2> origin(0, 0);
        Coord<2> dimensions(256, 128);
        std::vector<std::size_t> weights;
        weights << 100 << 100 << 100 << 100;
        PtscotchPartition<2> p(origin, dimensions, 0, weights);

        Region<2> expected0;

        expected0 << CoordBox<2>(Coord<2>(0,0), Coord<2>(256,128));

        Region<2> complete = p.getRegion(0) + p.getRegion(1) + p.getRegion(2) + p.getRegion(3);

        TS_ASSERT_EQUALS(expected0, complete);
     }

     void testOverlapse(){
        Coord<2> origin(0, 0);
        Coord<2> dimensions(128, 231);
        std::vector<std::size_t> weights;
        weights << 100 << 100 << 100 << 100;
        PtscotchPartition<2> p(origin, dimensions, 0, weights);

        Region<2> expected0;

        expected0 << CoordBox<2>(Coord<2>(0,0), Coord<2>(0,0));

        Region<2> cut = p.getRegion(0) & p.getRegion(1) & p.getRegion(2) & p.getRegion(3);

        TS_ASSERT_EQUALS(expected0, cut);
    }

};

}
