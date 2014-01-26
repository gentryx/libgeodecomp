#include <libgeodecomp/config.h>
#include <libgeodecomp/communication/mpilayer.h>
#include <libgeodecomp/geometry/partitions/ptscotchpartition.h>

#include <iostream>

using namespace LibGeoDecomp;

namespace LibGeoDecomp {

class PTScotchPartitionTest : public CxxTest::TestSuite
{
public:
    void testEqual()
    {
#ifdef LIBGEODECOMP_WITH_SCOTCH
        Coord<2> origin(0, 0);
        Coord<2> dimensions( 1023,511);
        std::vector<std::size_t> weights;
        weights << 100 << 100 << 100 << 100;
        PTScotchPartition<2> p(origin, dimensions, 0, weights);
        std::size_t sizeRegion0 = p.getRegion(0).size();
        std::size_t compSize;

        for(int i = 1 ; i < weights.size() ; ++i){
            compSize = p.getRegion(i).size();
            TS_ASSERT(sizeRegion0 == compSize ||
                      sizeRegion0 == compSize - 1 ||
                      sizeRegion0 == compSize + 1);
        }
#endif
    }

    void testComplete()
    {
#ifdef LIBGEODECOMP_WITH_SCOTCH
        Coord<2> origin(0, 0);
        Coord<2> dimensions(543,234);
        std::vector<std::size_t> weights;
        weights << 100 << 100 << 100 << 100;
        PTScotchPartition<2> p(origin, dimensions, 0, weights);
        Region<2> expected0;

        expected0 << CoordBox<2>(Coord<2>(0,0), Coord<2>(543,234));

        Region<2> complete = p.getRegion(0)
            + p.getRegion(1)
            + p.getRegion(2)
            + p.getRegion(3);

        TS_ASSERT_EQUALS(expected0, complete);
#endif
     }

     void testOverlapse()
     {
#ifdef LIBGEODECOMP_WITH_SCOTCH
        Coord<2> origin(0, 0);
        Coord<2> dimensions(128, 231);
        std::vector<std::size_t> weights;
        weights << 100 << 100 << 100 << 100;
        PTScotchPartition<2> p(origin, dimensions, 0, weights);

        Region<2> expected0;

        expected0 << CoordBox<2>(Coord<2>(0,0), Coord<2>(0,0));

        Region<2> cut = p.getRegion(0)
            & p.getRegion(1)
            & p.getRegion(2)
            & p.getRegion(3);

        TS_ASSERT_EQUALS(expected0, cut);
#endif
     }

};

}
