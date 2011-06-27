#include <boost/assign/std/vector.hpp>
#include <cxxtest/TestSuite.h>
#include <libgeodecomp/parallelization/hiparsimulator/partitions/stripingpartition.h>
#include <libgeodecomp/parallelization/hiparsimulator/intersectingregionaccumulator.h>

using namespace boost::assign;
using namespace LibGeoDecomp;
using namespace HiParSimulator;

namespace LibGeoDecomp {
namespace HiParSimulator {

class IntersectingRegionAccumulatorTest : public CxxTest::TestSuite 
{
public:
    void testSimple()
    {
        Coord<2> origin(0, 0);
        Coord<2> dimensions(10, 10);
        StripingPartition<2> partition(origin, dimensions);

        unsigned offset = 9;
        SuperVector<unsigned> weights;
        weights += 25, 58, 8;
        Region<2> overlay;
        overlay << Streak<2>(Coord<2>(5, 0), 7)
                << Streak<2>(Coord<2>(5, 1), 7)
                << Streak<2>(Coord<2>(5, 2), 7)
                << Streak<2>(Coord<2>(5, 3), 7)
                << Streak<2>(Coord<2>(5, 4), 8)
                << Streak<2>(Coord<2>(4, 5), 7)
                << Streak<2>(Coord<2>(5, 6), 7)
                << Streak<2>(Coord<2>(5, 7), 7)
                << Streak<2>(Coord<2>(5, 8), 7)
                << Streak<2>(Coord<2>(3, 9), 9);
        // should look like this (w/o cutting)
        // 0: ---------0
        // 1: 0000000000
        // 2: 0000000000
        // 3: 0000111111
        // 4: 1111111111
        // 5: 1111111111
        // 6: 1111111111
        // 7: 1111111111
        // 8: 1111111111
        // 9: 1122222222
        // 
        // overlay:
        // 0: -----XX---
        // 1: -----XX---
        // 2: -----XX---
        // 3: -----XX---
        // 4: -----XXX--
        // 5: ----XXX---
        // 6: -----XX---
        // 7: -----XX---
        // 8: -----XX---
        // 9: ---XXXXXX-
        //
        // should look like this (with cutting)
        // 0: ----------
        // 1: -----00---
        // 2: -----00---
        // 3: -----11---
        // 4: -----111--
        // 5: ----511---
        // 6: -----11---
        // 7: -----11---
        // 8: -----11---
        // 9: ---222222-
        IntersectingRegionAccumulator<StripingPartition<2>, 2> ax(
            overlay, partition, offset, weights);
        
        Region<2> expected0, expected1, expected2;
        expected0 << Streak<2>(Coord<2>(5, 1), 7)
                  << Streak<2>(Coord<2>(5, 2), 7);
        expected1 << Streak<2>(Coord<2>(5, 3), 7)
                  << Streak<2>(Coord<2>(5, 4), 8)
                  << Streak<2>(Coord<2>(4, 5), 7)
                  << Streak<2>(Coord<2>(5, 6), 7)
                  << Streak<2>(Coord<2>(5, 7), 7)
                  << Streak<2>(Coord<2>(5, 8), 7);
        expected2 << Streak<2>(Coord<2>(3, 9), 9);

        TS_ASSERT_EQUALS(expected0, ax.getRegion(0));
        TS_ASSERT_EQUALS(expected1, ax.getRegion(1));
        TS_ASSERT_EQUALS(expected2, ax.getRegion(2));
    }
};

}
}
