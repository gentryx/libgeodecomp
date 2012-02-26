#include <boost/assign/std/vector.hpp>
#include <cxxtest/TestSuite.h>
#include <libgeodecomp/parallelization/hiparsimulator/partitions/stripingpartition.h>
#include <libgeodecomp/parallelization/hiparsimulator/vanillaregionaccumulator.h>


using namespace boost::assign;
using namespace LibGeoDecomp;
using namespace HiParSimulator;

namespace LibGeoDecomp {
namespace HiParSimulator {

class VanillaRegionAccumulatorTest : public CxxTest::TestSuite 
{
public:
    void testSimple()
    {
        Coord<2> origin(0, 0);
        Coord<2> dimensions(10, 10);
        StripingPartition<2> partition(origin, dimensions);


        unsigned offset = 9;
        SuperVector<long> weights;
        weights += 25, 58, 8;
        // should look like this:
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
        VanillaRegionAccumulator<StripingPartition<2> > ax(partition, offset, weights);
        
        Region<2> expected0, expected1, expected2;
        expected0 << Streak<2>(Coord<2>(9, 0), 10)
                  << Streak<2>(Coord<2>(0, 1), 10)
                  << Streak<2>(Coord<2>(0, 2), 10)
                  << Streak<2>(Coord<2>(0, 3),  4);
        expected1 << Streak<2>(Coord<2>(4, 3), 10)
                  << Streak<2>(Coord<2>(0, 4), 10)
                  << Streak<2>(Coord<2>(0, 5), 10)
                  << Streak<2>(Coord<2>(0, 6), 10)
                  << Streak<2>(Coord<2>(0, 7), 10)
                  << Streak<2>(Coord<2>(0, 8), 10)
                  << Streak<2>(Coord<2>(0, 9),  2);
        expected2 << Streak<2>(Coord<2>(2, 9), 10);

        TS_ASSERT_EQUALS(expected0, ax.getRegion(0));
        TS_ASSERT_EQUALS(expected1, ax.getRegion(1));
        TS_ASSERT_EQUALS(expected2, ax.getRegion(2));
    }
};

}
}
