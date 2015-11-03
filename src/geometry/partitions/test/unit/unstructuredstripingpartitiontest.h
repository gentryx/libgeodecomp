#include <libgeodecomp/geometry/partitions/unstructuredstripingpartition.h>

#include <boost/assign/std/vector.hpp>
#include <cxxtest/TestSuite.h>

using namespace LibGeoDecomp;

namespace LibGeoDecomp {

class UnstructuredStripingPartitionTest : public CxxTest::TestSuite
{
public:
    void testSingleDomain()
    {
        std::vector<std::size_t> weights;
        // empty:
        weights << 0;
        Region<1> expected;
        TS_ASSERT_EQUALS(UnstructuredStripingPartition(Coord<1>(0), Coord<1>(0), 0, weights).getRegion(0), expected);

        // with non-zero origin:
        weights[0] = 100;
        expected << Streak<1>(Coord<1>(50), 150);
        TS_ASSERT_EQUALS(UnstructuredStripingPartition(Coord<1>(50), Coord<1>(100), 0, weights).getRegion(0), expected);

        // with origin and offset:
        weights[0] = 200;
        expected.clear();
        expected << Streak<1>(Coord<1>(100), 300);
        TS_ASSERT_EQUALS(UnstructuredStripingPartition(Coord<1>(50), Coord<1>(250), 50, weights).getRegion(0), expected);
    }

    void testMultipleDomains()
    {
        std::vector<std::size_t> weights;
        weights <<  50
                << 100
                <<  50;
        Region<1> expected0;
        Region<1> expected1;
        Region<1> expected2;
        expected0 << Streak<1>(Coord<1>(  0),  50);
        expected1 << Streak<1>(Coord<1>( 50), 150);
        expected2 << Streak<1>(Coord<1>(150), 200);

        UnstructuredStripingPartition partition(Coord<1>(0), Coord<1>(200), 0, weights);
        TS_ASSERT_EQUALS(partition.getRegion(0), expected0);
        TS_ASSERT_EQUALS(partition.getRegion(1), expected1);
        TS_ASSERT_EQUALS(partition.getRegion(2), expected2);
    }
};

}
