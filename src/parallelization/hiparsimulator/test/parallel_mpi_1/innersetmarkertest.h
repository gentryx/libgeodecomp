#include <cxxtest/TestSuite.h>
#include <libgeodecomp/misc/superset.h>
#include <libgeodecomp/parallelization/hiparsimulator/partitions/stripingpartition.h>
#include <libgeodecomp/parallelization/hiparsimulator/innersetmarker.h>
#include <libgeodecomp/parallelization/hiparsimulator/vanillaregionaccumulator.h>

using namespace LibGeoDecomp;
using namespace HiParSimulator;

namespace LibGeoDecomp {
namespace HiParSimulator {

class InnerSetMarkerTest : public CxxTest::TestSuite 
{
public:
    void setUp()
    {
        intersector = PartitionManager<2>();
        dimensions = Coord<2>(20, 400);
        partition = StripingPartition<2>(Coord<2>(0, 0), dimensions);

        // assume this is a dual cluster setup and on the current
        // level we're in the second cluster which is responsible for
        // the (dimensions.y() - offset / dimensions.x()) last lines of
        // the StripingPartition.
        startLine = 130;
        offset = startLine * dimensions.x();
        ghostZoneWidth = 6;
        rank = 0;
        weights = SuperVector<unsigned> (9, 30 * dimensions.x());
        weights[3] = 40 * dimensions.x();
        weights[5] = 20 * dimensions.x();
        // sanity check
        TS_ASSERT_EQUALS(weights.sum() + offset, dimensions.x() * dimensions.y());

        intersector.resetRegions(
            CoordBox<2>(Coord<2>(), dimensions),
            new VanillaRegionAccumulator<StripingPartition<2>, 2>(
                partition,
                offset,
                weights),
            rank,
            ghostZoneWidth);
    }

    void testBasic()
    {
        InnerSetMarker<2> marker(intersector);
        SuperSet<Coord<2> > expected, actual;

        for (unsigned i = 0; i <= ghostZoneWidth; ++i) {
            for (int y = startLine + i; 
                 y < startLine + weights[0] / dimensions.x() - i;
                 ++y)
                for (int x = 0; x < dimensions.x(); ++x)
                    expected.insert(Coord<2>(x, y));
            for (Region<2>::Iterator c = marker.begin(i); c != marker.end(i); ++c)
                actual.insert(*c);
            TS_ASSERT_EQUALS(expected, actual);
        }
    }

private:
    PartitionManager<2> intersector;
    StripingPartition<2> partition;
    Coord<2> dimensions;
    SuperVector<unsigned> weights;
    unsigned startLine;
    unsigned offset;
    unsigned ghostZoneWidth;
    unsigned rank;

};

}
}
