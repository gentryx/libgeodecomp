#include <libgeodecomp/geometry/partitionmanager.h>
#include <libgeodecomp/geometry/partitions/recursivebisectionpartition.h>
#include <libgeodecomp/geometry/partitions/stripingpartition.h>

#include <boost/assign/std/vector.hpp>

using namespace boost::assign;
using namespace LibGeoDecomp;

namespace LibGeoDecomp {

class PartitionManagerTest : public CxxTest::TestSuite
{
public:
    void setUp()
    {
        dimensions = Coord<2>(20, 20);
        offset = 10 * 20 + 8;
        weights.clear();
        weights += 15, 12, 1, 32, 25, 40, 67;
        /**
         * the grid should look like this: (line no. at beginning of
         * the rows, n means that the corresponding cell belongs to
         * node n)
         *
         * 08: --------------------
         * 09: --------------------
         * 10: --------000000000000
         * 11: 00011111111111123333
         * 12: 33333333333333333333
         * 13: 33333333444444444444
         * 14: 44444444444445555555
         * 15: 55555555555555555555
         * 16: 55555555555556666666
         * 17: 66666666666666666666
         * 18: 66666666666666666666
         * 19: 66666666666666666666
         *
         */
        rank = 4;
        ghostZoneWidth = 3;
        /**
         *
         * the outer ghost zone of node 4 and width 3. "." is its
         * inner set, n means the corresponding cell has distance n to
         * the original region.
         *
         * 08: --------------------
         * 09: --------------------
         * 10: -----333333333333333
         * 11: 33333322222222222222
         * 12: 22222221111111111111
         * 13: 11111111............
         * 14: .............1111111
         * 15: 11111111111111222222
         * 16: 22222222222222233333
         * 17: 3333333333333333----
         * 18: --------------------
         * 19: --------------------
         *
         */
        partition.reset(new StripingPartition<2>(Coord<2>(), dimensions, offset, weights));
        boundingBoxes =
            fakeBoundingBoxes(
                offset,
                weights.size(),
                ghostZoneWidth,
                weights,
                partition);

        boost::shared_ptr<AdjacencyManufacturer<2> > dummyAdjacencyManufacturer(new DummyAdjacencyManufacturer<2>);

        partitionManager.resetRegions(
            dummyAdjacencyManufacturer,
            CoordBox<2>(Coord<2>(), dimensions),
            partition,
            rank,
            ghostZoneWidth);
        partitionManager.resetGhostZones(boundingBoxes);
    }

    void testResetRegionsAndGhostRegionFragments()
    {
        unsigned curOffset = offset;
        for (unsigned i = 0; i < 7; ++i) {
            // 23 because you have to intersect node 6's region with
            // the outer ghost zone. this leaves a fragment of length 22.
            unsigned length = (i != 6)? weights[i] : 23;
            // we're node 4 ourselves, so that non-existent halo can be skipped
            if (i != 4) {
                checkRegion(
                    partitionManager.getOuterGhostZoneFragments()[i].back(),
                    curOffset,
                    curOffset + length,
                    partition);
            }
            curOffset += weights[i];
        }
    }

    void testResetRegionsAndExtendedRegions()
    {
        for (unsigned i = 0; i <= ghostZoneWidth; ++i) {
            checkRegion(partitionManager.getRegion(3, i),
                        (11 - i) * 20 + 16 - i,
                        (13 + i) * 20 +  8 + i,
                        partition);
            checkRegion(partitionManager.getRegion(4, i),
                        (13 - i) * 20 +  8 - i,
                        (14 + i) * 20 + 13 + i,
                        partition);
            checkRegion(partitionManager.getRegion(5, i),
                        (14 - i) * 20 + 13 - i,
                        (16 + i) * 20 + 13 + i,
                        partition);
        }
    }

    void testResetRegionsAndOuterAndInnerRims()
    {
        checkRegion(
            partitionManager.rim(1),
            11 * 20 + 6,
            16 * 20 + 15,
            partition);
    }

    void testResetRegionsAndInnerSets()
    {
        TS_ASSERT(!partitionManager.innerSet(0).empty());
        TS_ASSERT(partitionManager.innerSet(1).empty());
        TS_ASSERT(partitionManager.innerSet(2).empty());
        TS_ASSERT(partitionManager.innerSet(3).empty());
    }

    void testGetOuterRim()
    {
        Region<2> expected = partitionManager.ownRegion(ghostZoneWidth) -
            partitionManager.ownRegion();
        TS_ASSERT_EQUALS(expected, partitionManager.getOuterRim());
    }

    void test3D()
    {
        int ghostZoneWidth = 4;
        CoordBox<3> box(Coord<3>(), Coord<3>(55, 47, 31));

        std::vector<std::size_t> weights;
        weights += 10000, 15000, 25000;
        weights << box.dimensions.prod() - sum(weights);
        boost::shared_ptr<Partition<3> > partition(
            new StripingPartition<3>(Coord<3>(), box.dimensions, 0, weights));

        boost::shared_ptr<AdjacencyManufacturer<3> > dummyAdjacencyManufacturer(new DummyAdjacencyManufacturer<3>);

        PartitionManager<Topologies::Torus<3>::Topology> partitionManager;
        partitionManager.resetRegions(
            dummyAdjacencyManufacturer,
            box,
            partition,
            0,
            ghostZoneWidth);

        std::vector<CoordBox<3> > boundingBoxes;
        for (int i = 0; i < 4; ++i)
            boundingBoxes << partitionManager.getRegion(i, 0).boundingBox();

        partitionManager.resetGhostZones(boundingBoxes);

        Region<3> expected;
        for (int z = 0; z < 3; ++z) {
            for (int y = 0; y < 47; ++y) {
                expected << Streak<3>(Coord<3>(0, y, z), 55);
            }
        }

        for (int y = 0; y < 40; ++y) {
            expected << Streak<3>(Coord<3>(0, y, 3), 55);
        }

        expected << Streak<3>(Coord<3>(0,  40, 3), 45);
        TS_ASSERT_EQUALS(expected, partitionManager.innerSet(0));

        expected.clear();
        for (int z = 1; z < 2; ++z) {
            for (int y = 0; y < 47; ++y) {
                expected << Streak<3>(Coord<3>(0, y, z), 55);
            }
        }

        for (int y = 1; y < 39; ++y) {
            expected << Streak<3>(Coord<3>(0, y, 2), 55);
        }

        expected << Streak<3>(Coord<3>(1,  39, 2), 44);
        TS_ASSERT_EQUALS(expected, partitionManager.innerSet(1));
    }

private:
    Coord<2> dimensions;
    unsigned offset;
    boost::shared_ptr<StripingPartition<2> > partition;
    std::vector<std::size_t> weights;
    unsigned rank;
    unsigned ghostZoneWidth;
    std::vector<CoordBox<2> > boundingBoxes;
    PartitionManager<Topologies::Cube<2>::Topology> partitionManager;

    std::vector<CoordBox<2> > fakeBoundingBoxes(
        unsigned offset,
        unsigned size,
        unsigned ghostZoneWidth,
        const std::vector<std::size_t>& weights,
        const boost::shared_ptr<StripingPartition<2> > partition)
    {
        std::vector<CoordBox<2> > boundingBoxes(size);
        long startOffset = offset;
        long endOffset = offset;

        for (unsigned i = 0; i < size; ++i) {
            endOffset += weights[i];
            Region<2> s;

            for (StripingPartition<2>::Iterator coords = (*partition)[startOffset];
                 coords != (*partition)[endOffset];
                 ++coords) {
                s << *coords;
            }

            s = s.expand(ghostZoneWidth);
            boundingBoxes[i] = s.boundingBox();
            startOffset = endOffset;
        }

        return boundingBoxes;
    }

    template<class PARTITION>
    void checkRegion(
        const Region<2>& region,
        unsigned start,
        unsigned end,
        const boost::shared_ptr<PARTITION> partition)
    {
        std::vector<Coord<2> > expected;
        std::vector<Coord<2> > actual;
        for (typename PARTITION::Iterator i = (*partition)[start];
             i != (*partition)[end];
             ++i)
            expected += *i;
        for (Region<2>::Iterator i = region.begin(); i != region.end(); ++i)
            actual += *i;

        TS_ASSERT_EQUALS(expected, actual);
    }
};

}
