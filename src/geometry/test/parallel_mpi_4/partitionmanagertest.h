#include <libgeodecomp/communication/mpilayer.h>
#include <libgeodecomp/geometry/partitionmanager.h>
#include <libgeodecomp/geometry/partitions/unstructuredstripingpartition.h>
#include <libgeodecomp/io/unstructuredtestinitializer.h>

using namespace LibGeoDecomp;

namespace LibGeoDecomp {

class PartitionManagerTest : public CxxTest::TestSuite
{
public:
    void setUp()
    {
        layer.reset(new MPILayer());
    }

    void tearDown()
    {
        layer.reset();
    }

    void testUnstructured()
    {
        typedef Topologies::Unstructured::Topology Topology;
        CoordBox<1> box(Coord<1>(0), Coord<1>(614));
        std::vector<std::size_t> weights;
        weights << 153
                << 154
                << 153
                << 154;
        int ghostZoneWidth = 1;
        boost::shared_ptr<Partition<1> > partition(new UnstructuredStripingPartition(Coord<1>(), Coord<1>(), 0, weights));
        boost::shared_ptr<AdjacencyManufacturer<1> > init(new UnstructuredTestInitializer<UnstructuredTestCell<> >(614, 100, 0));

        PartitionManager<Topology> manager(box);
        manager.resetRegions(
            init,
            box,
            partition,
            layer->rank(),
            ghostZoneWidth);

        std::vector<CoordBox<1> > boundingBoxes;
        std::vector<CoordBox<1> > expandedBoundingBoxes;

        for (int i = 0; i < 4; ++i) {
            Region<1> region = partition->getRegion(i);
            Region<1> expandedRegion = region.expandWithTopology(
                ghostZoneWidth,
                box.dimensions,
                Topology(),
                *init->getAdjacency(region));

            boundingBoxes << region.boundingBox();
            expandedBoundingBoxes << expandedRegion.boundingBox();
        }

        manager.resetGhostZones(boundingBoxes, expandedBoundingBoxes);

        typedef PartitionManager<Topology>::RegionVecMap RegionVecMap;
        RegionVecMap outerFragments = manager.getOuterGhostZoneFragments();
        RegionVecMap innerFragments = manager.getInnerGhostZoneFragments();

        TS_ASSERT_EQUALS(2, outerFragments.size());
        TS_ASSERT_EQUALS(2, innerFragments.size());

        TS_ASSERT_EQUALS(2, outerFragments[PartitionManager<Topology>::OUTGROUP].size());
        TS_ASSERT_EQUALS(2, innerFragments[PartitionManager<Topology>::OUTGROUP].size());

        TS_ASSERT_EQUALS(2, outerFragments[(layer->rank() + 1    ) % 4].size());
        TS_ASSERT_EQUALS(2, innerFragments[(layer->rank() - 1 + 4) % 4].size());
    }

private:
    boost::shared_ptr<MPILayer> layer;
};

}
