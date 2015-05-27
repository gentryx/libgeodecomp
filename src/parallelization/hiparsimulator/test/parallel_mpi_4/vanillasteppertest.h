#include <cxxtest/TestSuite.h>

#include <libgeodecomp.h>
#include <libgeodecomp/communication/mpilayer.h>
#include <libgeodecomp/communication/patchlink.h>
#include <libgeodecomp/geometry/partitionmanager.h>
#include <libgeodecomp/io/testinitializer.h>
#include <libgeodecomp/misc/testhelper.h>
#include <libgeodecomp/parallelization/hiparsimulator/vanillastepper.h>

#include <boost/assign/std/vector.hpp>

using namespace boost::assign;
using namespace LibGeoDecomp;
using namespace HiParSimulator;

namespace LibGeoDecomp {
namespace HiParSimulator {

class VanillaStepperTest : public CxxTest::TestSuite
{
public:
    typedef APITraits::SelectTopology<TestCell<3> >::Value Topology;
    typedef PartitionManager<Topology> PartitionManagerType;
    typedef VanillaStepper<TestCell<3> > StepperType;
    typedef PatchLink<StepperType::GridType> PatchLinkType;
    typedef boost::shared_ptr<PatchLinkType::Accepter> PatchAccepterPtrType;
    typedef boost::shared_ptr<PatchLinkType::Provider> PatchProviderPtrType;

    void setUp()
    {
        mpiLayer.reset(new MPILayer());
    }

    void tearDown()
    {
        stepper.reset();
        mpiLayer.reset();
    }

    void testFoo()
    {
        // Init utility classes
        ghostZoneWidth = 4;
        Coord<3> gridDim(55, 47, 31);
        init.reset(new TestInitializer<TestCell<3> >(gridDim));
        CoordBox<3> box = init->gridBox();

        std::vector<std::size_t> weights;
        weights += 10000, 15000, 25000;
        weights << box.dimensions.prod() - sum(weights);
        boost::shared_ptr<Partition<3> > partition(
            new StripingPartition<3>(Coord<3>(), box.dimensions, 0, weights));

        partitionManager.reset(new PartitionManagerType());
        partitionManager->resetRegions(
            box,
            partition,
            mpiLayer->rank(),
            ghostZoneWidth);

        std::vector<CoordBox<3> > boundingBoxes;
        for (int i = 0; i < 4; ++i) {
            boundingBoxes << partitionManager->getRegion(i, 0).boundingBox();
        }
        partitionManager->resetGhostZones(boundingBoxes);

        stepper.reset(new StepperType(partitionManager, init));

        // verify that the grids got set up properly
        Coord<3> expectedOffset;
        Coord<3> expectedDimensions;
        switch (mpiLayer->rank()) {
        case 0:
            expectedOffset = Coord<3>(0, 0, -4);
            expectedDimensions = Coord<3>(55, 47, 12);
            break;
        case 1:
            expectedOffset = Coord<3>(0, 0, -1);
            expectedDimensions = Coord<3>(55, 47, 15);
            break;
        case 2:
            expectedOffset = Coord<3>(0, 0,  5);
            expectedDimensions = Coord<3>(55, 47, 19);
            break;
        case 3:
            expectedOffset = Coord<3>(0, 0, 15);
            expectedDimensions = Coord<3>(55, 47, 20);
            break;
        default:
            expectedOffset = Coord<3>(-1, -1, -1);
            expectedDimensions = Coord<3>(-1 , -1, -1);
            break;
        }

        TS_ASSERT_EQUALS(expectedOffset,
                         stepper->oldGrid->getOrigin());
        TS_ASSERT_EQUALS(expectedDimensions,
                         stepper->oldGrid->getDimensions());
        TS_ASSERT_EQUALS(gridDim,
                         stepper->oldGrid->topologicalDimensions());

        // ensure that the ghostzones we're about to send/receive do
        // actually match
        for (int sender = 0; sender < mpiLayer->size(); ++sender) {
            for (int recver = 0; recver < mpiLayer->size(); ++recver) {
                if (sender != recver) {
                    if (sender == mpiLayer->rank()) {
                        PartitionManagerType::RegionVecMap m =
                            stepper->partitionManager->getInnerGhostZoneFragments();
                        Region<3> region;
                        if (m.count(recver) > 0)
                            region = m[recver][ghostZoneWidth];
                        mpiLayer->sendRegion(region, recver);
                    }

                    if (recver == mpiLayer->rank()) {
                        PartitionManagerType::RegionVecMap m =
                            stepper->partitionManager->getOuterGhostZoneFragments();
                        Region<3> expected;
                        if (m.count(sender) > 0)
                            expected = m[sender][ghostZoneWidth];
                        Region<3> actual;
                        mpiLayer->recvRegion(&actual, sender);
                        TS_ASSERT_EQUALS(actual, expected);
                    }
                }
            }
        }

        int tag = 4711;

        std::vector<PatchProviderPtrType> providers;
        std::vector<PatchAccepterPtrType> accepters;

        // manually set up patch links for ghost zone communication
        PartitionManagerType::RegionVecMap m;
        m = partitionManager->getOuterGhostZoneFragments();
        for (PartitionManagerType::RegionVecMap::iterator i = m.begin(); i != m.end(); ++i) {
            if (i->first != PartitionManagerType::OUTGROUP) {
                Region<3>& region = i->second[ghostZoneWidth];
                if (!region.empty()) {
                    PatchProviderPtrType p(
                        new PatchLinkType::Provider(
                            region,
                            i->first,
                            tag,
                            Typemaps::lookup<TestCell<3> >()));
                    providers << p;
                    stepper->addPatchProvider(p, StepperType::GHOST);
                }
            }
        }

        m = partitionManager->getInnerGhostZoneFragments();
        for (PartitionManagerType::RegionVecMap::iterator i = m.begin(); i != m.end(); ++i) {
            if (i->first != PartitionManagerType::OUTGROUP) {
                Region<3>& region = i->second[ghostZoneWidth];
                if (!region.empty()) {
                    PatchAccepterPtrType p(
                        new PatchLinkType::Accepter(
                            region,
                            i->first,
                            tag,
                            Typemaps::lookup<TestCell<3> >()));
                    accepters << p;
                    stepper->addPatchAccepter(p, StepperType::GHOST);
                }
            }
        }

        // add events to patchlinks
        for (std::vector<PatchProviderPtrType>::iterator i = providers.begin();
             i != providers.end();
             ++i) {
            (*i)->charge(ghostZoneWidth, ghostZoneWidth * 5, ghostZoneWidth);
        }

        for (std::vector<PatchAccepterPtrType>::iterator i = accepters.begin();
             i != accepters.end();
             ++i) {
            (*i)->charge(ghostZoneWidth, ghostZoneWidth * 5, ghostZoneWidth);
        }

        // need to re-init after PatchLinks have been added since
        // initGrids() will also re-update the ghost zone. during that
        // update the patch accepters will be notified, which is
        // required to get the patch communication going.
        stepper->initGrids();

        // let's go
        checkInnerSet(0, 0);

        stepper->update(1);
        checkInnerSet(1, 1);

        stepper->update(3);
        checkInnerSet(0, 4);

        stepper->update(11);
        checkInnerSet(3, 15);
    }

private:
    int ghostZoneWidth;
    boost::shared_ptr<TestInitializer<TestCell<3> > > init;
    boost::shared_ptr<PartitionManagerType> partitionManager;
    boost::shared_ptr<StepperType> stepper;
    boost::shared_ptr<MPILayer> mpiLayer;

    void checkInnerSet(
        const unsigned& shrink,
        const unsigned& expectedStep)
    {
        TS_ASSERT_TEST_GRID_REGION(
            StepperType::GridType,
            stepper->grid(),
            partitionManager->innerSet(shrink),
            expectedStep);
    }

    void checkRim(
        const unsigned& shrink,
        const unsigned& expectedStep)
    {
        TS_ASSERT_TEST_GRID_REGION(
            StepperType::GridType,
            stepper->grid(),
            partitionManager->rim(shrink),
            expectedStep);
    }

};

}
}
