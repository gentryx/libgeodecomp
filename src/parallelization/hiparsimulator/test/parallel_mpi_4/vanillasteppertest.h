#include <cxxtest/TestSuite.h>

#include <libgeodecomp/io/testinitializer.h>
#include <libgeodecomp/misc/testhelper.h>
#include <libgeodecomp/mpilayer/mpilayer.h>
#include <libgeodecomp/parallelization/hiparsimulator/partitionmanager.h>
#include <libgeodecomp/parallelization/hiparsimulator/patchlink.h>
#include <libgeodecomp/parallelization/hiparsimulator/vanillastepper.h>

using namespace LibGeoDecomp; 
using namespace HiParSimulator; 

namespace LibGeoDecomp {
namespace HiParSimulator {

class OffsetHelperTest : public CxxTest::TestSuite
{
public:
    void testTorus() 
    {
        Coord<2> offset;
        Coord<2> dimensions;
        OffsetHelper<1, 2, Topologies::Torus<2>::Topology>()(
            &offset,
            &dimensions,
            CoordBox<2>(Coord<2>(1, 1),
                        Coord<2>(5, 3)),
            CoordBox<2>(Coord<2>(0, 0),
                        Coord<2>(10, 8)),
            2);
        TS_ASSERT_EQUALS(Coord<2>(-1, -1), offset);
        TS_ASSERT_EQUALS(Coord<2>(9, 7), dimensions);
    }

    void testCube() 
    {
        Coord<2> offset;
        Coord<2> dimensions;
        OffsetHelper<1, 2, Topologies::Cube<2>::Topology>()(
            &offset,
            &dimensions,
            CoordBox<2>(Coord<2>(1, 1),
                        Coord<2>(6, 3)),
            CoordBox<2>(Coord<2>(0, 0),
                        Coord<2>(8, 8)),
            2);
        TS_ASSERT_EQUALS(Coord<2>(0, 0), offset);
        TS_ASSERT_EQUALS(Coord<2>(8, 6), dimensions);
    }
};

class VanillaStepperTest : public CxxTest::TestSuite
{
public:
    typedef PartitionManager<3, TestCell<3>::Topology> MyPartitionManager;
    typedef VanillaStepper<TestCell<3> > MyStepper;
    typedef PatchLink<MyStepper::GridType> MyPatchLink;
    typedef boost::shared_ptr<MyPatchLink::Accepter> MyPatchAccepterPtr;
    typedef boost::shared_ptr<MyPatchLink::Provider> MyPatchProviderPtr;

    void tearDown()
    {
        stepper.reset();
    }

    void testFoo() 
    {
        // Init utility classes
        ghostZoneWidth = 4;
        Coord<3> gridDim(55, 47, 31);
        init.reset(new TestInitializer<3>(gridDim));
        CoordBox<3> box = init->gridBox();

        StripingPartition<3> partition(Coord<3>(), box.dimensions);
        SuperVector<unsigned> weights;
        weights << 10000;
        weights << 15000;
        weights << 25000;
        weights << box.dimensions.prod() - weights.sum();

        partitionManager.reset(new MyPartitionManager());
        partitionManager->resetRegions(
            box, 
            new VanillaRegionAccumulator<StripingPartition<3> >(
                partition,
                0,
                weights),
            mpiLayer.rank(),
            ghostZoneWidth);

        SuperVector<CoordBox<3> > boundingBoxes;
        for (int i = 0; i < 4; ++i)
            boundingBoxes << partitionManager->getRegion(i, 0).boundingBox();
        partitionManager->resetGhostZones(boundingBoxes);
       
        stepper.reset(new MyStepper(partitionManager, init));

        // verify that the grids got set up properly
        Coord<3> expectedOffset;
        Coord<3> expectedDimensions;
        switch (mpiLayer.rank()) {
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
        for (int sender = 0; sender < mpiLayer.size(); ++sender) {
            for (int recver = 0; recver < mpiLayer.size(); ++recver) {
                if (sender != recver) {
                    if (sender == mpiLayer.rank()) {
                        MyPartitionManager::RegionVecMap m =
                            stepper->getPartitionManager().
                            getInnerGhostZoneFragments();
                        Region<3> region;
                        if (m.count(recver) > 0)
                            region = m[recver][ghostZoneWidth];
                        mpiLayer.sendRegion(region, recver);
                    }

                    if (recver == mpiLayer.rank()) {
                        MyPartitionManager::RegionVecMap m =
                            stepper->getPartitionManager().
                            getOuterGhostZoneFragments();
                        Region<3> expected;
                        if (m.count(sender) > 0)
                            expected = m[sender][ghostZoneWidth];
                        Region<3> actual;
                        mpiLayer.recvRegion(&actual, sender);
                        TS_ASSERT_EQUALS(actual, expected);
                    }
                }
            }
        }

        int tag = 4711;

        // fixme: once working, use this code to reimplement the
        // update group

        SuperVector<MyPatchProviderPtr> providers;
        SuperVector<MyPatchAccepterPtr> accepters;

        // manually set up patch links for ghost zone communication
        MyPartitionManager::RegionVecMap m;
        m = partitionManager->getOuterGhostZoneFragments();
        for (MyPartitionManager::RegionVecMap::iterator i = m.begin(); i != m.end(); ++i) {
            if (i->first != MyPartitionManager::OUTGROUP) {
                Region<3>& region = i->second[ghostZoneWidth];
                if (!region.empty()) {
                    MyPatchProviderPtr p(
                        new MyPatchLink::Provider(
                            region, 
                            i->first,
                            tag));                
                    providers << p;
                    stepper->addPatchProvider(p, MyStepper::GHOST);
                } 
            }
        }
         
        m = partitionManager->getInnerGhostZoneFragments();  
        for (MyPartitionManager::RegionVecMap::iterator i = m.begin(); i != m.end(); ++i) {
            if (i->first != MyPartitionManager::OUTGROUP) {
                Region<3>& region = i->second[ghostZoneWidth];
                if (!region.empty()) {
                    MyPatchAccepterPtr p(
                        new MyPatchLink::Accepter(
                            region,
                            i->first,
                            tag));
                    accepters << p;
                    stepper->addPatchAccepter(p, MyStepper::GHOST);
                } 
            }
        }

        // add events to patchlinks
        for (SuperVector<MyPatchProviderPtr>::iterator i = providers.begin();
             i != providers.end();
             ++i) {
            (*i)->charge(ghostZoneWidth, ghostZoneWidth * 5, ghostZoneWidth);
        }

        for (SuperVector<MyPatchAccepterPtr>::iterator i = accepters.begin();
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
        //fixme: check rim, too
        stepper->update(1);
        checkInnerSet(1, 1);

        stepper->update(3);
        checkInnerSet(0, 4);

        stepper->update(11);
        checkInnerSet(3, 15);
    }

private:
    int ghostZoneWidth;
    boost::shared_ptr<TestInitializer<3> > init;
    boost::shared_ptr<MyPartitionManager> partitionManager;
    boost::shared_ptr<MyStepper> stepper;
    MPILayer mpiLayer;

    void checkInnerSet(
        const unsigned& shrink, 
        const unsigned& expectedStep)
    {
        TS_ASSERT_TEST_GRID_REGION(
            MyStepper::GridType, 
            stepper->grid(), 
            partitionManager->innerSet(shrink),
            expectedStep);
    }

    void checkRim(
        const unsigned& shrink, 
        const unsigned& expectedStep)
    {
        TS_ASSERT_TEST_GRID_REGION(
            MyStepper::GridType, 
            stepper->grid(), 
            partitionManager->rim(shrink),
            expectedStep);
    }

};

}
}
