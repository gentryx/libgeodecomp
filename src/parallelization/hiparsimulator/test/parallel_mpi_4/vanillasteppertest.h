#include <cxxtest/TestSuite.h>

#include <libgeodecomp/io/testinitializer.h>
#include <libgeodecomp/misc/testhelper.h>
#include <libgeodecomp/mpilayer/mpilayer.h>
#include <libgeodecomp/parallelization/hiparsimulator/partitionmanager.h>
#include <libgeodecomp/parallelization/hiparsimulator/patchbuffer.h>
#include <libgeodecomp/parallelization/hiparsimulator/vanillastepper.h>

using namespace LibGeoDecomp; 
using namespace HiParSimulator; 

namespace LibGeoDecomp {
namespace HiParSimulator {

class OffsetHelperTest : public CxxTest::TestSuite
{
public:
    void testTorus() {
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

    void testCube() {
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

    void testFoo() {
        std::cout << "test fooooooo\n";

        // Init utility classes
        ghostZoneWidth = 4;
        Coord<3> gridDim(55, 47, 31);
        init.reset(new TestInitializer<3>(gridDim));
        CoordBox<3> box = init->gridBox();
        MPILayer mpiLayer;

        StripingPartition<3> partition(Coord<3>(), box.dimensions);
        SuperVector<unsigned> weights;
        weights << 10000;
        weights << 15000;
        weights << 25000;
        weights << box.dimensions.prod() - weights.sum();

        partitionManager.reset(new MyPartitionManager());
        partitionManager->resetRegions(
            box, 
            new VanillaRegionAccumulator<StripingPartition<3>, 3>(
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
                            stepper->partitionManager().
                            getInnerGhostZoneFragments();
                        Region<3> region;
                        if (m.count(recver) > 0)
                            region = m[recver][ghostZoneWidth];
                        mpiLayer.sendRegion(region, recver);
                    }

                    if (recver == mpiLayer.rank()) {
                        MyPartitionManager::RegionVecMap m =
                            stepper->partitionManager().
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

        // let's go
        // fixme: once working, use this code to reimplement the
        // update group
        checkInnerSet(0, 0);
        stepper->update(1);
        checkInnerSet(1, 1);
        stepper->update(3);
        checkInnerSet(4, 4);

        std::cout << "test baaaaaaaaar\n";
               
        /**
         * handle ghost (if validGhostZoneWidth == 0)
         *   restore ghost (inner and outer)
         *   save inner rim
         *   update ghost
         *   save inner ghost
         *   restore inner rim
         * update kernel
         * perform output
         */
    }

private:
    int ghostZoneWidth;
    boost::shared_ptr<TestInitializer<3> > init;
    boost::shared_ptr<MyPartitionManager> partitionManager;
    boost::shared_ptr<MyStepper> stepper;

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
