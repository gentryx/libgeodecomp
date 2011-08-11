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
    typedef VanillaStepper<TestCell<3>, 3> MyStepper;

    void testFoo() {
        std::cout << "test fooooooo\n";

        // Init utility classes
        ghostZoneWidth = 4;
        init.reset(new TestInitializer<3>(Coord<3>(55, 47, 31)));
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

        for (int i = 0; i < mpiLayer.size(); ++i) {
            mpiLayer.barrier();
            if (i == mpiLayer.rank())
                std::cout << "ownRegion[0]: " << partitionManager->ownRegion(0).boundingBox() 
                          << "ownRegion[1]: " << partitionManager->ownRegion(1).boundingBox()
        //                   << "ownRegExpand: " << partitionManager->ownRegion(0).expand(1).boundingBox()
        //                   << "simulationAr: " << partitionManager->simulationArea.boundingBox()
        //                   << "box: " << box 
                          << "ownExpandedRegion: " << partitionManager->ownExpandedRegion().boundingBox() << "\n";
            mpiLayer.barrier();
        }

        // Let's go
        MyStepper stepper(
            partitionManager,
            init);

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
        
        TS_ASSERT_EQUALS(expectedOffset, stepper.offset);
        TS_ASSERT_EQUALS(expectedDimensions, stepper.dimensions);
        
        for (int i = 0; i < mpiLayer.size(); ++i) {
            mpiLayer.barrier();
            if (i == mpiLayer.rank()) {
                std::cout << "offset[" << i << "]: " << stepper.offset << "\n";
                std::cout << "dimensions[" << i << "]: " << stepper.dimensions << "\n";
            }
            mpiLayer.barrier();
        }

        // boost::shared_ptr<
        //     PatchBuffer<MyStepper::GridType, MyStepper::GridType, TestCell<3> > > p1(
        //         new PatchBuffer<MyStepper::GridType, MyStepper::GridType, TestCell<3> >);
        // boost::shared_ptr< PatchAccepter<MyStepper::GridType> > p2(p1);
        // p1->pushRequest(&partitionManager->rim(ghostZoneWidth), 0);
        // stepper.addPatchAccepter(p1);

        // stepper.update(1);
        
        // MyStepper::GridType g(init->gridDimensions());
        // p1->get(g, partitionManager->rim(ghostZoneWidth), 0);

        std::cout << "test baaaaaaaaar\n";
               
        /**
         * update kernel
         * perform output
         * handle ghost
         *   restore ghost
         *   save inner rim
         *   update ghost
         *   save inner ghost
         *   restore inner rim
         */
    }

private:
    int ghostZoneWidth;
    boost::shared_ptr<TestInitializer<3> > init;
    boost::shared_ptr<MyPartitionManager> partitionManager;
    boost::shared_ptr<VanillaStepper<TestCell<3>, 3> > stepper;

};

}
}
