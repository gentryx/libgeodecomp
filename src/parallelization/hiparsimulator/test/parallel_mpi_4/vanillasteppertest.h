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

class VanillaStepperTest : public CxxTest::TestSuite
{
public:
    
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

        partitionManager.reset(new PartitionManager<3>());
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

        for (int i = 0; i < MPILayer().size(); ++i) {
            MPILayer().barrier();
            if (i == MPILayer().rank())
                std::cout << "ownRegion[0]: " << partitionManager->ownRegion(0).boundingBox() 
                          << "ownRegion[1]: " << partitionManager->ownRegion(1).boundingBox()
                          << "ownRegExpand: " << partitionManager->ownRegion(0).expand(1).boundingBox()
                          << "simulationAr: " << partitionManager->simulationArea.boundingBox()
                          << "box: " << box 
                          << "ownExpandedRegion: " << partitionManager->ownExpandedRegion().boundingBox() << "\n";
            MPILayer().barrier();
        }

        // Let's go
        MyStepper stepper(
            partitionManager,
            init);

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
    boost::shared_ptr<PartitionManager<3> > partitionManager;
    boost::shared_ptr<VanillaStepper<TestCell<3>, 3> > stepper;

};

}
}
