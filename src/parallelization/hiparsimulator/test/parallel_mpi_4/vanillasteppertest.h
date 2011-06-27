#include <cxxtest/TestSuite.h>

#include <libgeodecomp/io/testinitializer.h>
#include <libgeodecomp/misc/testhelper.h>
#include <libgeodecomp/mpilayer/mpilayer.h>
#include <libgeodecomp/parallelization/hiparsimulator/partitionmanager.h>
#include <libgeodecomp/parallelization/hiparsimulator/vanillastepper.h>

using namespace LibGeoDecomp; 
using namespace HiParSimulator; 

namespace LibGeoDecomp {
namespace HiParSimulator {

class VanillaStepperTest : public CxxTest::TestSuite
{
public:
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

        // Let's go
        VanillaStepper<TestCell<3>, 3> stepper(
            partitionManager,
            init);
               
        /**
         * save inner rim
         * update ghost
         * save inner ghost
         * restore inner rim
         * update kernel
         * restore ghost
         * perform output
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
