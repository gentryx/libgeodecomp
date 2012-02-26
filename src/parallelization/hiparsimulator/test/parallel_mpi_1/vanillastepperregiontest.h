#include <boost/shared_ptr.hpp>
#include <cxxtest/TestSuite.h>

#include <libgeodecomp/io/testinitializer.h>
#include <libgeodecomp/misc/testhelper.h>
#include <libgeodecomp/parallelization/hiparsimulator/patchbuffer.h>
#include <libgeodecomp/parallelization/hiparsimulator/vanillastepper.h>

using namespace LibGeoDecomp; 
using namespace HiParSimulator; 

namespace LibGeoDecomp {
namespace HiParSimulator {

typedef std::pair<Region<2>, unsigned> Event;

template<class GRID_TYPE1, class GRID_TYPE2>
class MockPatchBuffer : public PatchBuffer<GRID_TYPE1, GRID_TYPE2>
{
public:

    virtual void get(
        GRID_TYPE2& destinationGrid, 
        const Region<2>& patchRegion, 
        const unsigned& nanoStep) 
    {
        events.push_back(Event(patchRegion, nanoStep));
        PatchBuffer<GRID_TYPE1, GRID_TYPE2>::get(
            &destinationGrid,
            patchRegion,
            nanoStep);
    }

    inline
    const SuperVector<Event>& getEvents() const
    {
        return events;
    }

private:
    SuperVector<Event> events;
};

class VanillaStepperRegionTest : public CxxTest::TestSuite
{
public:
    typedef VanillaStepper<TestCell<2> > MyStepper;
    typedef MyStepper::GridType GridType;

    void setUp()
    {
        // Init simulation area
        ghostZoneWidth = 3;
        init.reset(new TestInitializer<2>(Coord<2>(17, 12)));
        CoordBox<2> rect = init->gridBox();


        // Set up a striping partition. We'll take rank 1, placing us
        // between the two others 0 and 2.
        StripingPartition<2> partition(Coord<2>(0, 0), rect.dimensions);
        SuperVector<long> weights(3);
        weights[0] = 4*17 + 7;
        weights[1] = 2*17 - 1;
        weights[2] = 12*17 - weights[0] - weights[1];
        VanillaRegionAccumulator<StripingPartition<2> > *accu = 
            new VanillaRegionAccumulator<StripingPartition<2> >(
                partition,
                0,
                weights);

        // Feed the partition into the partition manager
        partitionManager.reset(new PartitionManager<2>());
        partitionManager->resetRegions(
            init->gridBox(),
            accu,
            1,
            ghostZoneWidth);
        SuperVector<CoordBox<2> > boundingBoxes;
        for (int i = 0; i < 3; ++i)
            boundingBoxes.push_back(accu->getRegion(i).boundingBox());
        partitionManager->resetGhostZones(boundingBoxes);

        // The Unit Under Test: the stepper
        stepper.reset(new MyStepper(partitionManager, &*init));
    }

    void testUpdate()
    {
        checkInnerSet(0, 0);
        stepper->update();
        checkInnerSet(1, 1);
        stepper->update();
        checkInnerSet(2, 2);
    }

private:
    int ghostZoneWidth;
    boost::shared_ptr<TestInitializer<2> > init;
    boost::shared_ptr<PartitionManager<2> > partitionManager;
    boost::shared_ptr<MyStepper > stepper;

    void checkInnerSet(
        const unsigned& shrink, 
        const unsigned& expectedStep)
    {
        TS_ASSERT_TEST_GRID_REGION(
            GridType, 
            stepper->grid(), 
            partitionManager->innerSet(shrink),
            expectedStep);
    }
};

}
}
