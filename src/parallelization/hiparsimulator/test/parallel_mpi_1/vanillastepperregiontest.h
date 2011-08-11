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

template<class GRID_TYPE1, class GRID_TYPE2, class CELL_TYPE>
class MockPatchBuffer : public PatchBuffer<GRID_TYPE1, GRID_TYPE2, CELL_TYPE>
{
public:

    virtual void get(
        GRID_TYPE2& destinationGrid, 
        const Region<2>& patchRegion, 
        const unsigned& nanoStep) 
    {
        events.push_back(Event(patchRegion, nanoStep));
        PatchBuffer<GRID_TYPE1, GRID_TYPE2, CELL_TYPE>::get(
            destinationGrid,
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

    void setUp()
    {
        // Init simulation area
        ghostZoneWidth = 3;
        init.reset(new TestInitializer<2>(Coord<2>(17, 12)));
        CoordBox<2> rect = init->gridBox();


        // Set up a striping partition. We'll take rank 1, placing us
        // between the two others 0 and 2.
        StripingPartition<2> partition(Coord<2>(0, 0), rect.dimensions);
        SuperVector<unsigned> weights(3);
        weights[0] = 4*17 + 7;
        weights[1] = 2*17 - 1;
        weights[2] = 12*17 - weights[0] - weights[1];
        VanillaRegionAccumulator<StripingPartition<2>, 2> *accu = 
            new VanillaRegionAccumulator<StripingPartition<2>, 2>(
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
        patchBuffer.reset(
            new MockPatchBuffer<DisplacedGrid<TestCell<2> >, 
                                DisplacedGrid<TestCell<2> >, TestCell<2> >());
        stepper.reset(
            new VanillaStepper<TestCell<2>, 2>(partitionManager, init));
        stepper->addPatchProvider(patchBuffer);

        // As a reference the stepper itself is used, which is kind of
        // ugly, but this time we're not decomposing the grid. this
        // kind of behavior is checked in vanillastepperbasittest.h.
        referencePartitionManager.reset(new PartitionManager<2>(rect)); 
        referenceStepper.reset(
            new VanillaStepper<TestCell<2>, 2>(referencePartitionManager, init));
        // referenceStepper->update(ghostZoneWidth);
        // std::cout << "foo3c\n";
        // patchBuffer->pushRequest(
        //     &partitionManager->getOuterRim(), 
        //     ghostZoneWidth);
        // std::cout << "foo3d\n";
        // patchBuffer->put(
        //     referenceStepper->grid(), 
        //     referencePartitionManager->ownRegion(),
        //     ghostZoneWidth);
    }

    void testUpdate()
    {
        // checkRegion(3, 0);
        // stepper->update();

        // checkRegion(2, 1);
        // stepper->update();

        // checkRegion(1, 2);


        // Region<2> r = referencePartitionManager->ownRegion();
        // stepper->update();
        // checkRegion(3, 3);
        // stepper->update(2);
        // checkRegion(1, 5);

        // SuperVector<Event> events = patchBuffer->getEvents();
        // TS_ASSERT_EQUALS(1, events.size());
        // TS_ASSERT_EQUALS(3, events[0].second);
        // TS_ASSERT_EQUALS(partitionManager->getOuterRim(), events[0].first);
    }

private:
    int ghostZoneWidth;
    boost::shared_ptr<TestInitializer<2> > init;
    boost::shared_ptr<PartitionManager<2> > partitionManager;
    boost::shared_ptr<VanillaStepper<TestCell<2>, 2> > stepper;
    boost::shared_ptr<MockPatchBuffer<
                          DisplacedGrid<TestCell<2> >, 
                          DisplacedGrid<TestCell<2> >, 
                          TestCell<2> > > patchBuffer;

    boost::shared_ptr<PartitionManager<2> > referencePartitionManager;
    boost::shared_ptr<VanillaStepper<TestCell<2>, 2> > referenceStepper;

    void checkRegion(
        const unsigned& expansionWidth, 
        const unsigned& expectedStep)
    {
        TS_ASSERT_TEST_GRID_REGION(
            DisplacedGrid<TestCell<2> >, 
            stepper->grid(), 
            partitionManager->ownRegion(expansionWidth),
            expectedStep);
    }

};

}
}
