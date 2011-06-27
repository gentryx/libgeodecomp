#include <boost/assign/std/vector.hpp>
#include <libgeodecomp/io/testinitializer.h>
#include <libgeodecomp/parallelization/hiparsimulator/parallelstepper.h>
#include <libgeodecomp/parallelization/hiparsimulator/partitionmanager.h>
#include <libgeodecomp/parallelization/hiparsimulator/partitions/stripingpartition.h>
#include <libgeodecomp/parallelization/hiparsimulator/vanillaregionaccumulator.h>
#include <libgeodecomp/misc/testhelper.h>

using namespace boost::assign;
using namespace LibGeoDecomp;
using namespace HiParSimulator;

namespace LibGeoDecomp {
namespace HiParSimulator {

class ParallelStepperTest : public CxxTest::TestSuite 
{
public:
    typedef ParallelStepper<TestCell<2> > MyParallelStepper;

    void setUp()
    {
        // width = 131;
        // height = 241;
        // maxSteps = 1500;
        // firstStep = 20;
        // firstNanoStep = 18;
        // firstCycle = firstStep * TestCell::nanoSteps() + firstNanoStep;
        // init.reset(new TestInitializer(width, height, maxSteps, firstStep, firstNanoStep));
        // outputPeriod = 17;
        // loadBalancingPeriod = 31;
        // ghostZoneWidth = 10;
        // offset = 0;
        // SuperVector<unsigned> initialWeights;
        // initialWeights += 
        //     50 * 131, 1 * 131, 10 * 131,
        //     5 * 131, 15 * 131, 20 * 131,
        //     2 * 131, 48 * 131, 90 * 131;
        // partitionManager.reset(new PartitionManager());
        // partitionManager->resetRegions(
        //     CoordRectangle(Coord(0, 0), init->gridWidth(), init->gridHeight()),
        //     new VanillaRegionAccumulator<StripingPartition>(
        //         StripingPartition(Coord(0, 0), Coord(init->gridWidth(), init->gridHeight())),
        //         offset,
        //         initialWeights),
        //     MPILayer().rank(),
        //     ghostZoneWidth);

        // s.reset(new MyParallelStepper());
        // s->resetRegions(
        //     &*partitionManager, 
        //     &*init);
    // }

    // void tearDown()
    // {
    //     partitionManager.reset();
    //     s.reset(); 
    }

    void testUpdateGhostRegionMin()
    {
    //     s->updateGhostZones(1, firstCycle);
    //     TS_ASSERT_TEST_GRID_REGION(DisplacedGrid<TestCell>, *s->newGrid, partitionManager->rim(ghostZoneWidth), firstCycle + 1);
    // }

    // void testUpdateGhostRegionMax()
    // {
    //     s->updateGhostZones(ghostZoneWidth, firstCycle);
    //     TS_ASSERT_TEST_GRID_REGION(DisplacedGrid<TestCell>, *s->newGrid, partitionManager->rim(ghostZoneWidth), firstCycle + ghostZoneWidth);
    // }

    // void testUpdateInnerSet()
    // {
    //     s->updateInnerSet(0, ghostZoneWidth, firstCycle);
    //     TS_ASSERT_TEST_GRID_REGION(DisplacedGrid<TestCell>, *s->newGrid, partitionManager->innerSet(ghostZoneWidth), firstCycle + ghostZoneWidth);
    // }

    // void testUpdateInnerSetWithIntermediateStep()
    // {        
    //     unsigned pausestep = 5;
    //     s->updateInnerSet(0, pausestep, firstCycle);
    //     std::swap(s->oldGrid, s->newGrid);
    //     s->updateInnerSet(pausestep, ghostZoneWidth - pausestep, firstCycle + pausestep);
    //     TS_ASSERT_TEST_GRID_REGION(DisplacedGrid<TestCell>, *s->newGrid, partitionManager->innerSet(ghostZoneWidth), firstCycle + ghostZoneWidth);
    // }

    // void testUpdateGhostRegionAndInnerSetTogether()
    // {
    //     s->updateGhostZones(ghostZoneWidth, firstCycle);
    //     s->updateInnerSet(0, ghostZoneWidth, firstCycle);
    //     TS_ASSERT_TEST_GRID_REGION(DisplacedGrid<TestCell>, *s->newGrid, partitionManager->ownRegion(), firstCycle + ghostZoneWidth);
    // }

    // void testSendRecvGhostZones()
    // {
    //     unsigned comGhostZoneWidth = 3;
    //     for (Region::Iterator i = partitionManager->ownExpandedRegion().begin();
    //          i != partitionManager->ownExpandedRegion().end();
    //          ++i)
    //         (*s->newGrid)[*i].testValue = s->mpiLayer.rank() + 0.123;

    //     s->sendGhostZones(comGhostZoneWidth);
    //     s->recvGhostZones(comGhostZoneWidth);
    //     s->waitForGhostZones();

    //     for (PartitionManager::RegionVecMap::const_iterator i = 
    //              partitionManager->getOuterGhostZoneFragments().begin(); 
    //          i != partitionManager->getOuterGhostZoneFragments().end(); ++i) {
    //         if (i->first != PartitionManager::OUTGROUP) {
    //             Region fragment = i->second[comGhostZoneWidth];
    //             for (Region::Iterator pos = fragment.begin(); 
    //                  pos != fragment.end(); ++pos) 
    //                 TS_ASSERT_EQUALS_DOUBLE(
    //                     i->first + 0.123, (*s->newGrid)[*pos].testValue);
    //         }
    //     }
    // }

    // void testNanoStep1()
    // {
    //     unsigned curStop = firstNanoStep + 1;
    //     unsigned nextStop = firstNanoStep + 10;
    //     s->nanoStep(curStop, nextStop, firstNanoStep);
    //     TS_ASSERT_TEST_GRID_REGION(
    //         DisplacedGrid<TestCell>, 
    //         *s->getGrid(), 
    //         partitionManager->innerSet(ghostZoneWidth), 
    //         firstCycle + 1);
    // }

    // void testNanoStepMore()
    // {
    //     unsigned curStop = firstNanoStep + 30;
    //     unsigned nextStop = firstNanoStep + 40;
    //     s->nanoStep(curStop, nextStop, firstNanoStep);
    //     TS_ASSERT_TEST_GRID_REGION(
    //         DisplacedGrid<TestCell>, 
    //         *s->getGrid(), 
    //         partitionManager->innerSet(ghostZoneWidth), 
    //         firstCycle + 30);

    //     curStop = firstNanoStep + 40;
    //     nextStop = firstNanoStep + 60;
    //     s->nanoStep(curStop, nextStop, firstNanoStep + 30);
    //     TS_ASSERT_TEST_GRID_REGION(
    //         DisplacedGrid<TestCell>, 
    //         *s->getGrid(), 
    //         partitionManager->innerSet(ghostZoneWidth), 
    //         firstCycle + 40);
    // }

    // void testGhostZoneRegistration()
    // {
    //     for (MyParallelStepper::MPIRegionVecMap::iterator i = 
    //              s->innerGhostZoneFragments.begin();
    //          i != s->innerGhostZoneFragments.end();
    //          ++i)
    //     TS_ASSERT_EQUALS(i->second.size(), ghostZoneWidth + 1);
    //     for (MyParallelStepper::MPIRegionVecMap::iterator i = 
    //              s->outerGhostZoneFragments.begin();
    //          i != s->outerGhostZoneFragments.end();
    //          ++i)
    //         TS_ASSERT_EQUALS(i->second.size(), ghostZoneWidth + 1);
    }

private:
    // boost::shared_ptr<MyParallelStepper> s;
    // boost::shared_ptr<Initializer<TestCell> > init;
    // unsigned width;
    // unsigned height;
    // unsigned maxSteps;
    // unsigned firstStep;
    // unsigned firstNanoStep;
    // unsigned firstCycle;
    // unsigned outputPeriod;
    // unsigned loadBalancingPeriod;
    // unsigned ghostZoneWidth;
    // unsigned offset;
    // boost::shared_ptr<PartitionManager> partitionManager;
};

}
}
