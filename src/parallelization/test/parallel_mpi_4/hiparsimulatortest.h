#include <boost/shared_ptr.hpp>
#include <cxxtest/TestSuite.h>
#include "../../../io/mockwriter.h"
#include "../../../io/testinitializer.h"
#include "../../../misc/testcell.h"
#include "../../../misc/testhelper.h"
#include "../../hiparsimulator.h"
#include "../../hiparsimulator/partitions/stripingpartition.h"

using namespace LibGeoDecomp; 
using namespace HiParSimulator; 

namespace LibGeoDecomp {
namespace HiParSimulator {

class HiParSimulatorTest : public CxxTest::TestSuite
{
public:

    void setUp()
    {
        width = 131;
        height = 241;
        maxSteps = 1500;
        firstStep = 20;
        firstCycle = firstStep * TestCell<2>::nanoSteps();
        TestInitializer<2> *init = new TestInitializer<2>(
            Coord<2>(width, height), maxSteps, firstStep);
        
        outputPeriod = 17;
        loadBalancingPeriod = 31;
        ghostZoneWidth = 10;
        s.reset(new HiParSimulator<TestCell<2>, StripingPartition<2> >(
                    init, 0, outputPeriod, 
                    loadBalancingPeriod, ghostZoneWidth));
        // mockWriter = new MockWriter(&(*s));
    }

    void tearDown()
    {
        s.reset();        
    }

    void testRegionConsistency()
    {
        TS_ASSERT_EQUALS(s->partitionManager.ownRegion(), 
                         s->partitionManager.innerSet(ghostZoneWidth) + 
                         s->partitionManager.rim(ghostZoneWidth));
    }

    void testNanoStepSimple()
    {
        s->nanoStep(1);
        s->regionStepper.waitForGhostZones();

        TS_ASSERT_TEST_GRID_REGION(
            DisplacedGrid<TestCell<2> >, 
            *s->getDisplacedGrid(), 
            s->partitionManager.ownRegion(), 
            firstCycle + 1);
        TS_ASSERT_TEST_GRID_REGION(
            DisplacedGrid<TestCell<2> >, 
            *s->getDisplacedGrid(), 
            s->partitionManager.rim(0),      
            firstCycle + 1);
        TS_ASSERT_EQUALS(s->regionStepper.validGhostZoneWidth, ghostZoneWidth);

        s->nanoStep(1);
        s->regionStepper.waitForGhostZones();
        TS_ASSERT_TEST_GRID_REGION(
            DisplacedGrid<TestCell<2> >, 
            *s->getDisplacedGrid(), 
            s->partitionManager.ownRegion(), 
            firstCycle + 2);
        TS_ASSERT_TEST_GRID_REGION(
            DisplacedGrid<TestCell<2> >, 
            *s->getDisplacedGrid(), 
            s->partitionManager.rim(0),      
            firstCycle + 2);
        TS_ASSERT_EQUALS(s->regionStepper.validGhostZoneWidth, ghostZoneWidth);

        s->nanoStep(1);
        s->regionStepper.waitForGhostZones();
        TS_ASSERT_TEST_GRID_REGION(
            DisplacedGrid<TestCell<2> >, 
            *s->getDisplacedGrid(), 
            s->partitionManager.ownRegion(), 
            firstCycle + 3);
        TS_ASSERT_TEST_GRID_REGION(
            DisplacedGrid<TestCell<2> >, 
            *s->getDisplacedGrid(), 
            s->partitionManager.rim(0),      
            firstCycle + 3);
        TS_ASSERT_EQUALS(s->regionStepper.validGhostZoneWidth, ghostZoneWidth);
    }

    void testNanoStepStillSimple()
    {
        s->nanoStep(2);
        s->regionStepper.waitForGhostZones();
        TS_ASSERT_TEST_GRID_REGION(
            DisplacedGrid<TestCell<2> >, 
            *s->getDisplacedGrid(), 
            s->partitionManager.ownRegion(), 
            firstCycle + 2);
        TS_ASSERT_TEST_GRID_REGION(
            DisplacedGrid<TestCell<2> >, 
            *s->getDisplacedGrid(), 
            s->partitionManager.rim(0), 
            firstCycle + 2);
        TS_ASSERT_EQUALS(s->regionStepper.validGhostZoneWidth, ghostZoneWidth);

        s->nanoStep(3);
        s->regionStepper.waitForGhostZones();
        TS_ASSERT_TEST_GRID_REGION(
            DisplacedGrid<TestCell<2> >, 
            *s->getDisplacedGrid(), 
            s->partitionManager.ownRegion(), 
            firstCycle + 5);
        TS_ASSERT_TEST_GRID_REGION(
            DisplacedGrid<TestCell<2> >, 
            *s->getDisplacedGrid(), 
            s->partitionManager.rim(0), 
            firstCycle + 5);
        TS_ASSERT_EQUALS(s->regionStepper.validGhostZoneWidth, ghostZoneWidth);

        s->nanoStep(7);
        s->regionStepper.waitForGhostZones();
        TS_ASSERT_TEST_GRID_REGION(
            DisplacedGrid<TestCell<2> >, 
            *s->getDisplacedGrid(),
            s->partitionManager.ownRegion(),
            firstCycle + 12);
        TS_ASSERT_TEST_GRID_REGION(
            DisplacedGrid<TestCell<2> >,
            *s->getDisplacedGrid(),
            s->partitionManager.rim(0),
            firstCycle + 12);
        TS_ASSERT_EQUALS(s->regionStepper.validGhostZoneWidth, ghostZoneWidth);
    }

    void testNanoStepWithOneLoopIteration()
    {
        s->nanoStep(18);
        s->regionStepper.waitForGhostZones();
        TS_ASSERT_TEST_GRID_REGION(
            DisplacedGrid<TestCell<2> >, 
            *s->getDisplacedGrid(), 
            s->partitionManager.ownRegion(), 
            firstCycle + 18);
        TS_ASSERT_TEST_GRID_REGION(
            DisplacedGrid<TestCell<2> >, 
            *s->getDisplacedGrid(), 
            s->partitionManager.rim(0),
            firstCycle + 18);
        TS_ASSERT_EQUALS(s->regionStepper.validGhostZoneWidth, ghostZoneWidth);
    }

    void testNanoStepWithMultipleLoopIterations()
    {
        s->nanoStep(51);
        s->regionStepper.waitForGhostZones();
        TS_ASSERT_TEST_GRID_REGION(
            DisplacedGrid<TestCell<2> >, 
            *s->getDisplacedGrid(), 
            s->partitionManager.ownRegion(), 
            firstCycle + 51);
        TS_ASSERT_TEST_GRID_REGION(
            DisplacedGrid<TestCell<2> >, 
            *s->getDisplacedGrid(), 
            s->partitionManager.rim(0),
            firstCycle + 51);
        TS_ASSERT_EQUALS(s->regionStepper.validGhostZoneWidth, ghostZoneWidth);

        s->nanoStep(666);
        s->regionStepper.waitForGhostZones();
        TS_ASSERT_TEST_GRID_REGION(
            DisplacedGrid<TestCell<2> >,
            *s->getDisplacedGrid(),
            s->partitionManager.ownRegion(),
            firstCycle + 51 + 666);
        TS_ASSERT_TEST_GRID_REGION(
            DisplacedGrid<TestCell<2> >,
            *s->getDisplacedGrid(),
            s->partitionManager.rim(0),
            firstCycle + 51 + 666);
        TS_ASSERT_EQUALS(s->regionStepper.validGhostZoneWidth, ghostZoneWidth);
    }

    void testAllGatherGroupRegion1()
    {
        SuperVector<Region<2> > parts(4);
        parts[0] << Streak<2>(Coord<2>(0,  0), 10)
                 << Streak<2>(Coord<2>(0,  1), 10)
                 << Streak<2>(Coord<2>(0, 14), 10)
                 << Streak<2>(Coord<2>(0, 15), 10);
        parts[1] << Streak<2>(Coord<2>(0,  2), 10)
                 << Streak<2>(Coord<2>(0,  3), 10)
                 << Streak<2>(Coord<2>(0, 12), 10)
                 << Streak<2>(Coord<2>(0, 13), 10);
        parts[2] << Streak<2>(Coord<2>(0,  4), 10)
                 << Streak<2>(Coord<2>(0,  5), 10)
                 << Streak<2>(Coord<2>(0, 10), 10)
                 << Streak<2>(Coord<2>(0, 11), 10);
        parts[3] << Streak<2>(Coord<2>(0,  6), 10)
                 << Streak<2>(Coord<2>(0,  7), 10)
                 << Streak<2>(Coord<2>(0,  8), 10)
                 << Streak<2>(Coord<2>(0,  9), 10);
        Region<2> expected;
        for (int y = 0; y < 16; ++y)
            expected << Streak<2>(Coord<2>(0, y), 10);
        Region<2> actual = s->allGatherGroupRegion(parts[MPILayer().rank()]);
        TS_ASSERT_EQUALS(expected, actual);
    }

    void testAllGatherGroupRegion2()
    {
        Region<2> actual = s->allGatherGroupRegion();
        Region<2> expected;
        for (int y = 0; y < height; ++y)
            expected << Streak<2>(Coord<2>(0, y), width);
        TS_ASSERT_EQUALS(expected, actual);
    }


private:
    boost::shared_ptr<HiParSimulator<TestCell<2>, StripingPartition<2> > > s;
    unsigned width;
    unsigned height;
    unsigned maxSteps;
    unsigned firstStep;
    unsigned firstCycle;
    unsigned outputPeriod;
    unsigned loadBalancingPeriod;
    unsigned ghostZoneWidth;
    MockWriter *mockWriter;
};

};
};
