#include <boost/shared_ptr.hpp>
#include <cxxtest/TestSuite.h>
#include <sstream>
#include <libgeodecomp/io/mocksteerer.h>
#include <libgeodecomp/io/mockwriter.h>
#include <libgeodecomp/io/teststeerer.h>
#include <libgeodecomp/io/parallelmemorywriter.h>
#include <libgeodecomp/io/testinitializer.h>
#include <libgeodecomp/loadbalancer/mockbalancer.h>
#include <libgeodecomp/misc/testcell.h>
#include <libgeodecomp/misc/testhelper.h>
#include <libgeodecomp/parallelization/hiparsimulator.h>
#include <libgeodecomp/parallelization/hiparsimulator/partitions/zcurvepartition.h>

using namespace LibGeoDecomp; 
using namespace HiParSimulator; 

namespace LibGeoDecomp {
namespace HiParSimulator {

class HiParSimulatorTest : public CxxTest::TestSuite
{
public:
    typedef GridBase<TestCell<2>, 2> GridBaseType;
    typedef HiParSimulator<TestCell<2>, ZCurvePartition<2> > SimulatorType;
    typedef ParallelMemoryWriter<TestCell<2> > MemoryWriterType;
    typedef MockSteerer<TestCell<2> > MockSteererType;
    typedef TestSteerer<2 > TestSteererType;

    void setUp()
    {
        int width = 131;
        int height = 241;
        dim = Coord<2>(width, height);
        maxSteps = 100;
        firstStep = 20;
        firstCycle = firstStep * TestCell<2>::nanoSteps();
        TestInitializer<TestCell<2> > *init = new TestInitializer<TestCell<2> >(
            dim, maxSteps, firstStep);

        outputPeriod = 4;
        loadBalancingPeriod = 31;
        ghostZoneWidth = 10;
        s.reset(new SimulatorType(
                    init, 
                    new MockBalancer(), 
                    loadBalancingPeriod, 
                    ghostZoneWidth));
        mockWriter = new MockWriter(&*s);
        memoryWriter = new MemoryWriterType(&*s, outputPeriod);
    }

    void tearDown()
    {
        s.reset();
    }

    void testStep()
    {
        s->step();
        TS_ASSERT_EQUALS((31 - 1)       * 27, s->timeToNextEvent());
        TS_ASSERT_EQUALS((100 - 20 - 1) * 27, s->timeToLastEvent());
        s->step();
        s->step();
        s->step();

        TS_ASSERT_EQUALS((31 - 4)       * 27, s->timeToNextEvent());
        TS_ASSERT_EQUALS((100 - 20 - 4) * 27, s->timeToLastEvent());

        std::stringstream expectedEvents;
        expectedEvents << "initialized()\ninitialized()\n";
        for (int t = 21; t < 25; t += 1) {
            expectedEvents << "stepFinished(step=" << t << ")\n"
                           << "stepFinished(step=" << t << ")\n";
        }
        TS_ASSERT_EQUALS(expectedEvents.str(), mockWriter->events());

        for (int t = 20; t < 25; t += outputPeriod) {
            int globalNanoStep = t * TestCell<2>::nanoSteps();
            MemoryWriterType::GridMap grids = memoryWriter->getGrids();
            TS_ASSERT_TEST_GRID(
                MemoryWriterType::GridType, 
                grids[t], 
                globalNanoStep);
            TS_ASSERT_EQUALS(dim, grids[t].getDimensions());
        }
    }

    void testRun()
    {
        s->run();

        for (int t = firstStep; t < maxSteps; t += outputPeriod) {
            int globalNanoStep = t * TestCell<2>::nanoSteps();
            MemoryWriterType::GridMap grids = memoryWriter->getGrids();
            TS_ASSERT_TEST_GRID(
                MemoryWriterType::GridType, 
                grids[t], 
                globalNanoStep);
            TS_ASSERT_EQUALS(dim, grids[t].getDimensions());
        }

        if (MPILayer().rank() == 0) {
            std::string expectedEvents;
            for (int i = 0; i < 2; ++i) {
                expectedEvents += "balance() [7892, 7893, 7893, 7893] [1, 1, 1, 1]\n";
            }

            TS_ASSERT_EQUALS(expectedEvents, MockBalancer::events);
        }
    }

    void testSteererCallback()
    {
        std::stringstream events;
        s->addSteerer(new MockSteererType(5, &events));
        s->run();
        s.reset();        

        std::stringstream expected;
        expected << "created, period = 5\n";
        for (int i = 25; i <= 100; i += 5) {
            expected << "nextStep(" << i << ")\n";
            expected << "nextStep(" << i << ")\n";
        }
        expected << "deleted\n";

        TS_ASSERT_EQUALS(events.str(), expected.str());
    }

    void testSteererFunctionality()
    {
        s->addSteerer(new TestSteererType(5, 25, 4711 * 27));
        s->run();

        const Region<2> *region = &s->updateGroup->partitionManager->ownRegion();
        const GridBaseType *grid = &s->updateGroup->grid();
        int cycle = 100 * 27 + 4711 * 27;

        TS_ASSERT_TEST_GRID_REGION(
            GridBaseType, 
            *grid, 
            *region, 
            cycle);
    }

private:
    boost::shared_ptr<SimulatorType> s;
    Coord<2> dim;
    unsigned maxSteps;
    unsigned firstStep;
    unsigned firstCycle;
    unsigned outputPeriod;
    unsigned loadBalancingPeriod;
    unsigned ghostZoneWidth;
    MockWriter *mockWriter;
    MemoryWriterType *memoryWriter;
};

}
}
