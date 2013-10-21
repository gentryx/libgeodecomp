#include <libgeodecomp.h>
#include <libgeodecomp/io/mocksteerer.h>
#include <libgeodecomp/io/mockwriter.h>
#include <libgeodecomp/io/teststeerer.h>
#include <libgeodecomp/io/parallelmemorywriter.h>
#include <libgeodecomp/io/testinitializer.h>
#include <libgeodecomp/geometry/partitions/stripingpartition.h>
#include <libgeodecomp/misc/testcell.h>
#include <libgeodecomp/misc/testhelper.h>
#include <libgeodecomp/parallelization/hiparsimulator.h>

#include <boost/assign/std/vector.hpp>
#include <boost/shared_ptr.hpp>
#include <cxxtest/TestSuite.h>

using namespace boost::assign;
using namespace LibGeoDecomp;
using namespace HiParSimulator;

namespace LibGeoDecomp {
namespace HiParSimulator {

class HiParSimulatorTest : public CxxTest::TestSuite
{
public:
    typedef GridBase<TestCell<2>, 2> GridBaseType;
    typedef HiParSimulator<TestCell<2>, StripingPartition<2> > SimulatorType;
    typedef ParallelMemoryWriter<TestCell<2> > MemoryWriterType;
    typedef MockSteerer<TestCell<2> > MockSteererType;
    typedef TestSteerer<2 > TestSteererType;

    void setUp()
    {
        int width = 11;
        int height = 21;
        dim = Coord<2>(width, height);
        maxSteps = 200;
        firstStep = 20;
        TestInitializer<TestCell<2> > *init = new TestInitializer<TestCell<2> >(
            dim, maxSteps, firstStep);

        outputPeriod = 1;
        loadBalancingPeriod = 31;
        ghostzZoneWidth = 10;
        s.reset(new SimulatorType(
                    init, 0, loadBalancingPeriod, ghostzZoneWidth));
        mockWriter = new MockWriter();
        memoryWriter = new MemoryWriterType(outputPeriod);
        s->addWriter(mockWriter);
        s->addWriter(memoryWriter);
    }

    void tearDown()
    {
        s.reset();
    }

    void testInitialWeights()
    {
        std::vector<double> rankSpeeds;
        std::vector<size_t> expected;

        rankSpeeds << 1 << 1 << 1 << 1;
        expected << 2 << 3 << 2 << 3;
        TS_ASSERT_EQUALS(s->initialWeights(10, rankSpeeds), expected);
        rankSpeeds.clear();
        expected.clear();

        rankSpeeds << 3 << 2 << 2 << 3;
        expected << 6 << 4 << 4 << 6;
        TS_ASSERT_EQUALS(s->initialWeights(20, rankSpeeds), expected);
        rankSpeeds.clear();
        expected.clear();

        rankSpeeds << 13;
        expected << 100;
        TS_ASSERT_EQUALS(s->initialWeights(100, rankSpeeds), expected);
        rankSpeeds.clear();
        expected.clear();

        rankSpeeds << 2 << 2 << 2 << 2 << 2;
        expected << 2 << 2 << 2 << 2 << 3;
        TS_ASSERT_EQUALS(s->initialWeights(11, rankSpeeds), expected);
        rankSpeeds.clear();
        expected.clear();


        rankSpeeds << 1 << 1 << 1 << 1;
        expected << 4 << 5 << 5 << 5;
        TS_ASSERT_EQUALS(s->initialWeights(19, rankSpeeds), expected);
        rankSpeeds.clear();
        expected.clear();
    }

    void testStep()
    {
        s->step();

        MockWriter::EventVec expectedEvents;
        expectedEvents << MockWriterHelpers::MockWriterEvent(20, WRITER_INITIALIZED,   0, false)
                       << MockWriterHelpers::MockWriterEvent(20, WRITER_INITIALIZED,   0, true )
                       << MockWriterHelpers::MockWriterEvent(21, WRITER_STEP_FINISHED, 0, false)
                       << MockWriterHelpers::MockWriterEvent(21, WRITER_STEP_FINISHED, 0, true );
        TS_ASSERT_EQUALS(expectedEvents, mockWriter->events());

        std::vector<unsigned> actualSteps;
        std::vector<unsigned> expectedSteps;
        expectedSteps += 20, 21;

        MemoryWriterType::GridMap grids = memoryWriter->getGrids();
        for (MemoryWriterType::GridMap::iterator iter = grids.begin(); iter != grids.end(); ++iter) {
            actualSteps << iter->first;
            int globalNanoStep = iter->first * APITraits::SelectNanoSteps<TestCell<2> >::VALUE;
            TS_ASSERT_TEST_GRID(
                MemoryWriterType::GridType, iter->second, globalNanoStep);
        }

        TS_ASSERT_EQUALS(expectedSteps, actualSteps);
    }

    void testRun()
    {
        s->run();

        MockWriter::EventVec expectedEvents;
        expectedEvents << MockWriterHelpers::MockWriterEvent(20, WRITER_INITIALIZED,   0, false)
                       << MockWriterHelpers::MockWriterEvent(20, WRITER_INITIALIZED,   0, true);
        for (int t = 21; t < 200; ++t) {
            expectedEvents << MockWriterHelpers::MockWriterEvent(t, WRITER_STEP_FINISHED, 0, false)
                           << MockWriterHelpers::MockWriterEvent(t, WRITER_STEP_FINISHED, 0, true);
        }
        expectedEvents << MockWriterHelpers::MockWriterEvent(200, WRITER_ALL_DONE, 0, false)
                       << MockWriterHelpers::MockWriterEvent(200, WRITER_ALL_DONE, 0, true);

        TS_ASSERT_EQUALS(expectedEvents, mockWriter->events());

        for (int t = 20; t <= 200; ++t) {
            int globalNanoStep = t * APITraits::SelectNanoSteps<TestCell<2> >::VALUE;
            MemoryWriterType::GridMap grids = memoryWriter->getGrids();
            TS_ASSERT_TEST_GRID(
                MemoryWriterType::GridType,
                grids[t],
                globalNanoStep);
            TS_ASSERT_EQUALS(dim, grids[t].getDimensions());
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
        for (int i = 25; i <= 200; i += 5) {
            expected << "nextStep(" << i << ", STEERER_NEXT_STEP, 0, 0)\n";
            expected << "nextStep(" << i << ", STEERER_NEXT_STEP, 0, 1)\n";
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
        int cycle = 200 * 27 + 4711 * 27;

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
    unsigned outputPeriod;
    unsigned loadBalancingPeriod;
    unsigned ghostzZoneWidth;
    MockWriter *mockWriter;
    MemoryWriterType *memoryWriter;
};

}
}
