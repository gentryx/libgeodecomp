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

#include <boost/shared_ptr.hpp>
#include <cxxtest/TestSuite.h>

using namespace LibGeoDecomp;

namespace LibGeoDecomp {

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
        events.reset(new MockWriter<>::EventsStore);
        mockWriter = new MockWriter<>(events);
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
        std::vector<std::size_t> expected;

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

        MockWriter<>::EventsStore expectedEvents;
        expectedEvents << MockWriter<>::Event(20, WRITER_INITIALIZED,   0, false)
                       << MockWriter<>::Event(20, WRITER_INITIALIZED,   0, true )
                       << MockWriter<>::Event(21, WRITER_STEP_FINISHED, 0, false)
                       << MockWriter<>::Event(21, WRITER_STEP_FINISHED, 0, true );
        TS_ASSERT_EQUALS(expectedEvents, *events);

        std::vector<unsigned> actualSteps;
        std::vector<unsigned> expectedSteps;
        expectedSteps << 20 << 21;

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

        MockWriter<>::EventsStore expectedEvents;
        expectedEvents << MockWriter<>::Event(20, WRITER_INITIALIZED,   0, false)
                       << MockWriter<>::Event(20, WRITER_INITIALIZED,   0, true);
        for (int t = 21; t < 200; ++t) {
            expectedEvents << MockWriter<>::Event(t, WRITER_STEP_FINISHED, 0, false)
                           << MockWriter<>::Event(t, WRITER_STEP_FINISHED, 0, true);
        }
        expectedEvents << MockWriter<>::Event(200, WRITER_ALL_DONE, 0, false)
                       << MockWriter<>::Event(200, WRITER_ALL_DONE, 0, true);

        TS_ASSERT_EQUALS(expectedEvents, *events);

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
        SharedPtr<MockSteererType::EventsStore>::Type events(new MockSteererType::EventsStore);
        s->addSteerer(new MockSteererType(5, events));
        s->run();
        s.reset();

        MockSteererType::EventsStore expected;
        typedef MockSteererType::Event Event;
        expected << Event(20, STEERER_INITIALIZED, 0, false)
                 << Event(20, STEERER_INITIALIZED, 0, true);
        for (int i = 25; i < 200; i += 5) {
            expected << Event(i, STEERER_NEXT_STEP, 0, false)
                     << Event(i, STEERER_NEXT_STEP, 0, true);
        }
        expected << Event(200, STEERER_ALL_DONE,  0, false)
                 << Event(200, STEERER_ALL_DONE,  0, true)
                 << Event(-1,  STEERER_ALL_DONE, -1, true);

        TS_ASSERT_EQUALS(*events, expected);
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
    SharedPtr<SimulatorType>::Type s;
    Coord<2> dim;
    unsigned maxSteps;
    unsigned firstStep;
    unsigned outputPeriod;
    unsigned loadBalancingPeriod;
    unsigned ghostzZoneWidth;
    MockWriter<> *mockWriter;
    SharedPtr<MockWriter<>::EventsStore>::Type events;
    MemoryWriterType *memoryWriter;
};

}
