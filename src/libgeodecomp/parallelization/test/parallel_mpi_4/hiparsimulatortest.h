#include <libgeodecomp.h>
#include <libgeodecomp/geometry/partitions/zcurvepartition.h>
#include <libgeodecomp/io/mocksteerer.h>
#include <libgeodecomp/io/mockwriter.h>
#include <libgeodecomp/io/teststeerer.h>
#include <libgeodecomp/io/testwriter.h>
#include <libgeodecomp/io/parallelmemorywriter.h>
#include <libgeodecomp/io/paralleltestwriter.h>
#include <libgeodecomp/io/testinitializer.h>
#include <libgeodecomp/io/unstructuredtestinitializer.h>
#include <libgeodecomp/loadbalancer/mockbalancer.h>
#include <libgeodecomp/misc/nonpodtestcell.h>
#include <libgeodecomp/misc/testcell.h>
#include <libgeodecomp/misc/testhelper.h>
#include <libgeodecomp/parallelization/hiparsimulator.h>

#include <cxxtest/TestSuite.h>
#include <sstream>

using namespace LibGeoDecomp;

namespace LibGeoDecomp {

// fixme: use this writer in simulator verification tests in simulation factory
class AccumulatingWriter : public Clonable<ParallelWriter<TestCell<2> >, AccumulatingWriter>
{
public:
    using ParallelWriter<TestCell<2> >::GridType;
    using ParallelWriter<TestCell<2> >::RegionType;
    using ParallelWriter<TestCell<2> >::CoordType;
    using ParallelWriter<TestCell<2> >::region;

    AccumulatingWriter() :
        Clonable<ParallelWriter<TestCell<2> >, AccumulatingWriter>("", 1),
        cellsSeen(0)
    {}

    virtual void setRegion(const Region<Topology::DIM>& newRegion)
    {
        Clonable<ParallelWriter<TestCell<2> >, AccumulatingWriter>::setRegion(newRegion);
        domainSize = newRegion.size();
    }

    virtual void stepFinished(
        const GridType& grid,
        const RegionType& validRegion,
        const CoordType& globalDimensions,
        unsigned step,
        WriterEvent event,
        std::size_t rank,
        bool lastCall)
    {
        cellsSeen += validRegion.size();

        if (lastCall) {
            TS_ASSERT_EQUALS(cellsSeen, domainSize);
            cellsSeen = 0;
        }
    }

private:
    std::size_t domainSize;
    std::size_t cellsSeen;
};

class HiParSimulatorTest : public CxxTest::TestSuite
{
public:
    typedef GridBase<TestCell<2>, 2> GridBaseType;
    typedef HiParSimulator<TestCell<2>, ZCurvePartition<2> > SimulatorType;
    typedef ParallelMemoryWriter<TestCell<2> > MemoryWriterType;
    typedef MockSteerer<TestCell<2> > MockSteererType;
    typedef TestSteerer<2> TestSteererType;

    static const unsigned NANO_STEPS = APITraits::SelectNanoSteps<TestCell<2> >::VALUE;

    void setUp()
    {
        int width = 51;
        int height = 111;
        dim = Coord<2>(width, height);
        maxSteps = 101;
        firstStep = 20;
        firstCycle = firstStep * NANO_STEPS;
        TestInitializer<TestCell<2> > *init = new TestInitializer<TestCell<2> >(
            dim, maxSteps, firstStep);

        outputPeriod = 4;
        loadBalancingPeriod = 31;
        ghostZoneWidth = 10;
        sim.reset(new SimulatorType(
                    init,
                    new MockBalancer(),
                    loadBalancingPeriod,
                    ghostZoneWidth));
        events.reset(new MockWriter<>::EventsStore);
        mockWriter = new MockWriter<>(events);
        memoryWriter = new MemoryWriterType(outputPeriod);
        sim->addWriter(mockWriter);
        sim->addWriter(memoryWriter);
        rank = MPILayer().rank();
    }

    void tearDown()
    {
        sim.reset();
    }

    void testStep()
    {
        sim->step();
        TS_ASSERT_EQUALS((31 - 1)       * 27, sim->timeToNextEvent());
        TS_ASSERT_EQUALS((101 - 20 - 1) * 27, sim->timeToLastEvent());
        sim->step();
        sim->step();
        sim->step();

        TS_ASSERT_EQUALS((31 - 4)       * 27, sim->timeToNextEvent());
        TS_ASSERT_EQUALS((101 - 20 - 4) * 27, sim->timeToLastEvent());

        MockWriter<>::EventsStore expectedEvents;
        expectedEvents << MockWriter<>::Event(20, WRITER_INITIALIZED, rank, false)
                       << MockWriter<>::Event(20, WRITER_INITIALIZED, rank, true);
        for (int t = 21; t < 25; t += 1) {
            expectedEvents << MockWriter<>::Event(t, WRITER_STEP_FINISHED, rank, false)
                           << MockWriter<>::Event(t, WRITER_STEP_FINISHED, rank, true);
        }
        TS_ASSERT_EQUALS(expectedEvents, *events);

        for (int t = 20; t < 25; t += outputPeriod) {
            int globalNanoStep = t * NANO_STEPS;
            MemoryWriterType::GridMap& grids = memoryWriter->getGrids();
            TS_ASSERT_TEST_GRID(
                MemoryWriterType::GridType,
                grids[t],
                globalNanoStep);
            TS_ASSERT_EQUALS(dim, grids[t].getDimensions());
        }
    }

    void testRun()
    {
        sim->run();

        for (unsigned t = firstStep; t < maxSteps; t += outputPeriod) {
            unsigned globalNanoStep = t * NANO_STEPS;
            MemoryWriterType::GridMap& grids = memoryWriter->getGrids();
            TS_ASSERT_TEST_GRID(
                MemoryWriterType::GridType,
                grids[t],
                globalNanoStep);
            TS_ASSERT_EQUALS(dim, grids[t].getDimensions());
        }

        // check last step, too
        unsigned t = maxSteps;
        unsigned globalNanoStep = t * NANO_STEPS;
        MemoryWriterType::GridMap grids = memoryWriter->getGrids();
        TS_ASSERT_TEST_GRID(
            MemoryWriterType::GridType,
            grids[t],
            globalNanoStep);
        TS_ASSERT_EQUALS(dim, grids[t].getDimensions());

        if (rank == 0) {
            std::string expectedEvents;
            for (int i = 0; i < 2; ++i) {
                expectedEvents += "balance() [1415, 1415, 1415, 1416] [1, 1, 1, 1]\n";
            }

            TS_ASSERT_EQUALS(expectedEvents, MockBalancer::events);
        }
    }

    void testSteererCallback()
    {
        SharedPtr<MockSteererType::EventsStore>::Type events(new MockSteererType::EventsStore);
        sim->addSteerer(new MockSteererType(5, events));
        sim->run();
        sim.reset();

        MockSteererType::EventsStore expected;
        typedef MockSteererType::Event Event;
        expected << Event(20, STEERER_INITIALIZED, rank, false)
                 << Event(20, STEERER_INITIALIZED, rank, true);
        for (unsigned i = 25; i < maxSteps; i += 5) {
            expected << Event(i, STEERER_NEXT_STEP, rank, false)
                     << Event(i, STEERER_NEXT_STEP, rank, true);
        }

        unsigned deletionCode = Limits<unsigned>::getMax();
        expected << Event(101,           STEERER_ALL_DONE, rank,         false)
                 << Event(101,           STEERER_ALL_DONE, rank,         true)
                 << Event(deletionCode,  STEERER_ALL_DONE, deletionCode, true);

        TS_ASSERT_EQUALS(*events, expected);
    }

    void testSteererFunctionalityBasic()
    {
        sim->addSteerer(new TestSteererType(5, 25, 4711 * 27));
        sim->run();

        const Region<2> *region = &sim->updateGroup->partitionManager->innerSet(ghostZoneWidth);
        const GridBaseType *grid = &sim->updateGroup->grid();
        int cycle = 101 * 27 + 4711 * 27;

        TS_ASSERT_TEST_GRID_REGION(
            GridBaseType,
            *grid,
            *region,
            cycle);
    }

    void testSteererFunctionality2DWithGhostZoneWidth1()
    {
        typedef HiParSimulator<TestCell<2>, ZCurvePartition<2> > SimulatorType;
        int maxTimeSteps = 90;
        Coord<2> dim(20, 25);

        TestInitializer<TestCell<2> > *init = new TestInitializer<TestCell<2> >(dim, maxTimeSteps);
        int loadBalancingPeriod = 10;
        int ghostZoneWidth = 1;
        SimulatorType sim(
            init,
            new MockBalancer(),
            loadBalancingPeriod,
            ghostZoneWidth);

        sim.addSteerer(new TestSteerer<2>(5, 25, 4711 * 27));
        sim.run();
    }

    void testSteererFunctionality2DWithGhostZoneWidth2()
    {
        typedef HiParSimulator<TestCell<2>, ZCurvePartition<2> > SimulatorType;
        int maxTimeSteps = 90;
        Coord<2> dim(20, 25);

        TestInitializer<TestCell<2> > *init = new TestInitializer<TestCell<2> >(dim, maxTimeSteps);
        int loadBalancingPeriod = 10;
        int ghostZoneWidth = 2;
        SimulatorType sim(
            init,
            new MockBalancer(),
            loadBalancingPeriod,
            ghostZoneWidth);

        sim.addSteerer(new TestSteerer<2>(5, 25, 4711 * 27));
        sim.run();
    }

    void testSteererFunctionality2DWithGhostZoneWidth3()
    {
        typedef HiParSimulator<TestCell<2>, ZCurvePartition<2> > SimulatorType;
        int maxTimeSteps = 90;
        Coord<2> dim(20, 25);

        TestInitializer<TestCell<2> > *init = new TestInitializer<TestCell<2> >(dim, maxTimeSteps);
        int loadBalancingPeriod = 10;
        int ghostZoneWidth = 3;
        SimulatorType sim(
            init,
            new MockBalancer(),
            loadBalancingPeriod,
            ghostZoneWidth);

        sim.addSteerer(new TestSteerer<2>(5, 25, 4711 * 27));
        sim.run();
    }

    void testSteererFunctionality2DWithGhostZoneWidth4()
    {
        typedef HiParSimulator<TestCell<2>, ZCurvePartition<2> > SimulatorType;
        int maxTimeSteps = 90;
        Coord<2> dim(20, 25);

        TestInitializer<TestCell<2> > *init = new TestInitializer<TestCell<2> >(dim, maxTimeSteps);
        int loadBalancingPeriod = 10;
        int ghostZoneWidth = 4;
        SimulatorType sim(
            init,
            new MockBalancer(),
            loadBalancingPeriod,
            ghostZoneWidth);

        sim.addSteerer(new TestSteerer<2>(5, 25, 4711 * 27));
        sim.run();
    }

    void testSteererFunctionality2DWithGhostZoneWidth5()
    {
        typedef HiParSimulator<TestCell<2>, ZCurvePartition<2> > SimulatorType;
        int maxTimeSteps = 90;
        Coord<2> dim(20, 25);

        TestInitializer<TestCell<2> > *init = new TestInitializer<TestCell<2> >(dim, maxTimeSteps);
        int loadBalancingPeriod = 10;
        int ghostZoneWidth = 5;
        SimulatorType sim(
            init,
            new MockBalancer(),
            loadBalancingPeriod,
            ghostZoneWidth);

        sim.addSteerer(new TestSteerer<2>(5, 25, 4711 * 27));
        sim.run();
    }

    void testSteererFunctionality2DWithGhostZoneWidth6()
    {
        typedef HiParSimulator<TestCell<2>, ZCurvePartition<2> > SimulatorType;
        int maxTimeSteps = 90;
        Coord<2> dim(20, 25);

        TestInitializer<TestCell<2> > *init = new TestInitializer<TestCell<2> >(dim, maxTimeSteps);
        int loadBalancingPeriod = 10;
        int ghostZoneWidth = 6;
        SimulatorType sim(
            init,
            new MockBalancer(),
            loadBalancingPeriod,
            ghostZoneWidth);

        sim.addSteerer(new TestSteerer<2>(5, 25, 4711 * 27));
        sim.run();
    }

    void testSteererFunctionality3DWithGhostZoneWidth1()
    {
        typedef HiParSimulator<TestCell<3>, ZCurvePartition<3> > SimulatorType;
        int maxTimeSteps = 90;
        Coord<3> dim(20, 25, 30);

        TestInitializer<TestCell<3> > *init = new TestInitializer<TestCell<3> >(dim, maxTimeSteps);
        int loadBalancingPeriod = 10;
        int ghostZoneWidth = 1;
        SimulatorType sim(
            init,
            new MockBalancer(),
            loadBalancingPeriod,
            ghostZoneWidth);

        sim.addSteerer(new TestSteerer<3>(5, 25, 4711 * 27));
        sim.run();
    }

    void testSteererFunctionality3DWithGhostZoneWidth2()
    {
        typedef HiParSimulator<TestCell<3>, ZCurvePartition<3> > SimulatorType;
        int maxTimeSteps = 90;
        Coord<3> dim(20, 25, 30);

        TestInitializer<TestCell<3> > *init = new TestInitializer<TestCell<3> >(dim, maxTimeSteps);
        int loadBalancingPeriod = 10;
        int ghostZoneWidth = 2;
        SimulatorType sim(
            init,
            new MockBalancer(),
            loadBalancingPeriod,
            ghostZoneWidth);

        sim.addSteerer(new TestSteerer<3>(5, 25, 4711 * 27));
        sim.run();
    }

    void testSteererFunctionality3DWithGhostZoneWidth3()
    {
        typedef HiParSimulator<TestCell<3>, ZCurvePartition<3> > SimulatorType;
        int maxTimeSteps = 90;
        Coord<3> dim(20, 25, 30);

        TestInitializer<TestCell<3> > *init = new TestInitializer<TestCell<3> >(dim, maxTimeSteps);
        int loadBalancingPeriod = 10;
        int ghostZoneWidth = 3;
        SimulatorType sim(
            init,
            new MockBalancer(),
            loadBalancingPeriod,
            ghostZoneWidth);

        sim.addSteerer(new TestSteerer<3>(5, 25, 4711 * 27));
        sim.run();
    }

    void testSteererFunctionality3DWithGhostZoneWidth4()
    {
        typedef HiParSimulator<TestCell<3>, ZCurvePartition<3> > SimulatorType;
        int maxTimeSteps = 90;
        Coord<3> dim(20, 25, 30);

        TestInitializer<TestCell<3> > *init = new TestInitializer<TestCell<3> >(dim, maxTimeSteps);
        int loadBalancingPeriod = 10;
        int ghostZoneWidth = 4;
        SimulatorType sim(
            init,
            new MockBalancer(),
            loadBalancingPeriod,
            ghostZoneWidth);

        sim.addSteerer(new TestSteerer<3>(5, 25, 4711 * 27));
        sim.run();
    }

    void testSteererFunctionality3DWithGhostZoneWidth5()
    {
        typedef HiParSimulator<TestCell<3>, ZCurvePartition<3> > SimulatorType;
        int maxTimeSteps = 90;
        Coord<3> dim(20, 25, 30);

        TestInitializer<TestCell<3> > *init = new TestInitializer<TestCell<3> >(dim, maxTimeSteps);
        int loadBalancingPeriod = 10;
        int ghostZoneWidth = 5;
        SimulatorType sim(
            init,
            new MockBalancer(),
            loadBalancingPeriod,
            ghostZoneWidth);

        sim.addSteerer(new TestSteerer<3>(5, 25, 4711 * 27));
        sim.run();
    }

    void testSteererFunctionality3DWithGhostZoneWidth6()
    {
        typedef HiParSimulator<TestCell<3>, ZCurvePartition<3> > SimulatorType;
        int maxTimeSteps = 90;
        Coord<3> dim(20, 25, 30);

        TestInitializer<TestCell<3> > *init = new TestInitializer<TestCell<3> >(dim, maxTimeSteps);
        int loadBalancingPeriod = 10;
        int ghostZoneWidth = 6;
        SimulatorType sim(
            init,
            new MockBalancer(),
            loadBalancingPeriod,
            ghostZoneWidth);

        sim.addSteerer(new TestSteerer<3>(5, 25, 4711 * 27));
        sim.run();
    }

    void testWriterFunctionality2DWithGhostZoneWidth1()
    {
        typedef HiParSimulator<TestCell<2>, ZCurvePartition<2> > SimulatorType;
        int maxTimeSteps = 90;
        Coord<2> dim(20, 25);

        TestInitializer<TestCell<2> > *init = new TestInitializer<TestCell<2> >(dim, maxTimeSteps);
        int loadBalancingPeriod = 10;
        int ghostZoneWidth = 1;
        SimulatorType sim(
            init,
            new MockBalancer(),
            loadBalancingPeriod,
            ghostZoneWidth);

        std::vector<unsigned> expectedWriterSteps;
        std::vector<WriterEvent> expectedWriterEvents;

        expectedWriterSteps <<  0
                            <<  5
                            << 10
                            << 15
                            << 20
                            << 25
                            << 30
                            << 35
                            << 40
                            << 45
                            << 50
                            << 55
                            << 60
                            << 65
                            << 70
                            << 75
                            << 80
                            << 85
                            << 90;

        expectedWriterEvents << WRITER_INITIALIZED
                             << WRITER_STEP_FINISHED
                             << WRITER_STEP_FINISHED
                             << WRITER_STEP_FINISHED
                             << WRITER_STEP_FINISHED
                             << WRITER_STEP_FINISHED
                             << WRITER_STEP_FINISHED
                             << WRITER_STEP_FINISHED
                             << WRITER_STEP_FINISHED
                             << WRITER_STEP_FINISHED
                             << WRITER_STEP_FINISHED
                             << WRITER_STEP_FINISHED
                             << WRITER_STEP_FINISHED
                             << WRITER_STEP_FINISHED
                             << WRITER_STEP_FINISHED
                             << WRITER_STEP_FINISHED
                             << WRITER_STEP_FINISHED
                             << WRITER_STEP_FINISHED
                             << WRITER_ALL_DONE;

        sim.addWriter(new ParallelTestWriter<TestCell<2> >(5, expectedWriterSteps, expectedWriterEvents));
        sim.run();
    }

    void testWriterFunctionality2DWithGhostZoneWidth2()
    {
        typedef HiParSimulator<TestCell<2>, ZCurvePartition<2> > SimulatorType;
        int maxTimeSteps = 90;
        Coord<2> dim(20, 25);

        TestInitializer<TestCell<2> > *init = new TestInitializer<TestCell<2> >(dim, maxTimeSteps);
        int loadBalancingPeriod = 10;
        int ghostZoneWidth = 2;
        SimulatorType sim(
            init,
            new MockBalancer(),
            loadBalancingPeriod,
            ghostZoneWidth);

        std::vector<unsigned> expectedWriterSteps;
        std::vector<WriterEvent> expectedWriterEvents;

        expectedWriterSteps <<  0
                            <<  5
                            << 10
                            << 15
                            << 20
                            << 25
                            << 30
                            << 35
                            << 40
                            << 45
                            << 50
                            << 55
                            << 60
                            << 65
                            << 70
                            << 75
                            << 80
                            << 85
                            << 90;

        expectedWriterEvents << WRITER_INITIALIZED
                             << WRITER_STEP_FINISHED
                             << WRITER_STEP_FINISHED
                             << WRITER_STEP_FINISHED
                             << WRITER_STEP_FINISHED
                             << WRITER_STEP_FINISHED
                             << WRITER_STEP_FINISHED
                             << WRITER_STEP_FINISHED
                             << WRITER_STEP_FINISHED
                             << WRITER_STEP_FINISHED
                             << WRITER_STEP_FINISHED
                             << WRITER_STEP_FINISHED
                             << WRITER_STEP_FINISHED
                             << WRITER_STEP_FINISHED
                             << WRITER_STEP_FINISHED
                             << WRITER_STEP_FINISHED
                             << WRITER_STEP_FINISHED
                             << WRITER_STEP_FINISHED
                             << WRITER_ALL_DONE;

        sim.addWriter(new ParallelTestWriter<TestCell<2> >(5, expectedWriterSteps, expectedWriterEvents));
        sim.run();
    }

    void testWriterFunctionality2DWithGhostZoneWidth3()
    {
        typedef HiParSimulator<TestCell<2>, ZCurvePartition<2> > SimulatorType;
        int maxTimeSteps = 90;
        Coord<2> dim(20, 25);

        TestInitializer<TestCell<2> > *init = new TestInitializer<TestCell<2> >(dim, maxTimeSteps);
        int loadBalancingPeriod = 10;
        int ghostZoneWidth = 3;
        SimulatorType sim(
            init,
            new MockBalancer(),
            loadBalancingPeriod,
            ghostZoneWidth);

        std::vector<unsigned> expectedWriterSteps;
        std::vector<WriterEvent> expectedWriterEvents;

        expectedWriterSteps <<  0
                            <<  5
                            << 10
                            << 15
                            << 20
                            << 25
                            << 30
                            << 35
                            << 40
                            << 45
                            << 50
                            << 55
                            << 60
                            << 65
                            << 70
                            << 75
                            << 80
                            << 85
                            << 90;

        expectedWriterEvents << WRITER_INITIALIZED
                             << WRITER_STEP_FINISHED
                             << WRITER_STEP_FINISHED
                             << WRITER_STEP_FINISHED
                             << WRITER_STEP_FINISHED
                             << WRITER_STEP_FINISHED
                             << WRITER_STEP_FINISHED
                             << WRITER_STEP_FINISHED
                             << WRITER_STEP_FINISHED
                             << WRITER_STEP_FINISHED
                             << WRITER_STEP_FINISHED
                             << WRITER_STEP_FINISHED
                             << WRITER_STEP_FINISHED
                             << WRITER_STEP_FINISHED
                             << WRITER_STEP_FINISHED
                             << WRITER_STEP_FINISHED
                             << WRITER_STEP_FINISHED
                             << WRITER_STEP_FINISHED
                             << WRITER_ALL_DONE;

        sim.addWriter(new ParallelTestWriter<TestCell<2> >(5, expectedWriterSteps, expectedWriterEvents));
        sim.run();
    }

    void testWriterFunctionality2DWithGhostZoneWidth4()
    {
        typedef HiParSimulator<TestCell<2>, ZCurvePartition<2> > SimulatorType;
        int maxTimeSteps = 90;
        Coord<2> dim(20, 25);

        TestInitializer<TestCell<2> > *init = new TestInitializer<TestCell<2> >(dim, maxTimeSteps);
        int loadBalancingPeriod = 10;
        int ghostZoneWidth = 4;
        SimulatorType sim(
            init,
            new MockBalancer(),
            loadBalancingPeriod,
            ghostZoneWidth);

        std::vector<unsigned> expectedWriterSteps;
        std::vector<WriterEvent> expectedWriterEvents;

        expectedWriterSteps <<  0
                            <<  5
                            << 10
                            << 15
                            << 20
                            << 25
                            << 30
                            << 35
                            << 40
                            << 45
                            << 50
                            << 55
                            << 60
                            << 65
                            << 70
                            << 75
                            << 80
                            << 85
                            << 90;

        expectedWriterEvents << WRITER_INITIALIZED
                             << WRITER_STEP_FINISHED
                             << WRITER_STEP_FINISHED
                             << WRITER_STEP_FINISHED
                             << WRITER_STEP_FINISHED
                             << WRITER_STEP_FINISHED
                             << WRITER_STEP_FINISHED
                             << WRITER_STEP_FINISHED
                             << WRITER_STEP_FINISHED
                             << WRITER_STEP_FINISHED
                             << WRITER_STEP_FINISHED
                             << WRITER_STEP_FINISHED
                             << WRITER_STEP_FINISHED
                             << WRITER_STEP_FINISHED
                             << WRITER_STEP_FINISHED
                             << WRITER_STEP_FINISHED
                             << WRITER_STEP_FINISHED
                             << WRITER_STEP_FINISHED
                             << WRITER_ALL_DONE;

        sim.addWriter(new ParallelTestWriter<TestCell<2> >(5, expectedWriterSteps, expectedWriterEvents));
        sim.run();
    }

    void testWriterFunctionality2DWithGhostZoneWidth5()
    {
        typedef HiParSimulator<TestCell<2>, ZCurvePartition<2> > SimulatorType;
        int maxTimeSteps = 90;
        Coord<2> dim(20, 25);

        TestInitializer<TestCell<2> > *init = new TestInitializer<TestCell<2> >(dim, maxTimeSteps);
        int loadBalancingPeriod = 10;
        int ghostZoneWidth = 5;
        SimulatorType sim(
            init,
            new MockBalancer(),
            loadBalancingPeriod,
            ghostZoneWidth);

        std::vector<unsigned> expectedWriterSteps;
        std::vector<WriterEvent> expectedWriterEvents;

        expectedWriterSteps <<  0
                            <<  5
                            << 10
                            << 15
                            << 20
                            << 25
                            << 30
                            << 35
                            << 40
                            << 45
                            << 50
                            << 55
                            << 60
                            << 65
                            << 70
                            << 75
                            << 80
                            << 85
                            << 90;

        expectedWriterEvents << WRITER_INITIALIZED
                             << WRITER_STEP_FINISHED
                             << WRITER_STEP_FINISHED
                             << WRITER_STEP_FINISHED
                             << WRITER_STEP_FINISHED
                             << WRITER_STEP_FINISHED
                             << WRITER_STEP_FINISHED
                             << WRITER_STEP_FINISHED
                             << WRITER_STEP_FINISHED
                             << WRITER_STEP_FINISHED
                             << WRITER_STEP_FINISHED
                             << WRITER_STEP_FINISHED
                             << WRITER_STEP_FINISHED
                             << WRITER_STEP_FINISHED
                             << WRITER_STEP_FINISHED
                             << WRITER_STEP_FINISHED
                             << WRITER_STEP_FINISHED
                             << WRITER_STEP_FINISHED
                             << WRITER_ALL_DONE;

        sim.addWriter(new ParallelTestWriter<TestCell<2> >(5, expectedWriterSteps, expectedWriterEvents));
        sim.run();
    }

    void testWriterFunctionality2DWithGhostZoneWidth6()
    {
        typedef HiParSimulator<TestCell<2>, ZCurvePartition<2> > SimulatorType;
        int maxTimeSteps = 90;
        Coord<2> dim(20, 25);

        TestInitializer<TestCell<2> > *init = new TestInitializer<TestCell<2> >(dim, maxTimeSteps);
        int loadBalancingPeriod = 10;
        int ghostZoneWidth = 6;
        SimulatorType sim(
            init,
            new MockBalancer(),
            loadBalancingPeriod,
            ghostZoneWidth);

        std::vector<unsigned> expectedWriterSteps;
        std::vector<WriterEvent> expectedWriterEvents;

        expectedWriterSteps <<  0
                            <<  5
                            << 10
                            << 15
                            << 20
                            << 25
                            << 30
                            << 35
                            << 40
                            << 45
                            << 50
                            << 55
                            << 60
                            << 65
                            << 70
                            << 75
                            << 80
                            << 85
                            << 90;

        expectedWriterEvents << WRITER_INITIALIZED
                             << WRITER_STEP_FINISHED
                             << WRITER_STEP_FINISHED
                             << WRITER_STEP_FINISHED
                             << WRITER_STEP_FINISHED
                             << WRITER_STEP_FINISHED
                             << WRITER_STEP_FINISHED
                             << WRITER_STEP_FINISHED
                             << WRITER_STEP_FINISHED
                             << WRITER_STEP_FINISHED
                             << WRITER_STEP_FINISHED
                             << WRITER_STEP_FINISHED
                             << WRITER_STEP_FINISHED
                             << WRITER_STEP_FINISHED
                             << WRITER_STEP_FINISHED
                             << WRITER_STEP_FINISHED
                             << WRITER_STEP_FINISHED
                             << WRITER_STEP_FINISHED
                             << WRITER_ALL_DONE;

        sim.addWriter(new ParallelTestWriter<TestCell<2> >(5, expectedWriterSteps, expectedWriterEvents));
        sim.run();
    }

    void testWriterFunctionality3DWithGhostZoneWidth1()
    {
        typedef HiParSimulator<TestCell<3>, ZCurvePartition<3> > SimulatorType;
        int maxTimeSteps = 90;
        Coord<3> dim(20, 25, 30);

        TestInitializer<TestCell<3> > *init = new TestInitializer<TestCell<3> >(dim, maxTimeSteps);
        int loadBalancingPeriod = 10;
        int ghostZoneWidth = 1;
        SimulatorType sim(
            init,
            new MockBalancer(),
            loadBalancingPeriod,
            ghostZoneWidth);

        std::vector<unsigned> expectedWriterSteps;
        std::vector<WriterEvent> expectedWriterEvents;

        expectedWriterSteps <<  0
                            <<  5
                            << 10
                            << 15
                            << 20
                            << 25
                            << 30
                            << 35
                            << 40
                            << 45
                            << 50
                            << 55
                            << 60
                            << 65
                            << 70
                            << 75
                            << 80
                            << 85
                            << 90;

        expectedWriterEvents << WRITER_INITIALIZED
                             << WRITER_STEP_FINISHED
                             << WRITER_STEP_FINISHED
                             << WRITER_STEP_FINISHED
                             << WRITER_STEP_FINISHED
                             << WRITER_STEP_FINISHED
                             << WRITER_STEP_FINISHED
                             << WRITER_STEP_FINISHED
                             << WRITER_STEP_FINISHED
                             << WRITER_STEP_FINISHED
                             << WRITER_STEP_FINISHED
                             << WRITER_STEP_FINISHED
                             << WRITER_STEP_FINISHED
                             << WRITER_STEP_FINISHED
                             << WRITER_STEP_FINISHED
                             << WRITER_STEP_FINISHED
                             << WRITER_STEP_FINISHED
                             << WRITER_STEP_FINISHED
                             << WRITER_ALL_DONE;

        sim.addWriter(new ParallelTestWriter<TestCell<3> >(5, expectedWriterSteps, expectedWriterEvents));
        sim.run();
    }

    void testWriterFunctionality3DWithGhostZoneWidth2()
    {
        typedef HiParSimulator<TestCell<3>, ZCurvePartition<3> > SimulatorType;
        int maxTimeSteps = 90;
        Coord<3> dim(20, 25, 30);

        TestInitializer<TestCell<3> > *init = new TestInitializer<TestCell<3> >(dim, maxTimeSteps);
        int loadBalancingPeriod = 10;
        int ghostZoneWidth = 2;
        SimulatorType sim(
            init,
            new MockBalancer(),
            loadBalancingPeriod,
            ghostZoneWidth);

        std::vector<unsigned> expectedWriterSteps;
        std::vector<WriterEvent> expectedWriterEvents;

        expectedWriterSteps <<  0
                            <<  5
                            << 10
                            << 15
                            << 20
                            << 25
                            << 30
                            << 35
                            << 40
                            << 45
                            << 50
                            << 55
                            << 60
                            << 65
                            << 70
                            << 75
                            << 80
                            << 85
                            << 90;

        expectedWriterEvents << WRITER_INITIALIZED
                             << WRITER_STEP_FINISHED
                             << WRITER_STEP_FINISHED
                             << WRITER_STEP_FINISHED
                             << WRITER_STEP_FINISHED
                             << WRITER_STEP_FINISHED
                             << WRITER_STEP_FINISHED
                             << WRITER_STEP_FINISHED
                             << WRITER_STEP_FINISHED
                             << WRITER_STEP_FINISHED
                             << WRITER_STEP_FINISHED
                             << WRITER_STEP_FINISHED
                             << WRITER_STEP_FINISHED
                             << WRITER_STEP_FINISHED
                             << WRITER_STEP_FINISHED
                             << WRITER_STEP_FINISHED
                             << WRITER_STEP_FINISHED
                             << WRITER_STEP_FINISHED
                             << WRITER_ALL_DONE;

        sim.addWriter(new ParallelTestWriter<TestCell<3> >(5, expectedWriterSteps, expectedWriterEvents));
        sim.run();
    }

    void testWriterFunctionality3DWithGhostZoneWidth3()
    {
        typedef HiParSimulator<TestCell<3>, ZCurvePartition<3> > SimulatorType;
        int maxTimeSteps = 120;
        Coord<3> dim(20, 25, 30);

        TestInitializer<TestCell<3> > *init = new TestInitializer<TestCell<3> >(dim, maxTimeSteps);
        int loadBalancingPeriod = 10;
        int ghostZoneWidth = 3;
        SimulatorType sim(
            init,
            new MockBalancer(),
            loadBalancingPeriod,
            ghostZoneWidth);

        std::vector<unsigned> expectedWriterSteps;
        std::vector<WriterEvent> expectedWriterEvents;

        expectedWriterSteps <<   0
                            <<   5
                            <<  10
                            <<  15
                            <<  20
                            <<  25
                            <<  30
                            <<  35
                            <<  40
                            <<  45
                            <<  50
                            <<  55
                            <<  60
                            <<  65
                            <<  70
                            <<  75
                            <<  80
                            <<  85
                            <<  90
                            <<  95
                            << 100
                            << 105
                            << 110
                            << 115
                            << 120;

        expectedWriterEvents << WRITER_INITIALIZED
                             << WRITER_STEP_FINISHED
                             << WRITER_STEP_FINISHED
                             << WRITER_STEP_FINISHED
                             << WRITER_STEP_FINISHED
                             << WRITER_STEP_FINISHED
                             << WRITER_STEP_FINISHED
                             << WRITER_STEP_FINISHED
                             << WRITER_STEP_FINISHED
                             << WRITER_STEP_FINISHED
                             << WRITER_STEP_FINISHED
                             << WRITER_STEP_FINISHED
                             << WRITER_STEP_FINISHED
                             << WRITER_STEP_FINISHED
                             << WRITER_STEP_FINISHED
                             << WRITER_STEP_FINISHED
                             << WRITER_STEP_FINISHED
                             << WRITER_STEP_FINISHED
                             << WRITER_STEP_FINISHED
                             << WRITER_STEP_FINISHED
                             << WRITER_STEP_FINISHED
                             << WRITER_STEP_FINISHED
                             << WRITER_STEP_FINISHED
                             << WRITER_STEP_FINISHED
                             << WRITER_ALL_DONE;

        sim.addWriter(new ParallelTestWriter<TestCell<3> >(5, expectedWriterSteps, expectedWriterEvents));
        sim.run();
    }

    void testWriterFunctionality3DWithGhostZoneWidth4()
    {
        typedef HiParSimulator<TestCell<3>, ZCurvePartition<3> > SimulatorType;
        int maxTimeSteps = 90;
        Coord<3> dim(20, 25, 30);

        TestInitializer<TestCell<3> > *init = new TestInitializer<TestCell<3> >(dim, maxTimeSteps);
        int loadBalancingPeriod = 10;
        int ghostZoneWidth = 4;
        SimulatorType sim(
            init,
            new MockBalancer(),
            loadBalancingPeriod,
            ghostZoneWidth);

        std::vector<unsigned> expectedWriterSteps;
        std::vector<WriterEvent> expectedWriterEvents;

        expectedWriterSteps <<  0
                            <<  5
                            << 10
                            << 15
                            << 20
                            << 25
                            << 30
                            << 35
                            << 40
                            << 45
                            << 50
                            << 55
                            << 60
                            << 65
                            << 70
                            << 75
                            << 80
                            << 85
                            << 90;

        expectedWriterEvents << WRITER_INITIALIZED
                             << WRITER_STEP_FINISHED
                             << WRITER_STEP_FINISHED
                             << WRITER_STEP_FINISHED
                             << WRITER_STEP_FINISHED
                             << WRITER_STEP_FINISHED
                             << WRITER_STEP_FINISHED
                             << WRITER_STEP_FINISHED
                             << WRITER_STEP_FINISHED
                             << WRITER_STEP_FINISHED
                             << WRITER_STEP_FINISHED
                             << WRITER_STEP_FINISHED
                             << WRITER_STEP_FINISHED
                             << WRITER_STEP_FINISHED
                             << WRITER_STEP_FINISHED
                             << WRITER_STEP_FINISHED
                             << WRITER_STEP_FINISHED
                             << WRITER_STEP_FINISHED
                             << WRITER_ALL_DONE;

        sim.addWriter(new ParallelTestWriter<TestCell<3> >(5, expectedWriterSteps, expectedWriterEvents));
        sim.run();
    }

    void testWriterFunctionality3DWithGhostZoneWidth5()
    {
        typedef HiParSimulator<TestCell<3>, ZCurvePartition<3> > SimulatorType;
        int maxTimeSteps = 90;
        Coord<3> dim(20, 25, 30);

        TestInitializer<TestCell<3> > *init = new TestInitializer<TestCell<3> >(dim, maxTimeSteps);
        int loadBalancingPeriod = 10;
        int ghostZoneWidth = 5;
        SimulatorType sim(
            init,
            new MockBalancer(),
            loadBalancingPeriod,
            ghostZoneWidth);

        std::vector<unsigned> expectedWriterSteps;
        std::vector<WriterEvent> expectedWriterEvents;

        expectedWriterSteps <<  0
                            <<  5
                            << 10
                            << 15
                            << 20
                            << 25
                            << 30
                            << 35
                            << 40
                            << 45
                            << 50
                            << 55
                            << 60
                            << 65
                            << 70
                            << 75
                            << 80
                            << 85
                            << 90;

        expectedWriterEvents << WRITER_INITIALIZED
                             << WRITER_STEP_FINISHED
                             << WRITER_STEP_FINISHED
                             << WRITER_STEP_FINISHED
                             << WRITER_STEP_FINISHED
                             << WRITER_STEP_FINISHED
                             << WRITER_STEP_FINISHED
                             << WRITER_STEP_FINISHED
                             << WRITER_STEP_FINISHED
                             << WRITER_STEP_FINISHED
                             << WRITER_STEP_FINISHED
                             << WRITER_STEP_FINISHED
                             << WRITER_STEP_FINISHED
                             << WRITER_STEP_FINISHED
                             << WRITER_STEP_FINISHED
                             << WRITER_STEP_FINISHED
                             << WRITER_STEP_FINISHED
                             << WRITER_STEP_FINISHED
                             << WRITER_ALL_DONE;

        sim.addWriter(new ParallelTestWriter<TestCell<3> >(5, expectedWriterSteps, expectedWriterEvents));
        sim.run();
    }

    void testWriterFunctionality3DWithGhostZoneWidth6()
    {
        typedef HiParSimulator<TestCell<3>, ZCurvePartition<3> > SimulatorType;
        int maxTimeSteps = 90;
        Coord<3> dim(20, 25, 30);

        TestInitializer<TestCell<3> > *init = new TestInitializer<TestCell<3> >(dim, maxTimeSteps);
        int loadBalancingPeriod = 10;
        int ghostZoneWidth = 6;
        SimulatorType sim(
            init,
            new MockBalancer(),
            loadBalancingPeriod,
            ghostZoneWidth);

        std::vector<unsigned> expectedWriterSteps;
        std::vector<WriterEvent> expectedWriterEvents;

        expectedWriterSteps <<  0
                            <<  5
                            << 10
                            << 15
                            << 20
                            << 25
                            << 30
                            << 35
                            << 40
                            << 45
                            << 50
                            << 55
                            << 60
                            << 65
                            << 70
                            << 75
                            << 80
                            << 85
                            << 90;

        expectedWriterEvents << WRITER_INITIALIZED
                             << WRITER_STEP_FINISHED
                             << WRITER_STEP_FINISHED
                             << WRITER_STEP_FINISHED
                             << WRITER_STEP_FINISHED
                             << WRITER_STEP_FINISHED
                             << WRITER_STEP_FINISHED
                             << WRITER_STEP_FINISHED
                             << WRITER_STEP_FINISHED
                             << WRITER_STEP_FINISHED
                             << WRITER_STEP_FINISHED
                             << WRITER_STEP_FINISHED
                             << WRITER_STEP_FINISHED
                             << WRITER_STEP_FINISHED
                             << WRITER_STEP_FINISHED
                             << WRITER_STEP_FINISHED
                             << WRITER_STEP_FINISHED
                             << WRITER_STEP_FINISHED
                             << WRITER_ALL_DONE;

        sim.addWriter(new ParallelTestWriter<TestCell<3> >(5, expectedWriterSteps, expectedWriterEvents));
        sim.run();
    }

    void testParallelWriterInvocation()
    {
        unsigned period = 4;
        std::vector<unsigned> expectedSteps;
        std::vector<WriterEvent> expectedEvents;
        expectedSteps << 20
                      << 24
                      << 28
                      << 32
                      << 36
                      << 40
                      << 44
                      << 48
                      << 52
                      << 56
                      << 60
                      << 64
                      << 68
                      << 72
                      << 76
                      << 80
                      << 84
                      << 88
                      << 92
                      << 96
                      << 100
                      << 101;

        expectedEvents << WRITER_INITIALIZED
                       << WRITER_STEP_FINISHED
                       << WRITER_STEP_FINISHED
                       << WRITER_STEP_FINISHED
                       << WRITER_STEP_FINISHED
                       << WRITER_STEP_FINISHED
                       << WRITER_STEP_FINISHED
                       << WRITER_STEP_FINISHED
                       << WRITER_STEP_FINISHED
                       << WRITER_STEP_FINISHED
                       << WRITER_STEP_FINISHED
                       << WRITER_STEP_FINISHED
                       << WRITER_STEP_FINISHED
                       << WRITER_STEP_FINISHED
                       << WRITER_STEP_FINISHED
                       << WRITER_STEP_FINISHED
                       << WRITER_STEP_FINISHED
                       << WRITER_STEP_FINISHED
                       << WRITER_STEP_FINISHED
                       << WRITER_STEP_FINISHED
                       << WRITER_STEP_FINISHED
                       << WRITER_ALL_DONE;

        sim->addWriter(new ParallelTestWriter<>(period, expectedSteps, expectedEvents));
        sim->run();
    }

    void testNonPoDCell()
    {
#ifdef LIBGEODECOMP_WITH_BOOST_SERIALIZATION

        // fixme: disabled until #46 is fixed
        // ghostZoneWidth = 3;

        // HiParSimulator<NonPoDTestCell, ZCurvePartition<2> > sim(
        //     new NonPoDTestCell::Initializer(),
        //     new MockBalancer(),
        //     loadBalancingPeriod,
        //     ghostZoneWidth);
        // sim.run();

#endif
    }

    void testIO( )
    {
        sim->addWriter(new AccumulatingWriter());
        sim->run();
    }

    void testSoA()
    {
        unsigned startStep = 0;
        unsigned endStep = 21;

        HiParSimulator<TestCellSoA, ZCurvePartition<3> > sim(
            new TestInitializer<TestCellSoA>(),
            rank? 0 : new NoOpBalancer());

        Writer<TestCellSoA> *writer = 0;
        if (rank == 0) {
            writer = new TestWriter<TestCellSoA>(3, startStep, endStep);
        }
        sim.addWriter(new CollectingWriter<TestCellSoA>(writer));

        sim.run();
    }

    void testUnstructured()
    {
#ifdef LIBGEODECOMP_WITH_CPP14
        typedef UnstructuredTestCell<> TestCellType;

        int startStep = 7;
        int endStep = 20;

        HiParSimulator<TestCellType, UnstructuredStripingPartition> sim(
            new UnstructuredTestInitializer<TestCellType>(614, endStep, startStep),
            rank? 0 : new NoOpBalancer());

        std::vector<unsigned> expectedSteps;
        std::vector<WriterEvent> expectedEvents;
        expectedSteps << 7
                      << 10
                      << 13
                      << 16
                      << 19
                      << 20;
        expectedEvents << WRITER_INITIALIZED
                       << WRITER_STEP_FINISHED
                       << WRITER_STEP_FINISHED
                       << WRITER_STEP_FINISHED
                       << WRITER_STEP_FINISHED
                       << WRITER_ALL_DONE;
        sim.addWriter(new ParallelTestWriter<TestCellType>(3, expectedSteps, expectedEvents));

        sim.run();
#endif
    }

    void testUnstructuredSoA1()
    {
#ifdef LIBGEODECOMP_WITH_CPP14
        typedef UnstructuredTestCellSoA1 TestCellType;
        int startStep = 7;
        int endStep = 20;

        HiParSimulator<TestCellType, UnstructuredStripingPartition> sim(
            new UnstructuredTestInitializer<TestCellType>(614, endStep, startStep),
            rank? 0 : new NoOpBalancer());

        std::vector<unsigned> expectedSteps;
        std::vector<WriterEvent> expectedEvents;
        expectedSteps << 7
                      << 11
                      << 15
                      << 19
                      << 20;
        expectedEvents << WRITER_INITIALIZED
                       << WRITER_STEP_FINISHED
                       << WRITER_STEP_FINISHED
                       << WRITER_STEP_FINISHED
                       << WRITER_ALL_DONE;
        sim.addWriter(new ParallelTestWriter<TestCellType>(4, expectedSteps, expectedEvents));

        sim.run();
#endif
    }

    void testUnstructuredSoA2()
    {
#ifdef LIBGEODECOMP_WITH_CPP14
        typedef UnstructuredTestCellSoA2 TestCellType;
        int startStep = 7;
        int endStep = 15;

        HiParSimulator<TestCellType, UnstructuredStripingPartition> sim(
            new UnstructuredTestInitializer<TestCellType>(632, endStep, startStep),
        rank? 0 : new NoOpBalancer());

        std::vector<unsigned> expectedSteps;
        std::vector<WriterEvent> expectedEvents;
        expectedSteps << 7
                      << 9
                      << 11
                      << 13
                      << 15;
        expectedEvents << WRITER_INITIALIZED
                       << WRITER_STEP_FINISHED
                       << WRITER_STEP_FINISHED
                       << WRITER_STEP_FINISHED
                       << WRITER_ALL_DONE;
        sim.addWriter(new ParallelTestWriter<TestCellType>(2, expectedSteps, expectedEvents));

        sim.run();
#endif
    }

    void testUnstructuredSoA3()
    {
#ifdef LIBGEODECOMP_WITH_CPP14
        typedef UnstructuredTestCellSoA3 TestCellType;
        int startStep = 7;
        int endStep = 19;

        HiParSimulator<TestCellType, UnstructuredStripingPartition> sim(
            new UnstructuredTestInitializer<TestCellType>(655, endStep, startStep),
        rank? 0 : new NoOpBalancer());

        std::vector<unsigned> expectedSteps;
        std::vector<WriterEvent> expectedEvents;
        expectedSteps << 7
                      << 10
                      << 13
                      << 16
                      << 19;
        expectedEvents << WRITER_INITIALIZED
                       << WRITER_STEP_FINISHED
                       << WRITER_STEP_FINISHED
                       << WRITER_STEP_FINISHED
                       << WRITER_ALL_DONE;
        sim.addWriter(new ParallelTestWriter<TestCellType>(3, expectedSteps, expectedEvents));

        sim.run();
#endif
    }

    void testUnstructuredSoA4()
    {
#ifdef LIBGEODECOMP_WITH_CPP14
        typedef UnstructuredTestCellSoA1 TestCellType;
        int startStep = 5;
        int endStep = 24;

        HiParSimulator<TestCellType, UnstructuredStripingPartition> sim(
            new UnstructuredTestInitializer<TestCellType>(444, endStep, startStep),
            rank? 0 : new NoOpBalancer());

        std::vector<unsigned> expectedSteps;
        std::vector<WriterEvent> expectedEvents;
        expectedSteps << 5
                      << 8
                      << 11
                      << 14
                      << 17
                      << 20
                      << 23
                      << 24;
        expectedEvents << WRITER_INITIALIZED
                       << WRITER_STEP_FINISHED
                       << WRITER_STEP_FINISHED
                       << WRITER_STEP_FINISHED
                       << WRITER_STEP_FINISHED
                       << WRITER_STEP_FINISHED
                       << WRITER_STEP_FINISHED
                       << WRITER_ALL_DONE;
        sim.addWriter(new ParallelTestWriter<TestCellType>(3, expectedSteps, expectedEvents));

        sim.run();
#endif
    }

private:
    SharedPtr<SimulatorType>::Type sim;
    Coord<2> dim;
    unsigned maxSteps;
    unsigned firstStep;
    unsigned firstCycle;
    unsigned outputPeriod;
    unsigned loadBalancingPeriod;
    unsigned ghostZoneWidth;
    SharedPtr<MockWriter<>::EventsStore>::Type events;
    MockWriter<> *mockWriter;
    MemoryWriterType *memoryWriter;
    std::size_t rank;
};

}
