#include <cxxtest/TestSuite.h>
#include <libgeodecomp/communication/typemaps.h>
#include <libgeodecomp/io/clonableinitializerwrapper.h>
#include <libgeodecomp/io/memorywriter.h>
#include <libgeodecomp/io/mockwriter.h>
#include <libgeodecomp/io/paralleltestwriter.h>
#include <libgeodecomp/io/testinitializer.h>
#include <libgeodecomp/io/teststeerer.h>
#include <libgeodecomp/loadbalancer/noopbalancer.h>
#include <libgeodecomp/loadbalancer/randombalancer.h>
#include <libgeodecomp/misc/testhelper.h>
#include <libgeodecomp/parallelization/serialsimulator.h>
#include <libgeodecomp/parallelization/stripingsimulator.h>

using namespace LibGeoDecomp;

namespace LibGeoDecomp {

class CheckBalancer : public LoadBalancer
{
public:
    virtual NoOpBalancer::WeightVec balance(
        const NoOpBalancer::WeightVec& weights,
        const NoOpBalancer::LoadVec& relativeLoads)
    {
        for (unsigned i = 0; i < relativeLoads.size(); i++) {
            TS_ASSERT(relativeLoads[i] >= 0);
            TS_ASSERT(relativeLoads[i] < 1);
        }
        return weights;
    }

};


class ParallelStripingSimulatorTest : public CxxTest::TestSuite
{
public:
    typedef GridBase<TestCell<2>, 2> GridBaseType;
    typedef TestSteerer<2 > TestSteererType;

    void setUp()
    {
        layer.reset(new MPILayer());
        rank = layer->rank();
        size = layer->size();

        width = 17;
        height = 12;
        dim = Coord<2>(width, height);
        maxSteps = 50;
        firstStep = 20;
        firstCycle = firstStep * NANO_STEPS;
        init.reset(ClonableInitializerWrapper<TestInitializer<TestCell<2> > >::wrap(
                       TestInitializer<TestCell<2> >(dim, maxSteps, firstStep)));

        referenceSim.reset(new SerialSimulator<TestCell<2> >(init->clone()));

        LoadBalancer *balancer = rank == 0? new NoOpBalancer : 0;
        testSim.reset(new StripingSimulator<TestCell<2> >(init, balancer));
        events.reset(new MockWriter<>::EventsStore);
    }

    void tearDown()
    {
        referenceSim.reset();
        testSim.reset();
        layer.reset();
    }

    void testGhostHeight()
    {
        if (rank == 0) {
            TS_ASSERT_EQUALS(unsigned(0), testSim->ghostHeightUpper);
        } else {
            TS_ASSERT_EQUALS(unsigned(1), testSim->ghostHeightUpper);
        }

        if (rank == (size - 1)) {
            TS_ASSERT_EQUALS(unsigned(0), testSim->ghostHeightLower);
        } else {
            TS_ASSERT_EQUALS(unsigned(1), testSim->ghostHeightLower);
        }
    }

    void testNeighbors()
    {
        if (rank == 0) {
            TS_ASSERT_EQUALS(-1, testSim->upperNeighbor());
            TS_ASSERT_EQUALS( 1, testSim->lowerNeighbor());
        }
        if (rank == 1) {
            TS_ASSERT_EQUALS( 0, testSim->upperNeighbor());
            TS_ASSERT_EQUALS( 2, testSim->lowerNeighbor());
        }
        if (rank == 2) {
            TS_ASSERT_EQUALS( 1, testSim->upperNeighbor());
            TS_ASSERT_EQUALS( 3, testSim->lowerNeighbor());
        }
        if (rank == 3) {
            TS_ASSERT_EQUALS( 2, testSim->upperNeighbor());
            TS_ASSERT_EQUALS(-1, testSim->lowerNeighbor());
        }
    }

    void testInitRegions()
    {
        Region<2> regions[4];
        if (rank != 0) {
            layer->sendRegion(testSim->region, 0);
        }
        else {
            regions[0] = testSim->region;
            for (int i = 1; i < 4; ++i) {
                layer->recvRegion(&regions[i], i);
            }
        }
        layer->waitAll();

        if (rank == 0) {
            Region<2> whole;
            for (int i = 0; i < 4; ++i) {
                whole += regions[i];
            }

            Region<2> expected;
            CoordBox<2> box = init->gridBox();
            for (CoordBox<2>::Iterator i = box.begin(); i != box.end(); ++i) {
                expected << *i;
            }

            TS_ASSERT_EQUALS(expected, whole);
        }
    }

    void testStep()
    {
        TS_ASSERT_EQUALS(referenceSim->getStep(),
                         testSim->getStep());

        int cycle = firstCycle;

        TS_ASSERT_TEST_GRID_REGION(
            GridBaseType,
            *testSim->curStripe,
            testSim->region,
            cycle);

        for (int i = 0; i < 40; i++) {
            referenceSim->step();
            testSim->step();
            cycle += NANO_STEPS;

            TS_ASSERT_EQUALS(referenceSim->getStep(),
                             testSim->getStep());

            TS_ASSERT_TEST_GRID_REGION(
                GridBaseType,
                *testSim->curStripe,
                testSim->region,
                cycle);
        }
    }

    void testRun()
    {
        testSim->run();
        referenceSim->run();

        TS_ASSERT_EQUALS(init->maxSteps(), testSim->getStep());

        int cycle = maxSteps * NANO_STEPS;
        TS_ASSERT_TEST_GRID_REGION(
            GridBaseType,
            *testSim->curStripe,
            testSim->region,
            cycle);
    }

    void checkRunAndWriterInteraction(int everyN)
    {
        boost::shared_ptr<MockWriter<>::EventsStore> expectedEvents(new MockWriter<>::EventsStore);

        MockWriter<> *expectedCalls = new MockWriter<>(expectedEvents);
        MockWriter<> *actualCalls = new MockWriter<>(events);

        referenceSim->addWriter(expectedCalls);
        testSim->addWriter(actualCalls);

        testSim->run();
        referenceSim->run();

        for (MockWriter<>::EventsStore::iterator i = expectedEvents->begin();
             i != expectedEvents->end();
             ++i) {
            i->rank = MPILayer().rank();
        }

        TS_ASSERT_EQUALS(expectedEvents->size(), events->size());

        MockWriter<>::EventsStore::iterator j = events->begin();
        for (MockWriter<>::EventsStore::iterator i = expectedEvents->begin();
             i != expectedEvents->end();
             ++i, ++j) {
            TS_ASSERT_EQUALS(*i, *j);
        }
    }

    void testEveryN1()
    {
        checkRunAndWriterInteraction(1);
    }

    void testEveryN7()
    {
        checkRunAndWriterInteraction(7);
    }

    void testRedistributeGrid1()
    {
        NoOpBalancer::WeightVec weights1 = testSim->partitions;
        NoOpBalancer::WeightVec weights2 = toMonoPartitions(weights1);

        testSim->redistributeGrid(weights1, weights2);
        testSim->waitForGhostRegions();

        TS_ASSERT_EQUALS(testSim->ghostHeightLower, (unsigned)0);
        TS_ASSERT_EQUALS(testSim->ghostHeightUpper, (unsigned)0);
        if (rank == 0) {
            TS_ASSERT_EQUALS(testSim->curStripe->getDimensions(),
                             init->gridDimensions());
            Grid<TestCell<2> > expected(init->gridBox().dimensions);
            init->grid(&expected);
            TS_ASSERT_EQUALS(*testSim->curStripe->vanillaGrid(), expected);
        } else {
            TS_ASSERT_EQUALS((int)testSim->curStripe->getDimensions().y(), 0);
        }
    }

    void testRedistributeGrid2()
    {
        NoOpBalancer::WeightVec weights1 = testSim->partitions;
        NoOpBalancer::WeightVec weights2 = toWeirdoPartitions(weights1);

        testSim->redistributeGrid(weights1, weights2);
        testSim->waitForGhostRegions();


        unsigned s = weights2[rank    ] - testSim->ghostHeightUpper;
        unsigned e = weights2[rank + 1] + testSim->ghostHeightLower;

        int width = init->gridBox().dimensions.x();
        DisplacedGrid<TestCell<2> > expectedStripe(
            CoordBox<2>(Coord<2>(0, s), Coord<2>(width, e - s)));
        init->grid(&expectedStripe);
        Grid<TestCell<2> > actualStripe = *testSim->curStripe->vanillaGrid();
        TS_ASSERT_EQUALS(actualStripe, *expectedStripe.vanillaGrid());
    }

    void checkRunWithDifferentPartitions(NoOpBalancer::WeightVec weights2)
    {
        NoOpBalancer::WeightVec weights1 = testSim->partitions;
        testSim->redistributeGrid(weights1, weights2);
        testSim->waitForGhostRegions();

        testSim->run();
        referenceSim->run();

        int cycle = maxSteps * NANO_STEPS;

        TS_ASSERT_TEST_GRID_REGION(
            GridBaseType,
            *testSim->curStripe,
            testSim->region,
            cycle);
        TS_ASSERT_EQUALS(testSim->partitions, weights2);
        TS_ASSERT_EQUALS(init->maxSteps(), testSim->getStep());
    }

    void testRedistributeGrid3()
    {
        checkRunWithDifferentPartitions(
            toWeirdoPartitions(testSim->partitions));
    }

    void testRedistributeGrid4()
    {
        checkRunWithDifferentPartitions(
            toMonoPartitions(testSim->partitions));
    }

    void testRedistributeGrid5()
    {
        NoOpBalancer::WeightVec weights1 = testSim->partitions;
        NoOpBalancer::WeightVec weights2 = toWeirdoPartitions(weights1);

        testSim->redistributeGrid(weights1, weights2);

        TS_ASSERT_EQUALS(testSim->partitions, weights2);
        switch (rank) {
        case 0:
            TS_ASSERT_EQUALS((int)testSim->ghostHeightUpper, 0);
            TS_ASSERT_EQUALS((int)testSim->ghostHeightLower, 1);
            TS_ASSERT_EQUALS((int)testSim->curStripe->getDimensions().y(), 4);
            TS_ASSERT_EQUALS((int)testSim->newStripe->getDimensions().y(), 4);
            break;
        case 1:
            TS_ASSERT_EQUALS((int)testSim->ghostHeightUpper, 0);
            TS_ASSERT_EQUALS((int)testSim->ghostHeightLower, 0);
            TS_ASSERT_EQUALS((int)testSim->curStripe->getDimensions().y(), 0);
            TS_ASSERT_EQUALS((int)testSim->newStripe->getDimensions().y(), 0);
            break;
        case 2:
            TS_ASSERT_EQUALS((int)testSim->ghostHeightUpper, 1);
            TS_ASSERT_EQUALS((int)testSim->ghostHeightLower, 1);
            TS_ASSERT_EQUALS((int)testSim->curStripe->getDimensions().y(), 9);
            TS_ASSERT_EQUALS((int)testSim->newStripe->getDimensions().y(), 9);
            break;
        case 3:
            TS_ASSERT_EQUALS((int)testSim->ghostHeightUpper, 1);
            TS_ASSERT_EQUALS((int)testSim->ghostHeightLower, 0);
            TS_ASSERT_EQUALS((int)testSim->curStripe->getDimensions().y(), 3);
            TS_ASSERT_EQUALS((int)testSim->newStripe->getDimensions().y(), 3);
            break;
        }
    }

    void checkLoadBalancingRealistically(unsigned balanceEveryN)
    {
        boost::shared_ptr<MockWriter<>::EventsStore> expectedEvents(new MockWriter<>::EventsStore);

        LoadBalancer *balancer = rank? 0 : new RandomBalancer;
        StripingSimulator<TestCell<2> > localTestSim(
            new TestInitializer<TestCell<2> >(dim, maxSteps, firstStep),
            balancer,
            balanceEveryN);

        MockWriter<> *expectedCalls = new MockWriter<>(expectedEvents);
        MockWriter<> *actualCalls = new MockWriter<>(events);
        referenceSim->addWriter(expectedCalls);
        localTestSim.addWriter(actualCalls);

        localTestSim.run();
        referenceSim->run();

        for (MockWriter<>::EventsStore::iterator i = expectedEvents->begin();
             i != expectedEvents->end();
             ++i) {
            i->rank = MPILayer().rank();
        }

        TS_ASSERT_EQUALS(expectedEvents->size(), events->size());

        MockWriter<>::EventsStore::iterator j = events->begin();
        for (MockWriter<>::EventsStore::iterator i = expectedEvents->begin();
             i != expectedEvents->end();
             ++i, ++j) {
            TS_ASSERT_EQUALS(*i, *j);
        }
    }

    void testBalanceLoad1()
    {
        checkLoadBalancingRealistically(1);
    }

    void testBalanceLoad2()
    {
        checkLoadBalancingRealistically(2);
    }

    void testBalanceLoad3()
    {
        checkLoadBalancingRealistically(4);
    }

    void testBalanceLoad4()
    {
        checkLoadBalancingRealistically(7);
    }

    void testLoadGathering()
    {
        LoadBalancer *balancer = rank? 0 : new CheckBalancer;
        unsigned balanceEveryN = 2;
        StripingSimulator<TestCell<2> > testSim(
            new TestInitializer<TestCell<2> >(),
            balancer,
            balanceEveryN);
        testSim.run();
    }

    void testEmptyBalancer()
    {
        if (rank == 0) {
            TS_ASSERT_THROWS(
                StripingSimulator<TestCell<2> > s(new TestInitializer<TestCell<2> >(), 0),
                std::invalid_argument);
        } else {
            TS_ASSERT_THROWS(
                StripingSimulator<TestCell<2> > s(new TestInitializer<TestCell<2> >(),
                                                  new NoOpBalancer),
                std::invalid_argument);
        }
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
                      << 50;
        expectedEvents << WRITER_INITIALIZED
                       << WRITER_STEP_FINISHED
                       << WRITER_STEP_FINISHED
                       << WRITER_STEP_FINISHED
                       << WRITER_STEP_FINISHED
                       << WRITER_STEP_FINISHED
                       << WRITER_STEP_FINISHED
                       << WRITER_STEP_FINISHED
                       << WRITER_ALL_DONE;

        testSim->addWriter(new ParallelTestWriter(period, expectedSteps, expectedEvents));
        testSim->run();
    }

    void test3Dsimple()
    {
        StripingSimulator<TestCell<3> > s(
            new TestInitializer<TestCell<3> >(),
            rank? 0 : new NoOpBalancer());

        s.run();
    }

    void test3Dadvanced()
    {
        StripingSimulator<TestCell<3> > s(
            new TestInitializer<TestCell<3> >(),
            rank? 0 : new RandomBalancer());

        s.run();
    }

    void testSteererFunctionality()
    {
        testSim->addSteerer(new TestSteererType(5, 25, 4711 * 27));
        testSim->run();

        int cycle = 50 * 27 + 4711 * 27;

        TS_ASSERT_TEST_GRID_REGION(
            GridBaseType,
            *testSim->curStripe,
            testSim->region,
            cycle);
    }

private:
    boost::shared_ptr<MockWriter<>::EventsStore> events;
    static const unsigned NANO_STEPS = APITraits::SelectNanoSteps<TestCell<2> >::VALUE;
    boost::shared_ptr<MPILayer> layer;
    boost::shared_ptr<MonolithicSimulator<TestCell<2> > > referenceSim;
    boost::shared_ptr<StripingSimulator<TestCell<2> > > testSim;
    boost::shared_ptr<ClonableInitializer<TestCell<2> > > init;
    int rank;
    int size;
    int width;
    int height;
    Coord<2> dim;
    int maxSteps;
    int firstStep;
    int firstCycle;

    // just a boring partition: everything on _one_ node
    NoOpBalancer::WeightVec toMonoPartitions(NoOpBalancer::WeightVec weights1)
    {
        NoOpBalancer::WeightVec weights2(5);
        weights2[0] = 0;
        weights2[1] = weights1[4];
        weights2[2] = weights1[4];
        weights2[3] = weights1[4];
        weights2[4] = weights1[4];
        return weights2;
    }

    // a weird new partition, max. difference to the uniform one
    NoOpBalancer::WeightVec toWeirdoPartitions(NoOpBalancer::WeightVec weights1)
    {
        NoOpBalancer::WeightVec weights2(5);
        weights2[0] = 0;
        weights2[1] = 3;
        weights2[2] = 3;
        weights2[3] = 10;
        weights2[4] = weights1[4];
        // just to be sure the config file hasn't been tainted
        TS_ASSERT(weights1[4] > 10);

        return weights2;
    }

    Grid<TestCell<2> > mangledGrid(double foo)
    {
        Grid<TestCell<2> > grid(init->gridBox().dimensions);
        init->grid(&grid);
        for (int y = 0; y < grid.getDimensions().y(); y++) {
            for (int x = 0; x < grid.getDimensions().x(); x++) {
                grid[Coord<2>(x, y)].testValue = foo + y * grid.getDimensions().y() + x;
            }
        }
        return grid;
    }
};

};
