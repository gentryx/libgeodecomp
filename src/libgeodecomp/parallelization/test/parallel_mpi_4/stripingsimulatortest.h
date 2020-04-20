#include <cxxtest/TestSuite.h>
#include <libgeodecomp/communication/typemaps.h>
#include <libgeodecomp/io/clonableinitializerwrapper.h>
#include <libgeodecomp/io/collectingwriter.h>
#include <libgeodecomp/io/memorywriter.h>
#include <libgeodecomp/io/mockwriter.h>
#include <libgeodecomp/io/paralleltestwriter.h>
#include <libgeodecomp/io/testinitializer.h>
#include <libgeodecomp/io/teststeerer.h>
#include <libgeodecomp/io/testwriter.h>
#include <libgeodecomp/io/unstructuredtestinitializer.h>
#include <libgeodecomp/loadbalancer/noopbalancer.h>
#include <libgeodecomp/loadbalancer/randombalancer.h>
#include <libgeodecomp/misc/nonpodtestcell.h>
#include <libgeodecomp/misc/testhelper.h>
#include <libgeodecomp/misc/unstructuredtestcell.h>
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


class StripingSimulatorTest : public CxxTest::TestSuite
{
public:
    typedef GridBase<TestCell<2>, 2> GridBaseType;
    typedef TestSteerer<2> TestSteererType;

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

    void testNeighbors()
    {
        if ((rank > 0) && (rank < 3)) {
            TS_ASSERT_EQUALS(2, testSim->outerGhostRegions.size());
            TS_ASSERT_EQUALS(2, testSim->innerGhostRegions.size());
        } else {
            TS_ASSERT_EQUALS(1, testSim->outerGhostRegions.size());
            TS_ASSERT_EQUALS(1, testSim->innerGhostRegions.size());
        }

        if (rank > 0) {
            TS_ASSERT(!testSim->outerGhostRegions[rank - 1].empty());
            TS_ASSERT(!testSim->innerGhostRegions[rank - 1].empty());
        }

        if (rank < 3) {
            TS_ASSERT(!testSim->outerGhostRegions[rank + 1].empty());
            TS_ASSERT(!testSim->innerGhostRegions[rank + 1].empty());
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

        for (int i = 0; i < 40; ++i) {
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
        SharedPtr<MockWriter<>::EventsStore>::Type expectedEvents(new MockWriter<>::EventsStore);

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
        testSim->waitForGhostRegions(testSim->newStripe);

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
        testSim->waitForGhostRegions(testSim->curStripe);

        unsigned ghostHeightUpper = (rank > 0) ? 1 : 0;
        unsigned ghostHeightLower = (rank < 3) ? 1 : 0;
        int width = init->gridBox().dimensions.x();

        if (weights2[rank] == weights2[rank + 1]) {
            ghostHeightUpper = 0;
            ghostHeightLower = 0;
            width = 0;
        }

        unsigned s = weights2[rank    ] - ghostHeightUpper;
        unsigned e = weights2[rank + 1] + ghostHeightLower;

        DisplacedGrid<TestCell<2> > expectedStripe(
            CoordBox<2>(Coord<2>(0, s), Coord<2>(width, e - s)));
        init->grid(&expectedStripe);
        Grid<TestCell<2> > actualStripe = *testSim->curStripe->vanillaGrid();

        TS_ASSERT_EQUALS(actualStripe.boundingBox(), expectedStripe.vanillaGrid()->boundingBox());
        TS_ASSERT_EQUALS(actualStripe, *expectedStripe.vanillaGrid());
    }

    void checkRunWithDifferentPartitions(NoOpBalancer::WeightVec weights2)
    {
        NoOpBalancer::WeightVec weights1 = testSim->partitions;
        testSim->redistributeGrid(weights1, weights2);
        testSim->waitForGhostRegions(testSim->newStripe);

        TS_ASSERT_TEST_GRID_REGION(
            GridBaseType,
            *testSim->curStripe,
            testSim->regionWithOuterGhosts,
            540);
        testSim->nanoStep(0);
        testSim->waitForGhostRegions(testSim->curStripe);

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
            TS_ASSERT_EQUALS((int)testSim->curStripe->getDimensions().y(), 4);
            TS_ASSERT_EQUALS((int)testSim->newStripe->getDimensions().y(), 4);
            break;
        case 1:
            TS_ASSERT_EQUALS((int)testSim->curStripe->getDimensions().y(), 0);
            TS_ASSERT_EQUALS((int)testSim->newStripe->getDimensions().y(), 0);
            break;
        case 2:
            TS_ASSERT_EQUALS((int)testSim->curStripe->getDimensions().y(), 9);
            TS_ASSERT_EQUALS((int)testSim->newStripe->getDimensions().y(), 9);
            break;
        case 3:
            TS_ASSERT_EQUALS((int)testSim->curStripe->getDimensions().y(), 3);
            TS_ASSERT_EQUALS((int)testSim->newStripe->getDimensions().y(), 3);
            break;
        }
    }

    void checkLoadBalancingRealistically(unsigned balanceEveryN)
    {
        SharedPtr<MockWriter<>::EventsStore>::Type expectedEvents(new MockWriter<>::EventsStore);

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
                std::invalid_argument&);
        } else {
            TS_ASSERT_THROWS(
                StripingSimulator<TestCell<2> > s(new TestInitializer<TestCell<2> >(),
                                                  new NoOpBalancer),
                std::invalid_argument&);
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

        testSim->addWriter(new ParallelTestWriter<>(period, expectedSteps, expectedEvents));
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

// fixme
//     void testNonPoDCellLittle()
//     {
//         std::cout << "testNonPoDCellLittle\n";
// #ifdef LIBGEODECOMP_WITH_BOOST_SERIALIZATION
//         int scalingFactor = 1;

//         StripingSimulator<NonPoDTestCell> sim(
//             new NonPoDTestCell::Initializer(scalingFactor),
//             rank? 0 : new NoOpBalancer());
//         sim.run();
// #endif
//     }

//     void testNonPoDCellBig()
//     {
//         std::cout << "testNonPoDCellBig\n";
// #ifdef LIBGEODECOMP_WITH_BOOST_SERIALIZATION
//         int scalingFactor = 3;

//         StripingSimulator<NonPoDTestCell> sim(
//             new NonPoDTestCell::Initializer(scalingFactor),
//             rank? 0 : new NoOpBalancer());
//         sim.run();
// #endif
//     }

    void testSoA()
    {
        int startStep = 0;
        int endStep = 21;

        StripingSimulator<TestCellSoA> sim(
            new TestInitializer<TestCellSoA>(),
            rank? 0 : new NoOpBalancer());

        Writer<TestCellSoA> *writer = 0;
        if (MPILayer().rank() == 0) {
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

        StripingSimulator<TestCellType> sim(
            new UnstructuredTestInitializer<TestCellType>(614, endStep, startStep),
            rank? 0 : new NoOpBalancer());

        Writer<TestCellType> *writer = 0;
        if (MPILayer().rank() == 0) {
            writer = new TestWriter<TestCellType>(3, startStep, endStep);
        }
        sim.addWriter(new CollectingWriter<TestCellType>(writer));

        sim.run();
#endif
    }

    void testUnstructuredSoA1()
    {
#ifdef LIBGEODECOMP_WITH_CPP14
        typedef UnstructuredTestCellSoA1 TestCellType;
        int startStep = 7;
        int endStep = 20;

        StripingSimulator<TestCellType> sim(
            new UnstructuredTestInitializer<TestCellType>(614, endStep, startStep),
            rank? 0 : new NoOpBalancer());

        Writer<TestCellType> *writer = 0;
        if (MPILayer().rank() == 0) {
            writer = new TestWriter<TestCellType>(3, startStep, endStep);
        }
        sim.addWriter(new CollectingWriter<TestCellType>(writer));

        sim.run();
#endif
    }

    void testUnstructuredSoA2()
    {
#ifdef LIBGEODECOMP_WITH_CPP14
        typedef UnstructuredTestCellSoA2 TestCellType;
        int startStep = 7;
        int endStep = 15;

        StripingSimulator<TestCellType> sim(new UnstructuredTestInitializer<TestCellType>(632, endStep, startStep),
        rank? 0 : new NoOpBalancer());

        Writer<TestCellType> *writer = 0;
        if (MPILayer().rank() == 0) {
            writer = new TestWriter<TestCellType>(3, startStep, endStep);
        }
        sim.addWriter(new CollectingWriter<TestCellType>(writer));

        sim.run();
#endif
    }

    void testUnstructuredSoA3()
    {
#ifdef LIBGEODECOMP_WITH_CPP14
        typedef UnstructuredTestCellSoA3 TestCellType;
        int startStep = 7;
        int endStep = 19;

        StripingSimulator<TestCellType> sim(new UnstructuredTestInitializer<TestCellType>(655, endStep, startStep),
        rank? 0 : new NoOpBalancer());

        Writer<TestCellType> *writer = 0;
        if (MPILayer().rank() == 0) {
            writer = new TestWriter<TestCellType>(3, startStep, endStep);
        }
        sim.addWriter(new CollectingWriter<TestCellType>(writer));

        sim.run();
#endif
    }

    void testUnstructuredSoA4()
    {
#ifdef LIBGEODECOMP_WITH_CPP14
        typedef UnstructuredTestCellSoA1 TestCellType;
        int startStep = 5;
        int endStep = 24;

        StripingSimulator<TestCellType> sim(
            new UnstructuredTestInitializer<TestCellType>(444, endStep, startStep),
            rank? 0 : new NoOpBalancer());

        Writer<TestCellType> *writer = 0;
        if (MPILayer().rank() == 0) {
            writer = new TestWriter<TestCellType>(3, startStep, endStep);
        }
        sim.addWriter(new CollectingWriter<TestCellType>(writer));
        sim.run();
#endif
    }

private:
    SharedPtr<MockWriter<>::EventsStore>::Type events;
    static const unsigned NANO_STEPS = APITraits::SelectNanoSteps<TestCell<2> >::VALUE;
    SharedPtr<MPILayer>::Type layer;
    SharedPtr<MonolithicSimulator<TestCell<2> > >::Type referenceSim;
    SharedPtr<StripingSimulator<TestCell<2> > >::Type testSim;
    SharedPtr<ClonableInitializer<TestCell<2> > >::Type init;
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
