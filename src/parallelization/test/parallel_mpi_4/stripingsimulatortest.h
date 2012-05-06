#include <cxxtest/TestSuite.h>
#include <libgeodecomp/io/memorywriter.h>
#include <libgeodecomp/io/mockwriter.h>
#include <libgeodecomp/io/parallelmockwriter.h>
#include <libgeodecomp/io/testinitializer.h>
#include <libgeodecomp/loadbalancer/noopbalancer.h>
#include <libgeodecomp/loadbalancer/randombalancer.h>
#include <libgeodecomp/misc/testhelper.h>
#include <libgeodecomp/mpilayer/typemaps.h>
#include <libgeodecomp/parallelization/serialsimulator.h>
#include <libgeodecomp/parallelization/stripingsimulator.h>

using namespace LibGeoDecomp; 

namespace LibGeoDecomp {

class CheckBalancer : public LoadBalancer
{
public:
    virtual UVec balance(const UVec& currentLoads, const DVec& relativeLoads)
    {
        for (unsigned i = 0; i < relativeLoads.size(); i++) {
            TS_ASSERT(relativeLoads[i] >= 0);
            TS_ASSERT(relativeLoads[i] < 1);
        }
        return currentLoads;
    }

};


class ParallelStripingSimulatorTest : public CxxTest::TestSuite
{
private:
    typedef GridBase<TestCell<2>, 2> GridBaseType;

    MonolithicSimulator<TestCell<2> > *referenceSim;
    StripingSimulator<TestCell<2> > *testSim;
    int rank;
    int size;
    int width;
    int height;
    int maxSteps;
    int firstStep;
    int firstCycle;
    Initializer<TestCell<2> > *init;
    MPILayer layer;

public:
    void setUp() 
    {
        rank = layer.rank();
        size = layer.size();

        width = 17;
        height = 12;
        maxSteps = 50;
        firstStep = 20;
        firstCycle = firstStep * TestCell<2>::nanoSteps();
        init = new TestInitializer<2>(
            Coord<2>(width, height), maxSteps, firstStep);

        referenceSim = new SerialSimulator<TestCell<2> >(
            new TestInitializer<2>(
                Coord<2>(width, height), maxSteps, firstStep));
        LoadBalancer *balancer = rank == 0? new NoOpBalancer : 0;
        testSim = new StripingSimulator<TestCell<2> >(init, balancer);
    }

    void tearDown()
    {
        delete referenceSim;
        delete testSim;
    }

    void testGhostHeight()
    {
        if (rank == 0) {
            TS_ASSERT_EQUALS(0, testSim->ghostHeightUpper);
        } else {
            TS_ASSERT_EQUALS(1, testSim->ghostHeightUpper);
        }

        if (rank == (size - 1)) {
            TS_ASSERT_EQUALS(0, testSim->ghostHeightLower);
        } else {
            TS_ASSERT_EQUALS(1, testSim->ghostHeightLower);
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
        layer.sendRegion(testSim->region, 0);
        if (rank == 0) {
            for (int i = 0; i < 4; ++i) 
                layer.recvRegion(&regions[i], i);
        }
        layer.waitAll();

        if (rank == 0) {
            Region<2> whole;
            for (int i = 0; i < 4; ++i) 
                whole += regions[i];

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
        
        const Region<2> *region;
        const GridBaseType *grid;
        int cycle = firstCycle;

        testSim->getGridFragment(&grid, &region);
        TS_ASSERT_TEST_GRID_REGION(
            GridBaseType, 
            *grid, 
            *region, 
            cycle);

        for (int i = 0; i < 40; i++) {
            referenceSim->step();
            testSim->step();
            cycle += TestCell<2>::nanoSteps();

            TS_ASSERT_EQUALS(referenceSim->getStep(), 
                             testSim->getStep());

            testSim->getGridFragment(&grid, &region);
            TS_ASSERT_TEST_GRID_REGION(
                GridBaseType, 
                *grid, 
                *region, 
                cycle);
        }
    }

    void testRun()
    {
        testSim->run();
        referenceSim->run();

        TS_ASSERT_EQUALS(init->maxSteps(), testSim->getStep());

        const Region<2> *region;
        const GridBaseType *grid;
        testSim->getGridFragment(&grid, &region);
        int cycle = maxSteps * TestCell<2>::nanoSteps();

        TS_ASSERT_TEST_GRID_REGION(
            GridBaseType, 
            *grid, 
            *region, 
            cycle);
    }
    
    void checkRunAndWriterInteraction(int everyN)
    {
        MockWriter *expectedCalls = new MockWriter(referenceSim);
        ParallelMockWriter *actualCalls = new ParallelMockWriter(testSim);

        testSim->run();
        referenceSim->run();

        TS_ASSERT_EQUALS(expectedCalls->events(), actualCalls->events()); 
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
        UVec oldPartitions = testSim->partitions;
        UVec newPartitions = toMonoPartitions(oldPartitions);

        testSim->redistributeGrid(oldPartitions, newPartitions);
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
        UVec oldPartitions = testSim->partitions;
        UVec newPartitions = toWeirdoPartitions(oldPartitions);

        testSim->redistributeGrid(oldPartitions, newPartitions);
        testSim->waitForGhostRegions();


        unsigned s = newPartitions[rank    ] - testSim->ghostHeightUpper;
        unsigned e = newPartitions[rank + 1] + testSim->ghostHeightLower;

        int width = init->gridBox().dimensions.x();
        DisplacedGrid<TestCell<2> > expectedStripe(
            CoordBox<2>(Coord<2>(0, s), Coord<2>(width, e - s)));
        init->grid(&expectedStripe);
        Grid<TestCell<2> > actualStripe = *testSim->curStripe->vanillaGrid();
        TSM_ASSERT_EQUALS(actualStripe.diff(*expectedStripe.vanillaGrid()).c_str(), 
                          actualStripe, *expectedStripe.vanillaGrid());
    }

    void checkRunWithDifferentPartitions(UVec newPartitions)
    {
        UVec oldPartitions = testSim->partitions;
        testSim->redistributeGrid(oldPartitions, newPartitions);
        testSim->waitForGhostRegions();

        testSim->run();
        referenceSim->run();

        const Region<2> *region;
        const GridBaseType *grid;
        int cycle = maxSteps * TestCell<2>::nanoSteps();
        testSim->getGridFragment(&grid, &region);

        TS_ASSERT_TEST_GRID_REGION(
            GridBaseType, 
            *grid, 
            *region, 
            cycle);
        TS_ASSERT_EQUALS(testSim->partitions, newPartitions);
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
        UVec oldPartitions = testSim->partitions;
        UVec newPartitions = toWeirdoPartitions(oldPartitions);

        testSim->redistributeGrid(oldPartitions, newPartitions);
        
        TS_ASSERT_EQUALS(testSim->partitions, newPartitions);
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
        LoadBalancer *balancer = rank? 0 : new RandomBalancer;
        StripingSimulator<TestCell<2> > localTestSim(
            new TestInitializer<2>(
                Coord<2>(width, height), maxSteps, firstStep), 
            balancer, 
            balanceEveryN);

        MockWriter *expectedCalls = new MockWriter(referenceSim);
        ParallelMockWriter *actualCalls = new ParallelMockWriter(&localTestSim);

        localTestSim.run();
        referenceSim->run();

        TS_ASSERT_EQUALS(expectedCalls->events(), actualCalls->events()); 
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
            new TestInitializer<2>(),
            balancer, 
            balanceEveryN);
        testSim.run();
    }

    void testEmptyBalancer()
    {
        if (rank == 0) {
            TS_ASSERT_THROWS(
                StripingSimulator<TestCell<2> > s(new TestInitializer<2>(), 0),
                std::invalid_argument);
        } else {
            TS_ASSERT_THROWS(
                StripingSimulator<TestCell<2> > s(new TestInitializer<2>(), 
                                                  new NoOpBalancer),
                std::invalid_argument);
        }
    }

    void test3Dsimple()
    {
        StripingSimulator<TestCell<3> > s(
            new TestInitializer<3>(), 
            rank? 0 : new NoOpBalancer());

        s.run();
    }

    void test3Dadvanced()
    {
        StripingSimulator<TestCell<3> > s(
            new TestInitializer<3>(), 
            rank? 0 : new RandomBalancer());

        s.run();
    }

private:

    // just a boring partition: everything on _one_ node
    UVec toMonoPartitions(UVec oldPartitions) 
    {
        UVec newPartitions(5);
        newPartitions[0] = 0;
        newPartitions[1] = oldPartitions[4];
        newPartitions[2] = oldPartitions[4];
        newPartitions[3] = oldPartitions[4];
        newPartitions[4] = oldPartitions[4];
        return newPartitions;
    }

    // a weird new partition, max. difference to the uniform one
    UVec toWeirdoPartitions(UVec oldPartitions)
    {
        UVec newPartitions(5);
        newPartitions[0] = 0;
        newPartitions[1] = 3;
        newPartitions[2] = 3;
        newPartitions[3] = 10;
        newPartitions[4] = oldPartitions[4];
        // just to be sure the config file hasn't been tainted
        TS_ASSERT(oldPartitions[4] > 10);
        return newPartitions;

    }

    Grid<TestCell<2> > mangledGrid(double foo)
    {
        Grid<TestCell<2> > grid(init->gridBox().dimensions);
        init->grid(&grid);
        for (int y = 0; y < grid.getDimensions().y(); y++) {
            for (int x = 0; x < grid.getDimensions().x(); x++) 
                grid[y][x].testValue = foo + y * grid.getDimensions().y() + x; 
        }
        return grid;
    }    
};

};
