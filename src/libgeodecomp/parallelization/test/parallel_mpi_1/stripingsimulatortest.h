#include <cxxtest/TestSuite.h>

#include <libgeodecomp/io/memorywriter.h>
#include <libgeodecomp/io/mockwriter.h>
#include <libgeodecomp/io/testinitializer.h>
#include <libgeodecomp/loadbalancer/mockbalancer.h>
#include <libgeodecomp/loadbalancer/noopbalancer.h>
#include <libgeodecomp/misc/testhelper.h>
#include <libgeodecomp/parallelization/serialsimulator.h>
#include <libgeodecomp/parallelization/stripingsimulator.h>

using namespace LibGeoDecomp;

namespace LibGeoDecomp {

class BadBalancerSum : public LoadBalancer
{
    virtual NoOpBalancer::WeightVec balance(
        const NoOpBalancer::WeightVec& currentLoads,
        const NoOpBalancer::LoadVec&) {
        NoOpBalancer::WeightVec ret = currentLoads;
        ret[0]++;
        return ret;
    }
};

class BadBalancerNum : public LoadBalancer
{
    virtual NoOpBalancer::WeightVec balance(
        const NoOpBalancer::WeightVec& currentLoads,
        const NoOpBalancer::LoadVec&) {
        NoOpBalancer::WeightVec ret = currentLoads;
        ret.push_back(45);
        return ret;
    }
};

class StripingSimulatorTest : public CxxTest::TestSuite
{
public:
    typedef GridBase<TestCell<2>, 2> GridBaseType;

    void setUp()
    {
        balancer = new NoOpBalancer;
        init = new TestInitializer<TestCell<2> >();
        referenceSim = new SerialSimulator<TestCell<2> >(
            new TestInitializer<TestCell<2> >());
        testSim = new StripingSimulator<TestCell<2> >(
            new TestInitializer<TestCell<2> >(), balancer);
    }

    void tearDown()
    {
        delete referenceSim;
        delete testSim;
        delete init;
    }

    void testBadInit()
    {
        TS_ASSERT_THROWS(StripingSimulator<TestCell<2> > foo(
                             new TestInitializer<TestCell<2> >(), 0, 1),
                         std::invalid_argument);
        TS_ASSERT_THROWS(StripingSimulator<TestCell<2> > foo(
                             new TestInitializer<TestCell<2> >(), 0, 0),
                         std::invalid_argument);
    }

    void testDeleteBalancer()
    {
        MockBalancer::events = "";
        {
            StripingSimulator<TestCell<2> > foo(
                new TestInitializer<TestCell<2> >(), new MockBalancer());
        }
        TS_ASSERT_EQUALS("deleted\n", MockBalancer::events);
    }

    void testPartition()
    {
        int gridWidth = 27;
        int size = 1;
        NoOpBalancer::WeightVec actual = testSim->partition(gridWidth, size);
        NoOpBalancer::WeightVec expected(2);
        expected[0] = 0;
        expected[1] = 27;
        TS_ASSERT_EQUALS(actual, expected);

        size = 2;
        actual = testSim->partition(gridWidth, size);
        expected = NoOpBalancer::WeightVec(3);
        expected[0] = 0;
        expected[1] = 13;
        expected[2] = 27;
        TS_ASSERT_EQUALS(actual, expected);

        size = 3;
        actual = testSim->partition(gridWidth, size);
        expected = NoOpBalancer::WeightVec(4);
        expected[0] = 0;
        expected[1] = 9;
        expected[2] = 18;
        expected[3] = 27;
        TS_ASSERT_EQUALS(actual, expected);

        size = 4;
        actual = testSim->partition(gridWidth, size);
        expected = NoOpBalancer::WeightVec(5);
        expected[0] = 0;
        expected[1] = 6;
        expected[2] = 13;
        expected[3] = 20;
        expected[4] = 27;
        TS_ASSERT_EQUALS(actual, expected);
    }

    void testStep()
    {
        TS_ASSERT_EQUALS(referenceSim->getStep(),
                         testSim->getStep());
        TS_ASSERT(*(referenceSim->getGrid()) == *testSim->curStripe);

        for (int i = 0; i < 50; i++) {
            referenceSim->step();
            testSim->step();
            TS_ASSERT_EQUALS(referenceSim->getStep(),
                             testSim->getStep());
            TS_ASSERT_EQUALS(referenceSim->getGrid()->dimensions(),
                             testSim->curStripe->dimensions());
            TS_ASSERT(*referenceSim->getGrid() == *testSim->curStripe);
            TS_ASSERT_TEST_GRID(GridBaseType, *testSim->curStripe,
                                (i + 1) * APITraits::SelectNanoSteps<TestCell<2> >::VALUE);
        }
    }

    void testStripeWidth()
    {
        TS_ASSERT_EQUALS(testSim->curStripe->getDimensions().x(),
                         init->gridDimensions().x());
    }

    void testRun()
    {
        testSim->run();
        referenceSim->run();
        const GridBase<TestCell<2>, 2> *refGrid = referenceSim->getGrid();

        TS_ASSERT_EQUALS(init->maxSteps(), testSim->getStep());
        TS_ASSERT(*refGrid == *testSim->curStripe);
    }

    void testRunMustResetGridPriorToSimulation()
    {
        testSim->run();
        int cycle1 = testSim->curStripe->get(Coord<2>(4, 4)).cycleCounter;

        testSim->run();
        int cycle2 = testSim->curStripe->get(Coord<2>(4, 4)).cycleCounter;

        TS_ASSERT_EQUALS(cycle1, cycle2);
    }

    void testPartitionsToWorkloadsAndBackAgain()
    {
        NoOpBalancer::WeightVec partitions(6);
        partitions[0] =  0;
        partitions[1] =  1;
        partitions[2] = 11;
        partitions[3] = 11;
        partitions[4] = 20;
        partitions[5] = 90;

        NoOpBalancer::WeightVec workloads(5);
        workloads[0] =  1;
        workloads[1] = 10;
        workloads[2] =  0;
        workloads[3] =  9;
        workloads[4] = 70;

        TS_ASSERT_EQUALS(testSim->partitionsToWorkloads(partitions),
                         workloads);

        TS_ASSERT_EQUALS(testSim->workloadsToPartitions(workloads),
                         partitions);
    }

    void testBadBalancerSum()
    {
        StripingSimulator<TestCell<2> > s(
            new TestInitializer<TestCell<2> >(), new BadBalancerSum());
        TS_ASSERT_THROWS(s.balanceLoad(), std::invalid_argument&);
    }

    void testBadBalancerNum()
    {
        StripingSimulator<TestCell<2> > s(
            new TestInitializer<TestCell<2> >(), new BadBalancerNum());
        TS_ASSERT_THROWS(s.balanceLoad(), std::invalid_argument&);
    }

    void test3D()
    {
        StripingSimulator<TestCell<3> > s(
            new TestInitializer<TestCell<3> >(),
            new NoOpBalancer());

        s.run();
    }

private:
    LoadBalancer *balancer;
    MonolithicSimulator<TestCell<2> > *referenceSim;
    StripingSimulator<TestCell<2> > *testSim;
    Initializer<TestCell<2> > *init;
};

}
