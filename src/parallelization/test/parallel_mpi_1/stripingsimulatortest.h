#include <cxxtest/TestSuite.h>

//fixme: need those?
#include <emmintrin.h>
#include <sys/time.h>

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

class Cell
{
public:
    typedef Topologies::Cube<3>::Topology Topology;

    static inline unsigned nanoSteps() 
    { 
        return 1; 
    }

    inline explicit Cell(const double& v=0) : temp(v)
    {}  

    template<typename COORD_MAP>
    void update(const COORD_MAP& neighborhood, const unsigned& nanoStep)
    {
        temp = (neighborhood[Coord<3>( 0,  0, -1)].temp + 
                neighborhood[Coord<3>( 0, -1,  0)].temp + 
                neighborhood[Coord<3>(-1,  0,  0)].temp + 
                neighborhood[Coord<3>( 0,  0,  0)].temp +
                neighborhood[Coord<3>( 1,  0,  0)].temp + 
                neighborhood[Coord<3>( 0,  1,  0)].temp + 
                neighborhood[Coord<3>( 0,  0,  1)].temp) * (1.0 / 7.0);
    }

    static void update(
        Cell *target, Cell* right, Cell *top, 
        Cell *center, Cell *bottom, Cell *left, 
        const int& length, const unsigned& nanoStep) 
    {
        double factor = 1.0 / 7.0;
        __m128d xFactor, cell1, cell2, cell3, cell4, tmp0, tmp1, tmp2, tmp3, tmp4;
        xFactor = _mm_set_pd(factor, factor);

        tmp0 = _mm_loadu_pd((double*) &center[-1]);

        // for (int start = 0; start < 0; start +=8) {
        for (int start = 0; start < length - 7; start +=8) {
            cell1 = _mm_load_pd((double*) &right[start] + 0);
            cell2 = _mm_load_pd((double*) &right[start] + 2);
            cell3 = _mm_load_pd((double*) &right[start] + 4);
            cell4 = _mm_load_pd((double*) &right[start] + 6);

            tmp1 = _mm_load_pd((double*) &top[start] + 0);
            tmp2 = _mm_load_pd((double*) &top[start] + 2);
            tmp3 = _mm_load_pd((double*) &top[start] + 4);
            tmp4 = _mm_load_pd((double*) &top[start] + 6);

            cell1 = _mm_add_pd(cell1, tmp1);
            cell2 = _mm_add_pd(cell2, tmp2);
            cell3 = _mm_add_pd(cell3, tmp3);
            cell4 = _mm_add_pd(cell4, tmp4);

            tmp1 = _mm_load_pd((double*) &center[start] + 0);
            tmp2 = _mm_load_pd((double*) &center[start] + 2);
            tmp3 = _mm_load_pd((double*) &center[start] + 4);
            tmp4 = _mm_load_pd((double*) &center[start] + 6);

            cell1 = _mm_add_pd(cell1, tmp1);
            cell2 = _mm_add_pd(cell2, tmp2);
            cell3 = _mm_add_pd(cell3, tmp3);
            cell4 = _mm_add_pd(cell4, tmp4);

            tmp1 = _mm_loadu_pd((double*) &bottom[start] + 1);
            tmp2 = _mm_loadu_pd((double*) &bottom[start] + 3);
            tmp3 = _mm_loadu_pd((double*) &bottom[start] + 5);
            tmp4 = _mm_loadu_pd((double*) &bottom[start] + 7);

            cell1 = _mm_add_pd(cell1, tmp0);
            cell2 = _mm_add_pd(cell2, tmp1);
            cell3 = _mm_add_pd(cell3, tmp2);
            cell4 = _mm_add_pd(cell4, tmp3);

            cell1 = _mm_add_pd(cell1, tmp1);
            cell2 = _mm_add_pd(cell2, tmp2);
            cell3 = _mm_add_pd(cell3, tmp3);
            cell4 = _mm_add_pd(cell4, tmp4);

            tmp0 = tmp4;

            tmp1 = _mm_load_pd((double*) &center[start] + 0);
            tmp2 = _mm_load_pd((double*) &center[start] + 2);
            tmp3 = _mm_load_pd((double*) &center[start] + 4);
            tmp4 = _mm_load_pd((double*) &center[start] + 6);

            cell1 = _mm_add_pd(cell1, tmp1);
            cell2 = _mm_add_pd(cell2, tmp2);
            cell3 = _mm_add_pd(cell3, tmp3);
            cell4 = _mm_add_pd(cell4, tmp4);

            tmp1 = _mm_load_pd((double*) &left[start] + 0);
            tmp2 = _mm_load_pd((double*) &left[start] + 2);
            tmp3 = _mm_load_pd((double*) &left[start] + 4);
            tmp4 = _mm_load_pd((double*) &left[start] + 6);

            cell1 = _mm_add_pd(cell1, tmp1);
            cell2 = _mm_add_pd(cell2, tmp2);
            cell3 = _mm_add_pd(cell3, tmp3);
            cell4 = _mm_add_pd(cell4, tmp4);

            cell1 = _mm_mul_pd(cell1, xFactor);
            cell2 = _mm_mul_pd(cell2, xFactor);
            cell3 = _mm_mul_pd(cell3, xFactor);
            cell4 = _mm_mul_pd(cell4, xFactor);

            _mm_store_pd((double*) &target[start] + 0, cell1);
            _mm_store_pd((double*) &target[start] + 2, cell2);
            _mm_store_pd((double*) &target[start] + 4, cell3);
            _mm_store_pd((double*) &target[start] + 6, cell4);
        }
    }

    double temp;
};

template<>
class UpdateFunctor<Cell> : public StreakUpdateFunctor<Cell>
{};

class BadBalancerSum : public LoadBalancer {
    virtual NoOpBalancer::WeightVec balance(const NoOpBalancer::WeightVec& currentLoads, const NoOpBalancer::LoadVec&) {
        NoOpBalancer::WeightVec ret = currentLoads;
        ret[0]++;
        return ret;
    }
};


class BadBalancerNum : public LoadBalancer {
    virtual NoOpBalancer::WeightVec balance(const NoOpBalancer::WeightVec& currentLoads, const NoOpBalancer::LoadVec&) {
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
        init = new TestInitializer<2>();
        referenceSim = new SerialSimulator<TestCell<2> >(
            new TestInitializer<2>());
        testSim = new StripingSimulator<TestCell<2> >(
            new TestInitializer<2>(), balancer);
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
                             new TestInitializer<2>(), 0, 1), 
                         std::invalid_argument);
        TS_ASSERT_THROWS(StripingSimulator<TestCell<2> > foo(
                             new TestInitializer<2>(), 0, 0),
                         std::invalid_argument);
    }

    void testDeleteBalancer()
    {
        MockBalancer::events = "";
        {
            StripingSimulator<TestCell<2> > foo(
                new TestInitializer<2>(), new MockBalancer()); 
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
        const GridBaseType *grid;
        const Region<2> *region;
        
        testSim->getGridFragment(&grid, &region);
        TS_ASSERT_EQUALS(referenceSim->getStep(), 
                         testSim->getStep());
        TS_ASSERT(*(referenceSim->getGrid()) == *grid);

        for (int i = 0; i < 50; i++) {
            referenceSim->step();
            testSim->step();
            testSim->getGridFragment(&grid, &region);
            TS_ASSERT_EQUALS(referenceSim->getStep(), 
                             testSim->getStep());
            TS_ASSERT_EQUALS(referenceSim->getGrid()->getDimensions(), 
                             grid->boundingBox().dimensions);
            TS_ASSERT(*referenceSim->getGrid() == *grid);
            TS_ASSERT_TEST_GRID(GridBaseType, *grid, 
                                (i + 1) * TestCell<2>::nanoSteps());
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
        const GridBaseType *grid;
        const Region<2> *region;
        testSim->getGridFragment(&grid, &region);
        Grid<TestCell<2> > refGrid = *referenceSim->getGrid();

        TS_ASSERT_EQUALS(init->maxSteps(), testSim->getStep());
        TS_ASSERT(refGrid == *grid);
    }

    void testRunMustResetGridPriorToSimulation()
    {
        const Region<2> *r;
        const GridBaseType *g;

        testSim->run();
        testSim->getGridFragment(&g, &r);
        int cycle1 = g->at(Coord<2>(4, 4)).cycleCounter;

        testSim->run();
        testSim->getGridFragment(&g, &r);
        int cycle2 = g->at(Coord<2>(4, 4)).cycleCounter;

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
            new TestInitializer<2>(), new BadBalancerSum);
        TS_ASSERT_THROWS(s.balanceLoad(), std::invalid_argument);
    }

    void testBadBalancerNum()
    {    
        StripingSimulator<TestCell<2> > s(
            new TestInitializer<2>(), new BadBalancerNum);
        TS_ASSERT_THROWS(s.balanceLoad(), std::invalid_argument);
    }

    void test3D()
    {
        StripingSimulator<TestCell<3> > s(
            new TestInitializer<3>(), 
            new NoOpBalancer());

        s.run();
    }















    static void kernelLib(
        double* right, double *top, 
        double *center, double *bottom, double *left, 
        double *target, 
        const int& length) 
    {
        double factor = 1.0 / 7.0;
        __m128d xFactor, cell1, cell2, cell3, cell4, tmp0, tmp1, tmp2, tmp3, tmp4;
        xFactor = _mm_set_pd(factor, factor);

        tmp0 = _mm_loadu_pd((double*) &center[-1]);

        for (int start = 0; start < length; start +=8) {
            cell1 = _mm_load_pd((double*) &right[start] + 0);
            cell2 = _mm_load_pd((double*) &right[start] + 2);
            cell3 = _mm_load_pd((double*) &right[start] + 4);
            cell4 = _mm_load_pd((double*) &right[start] + 6);

            tmp1 = _mm_load_pd((double*) &top[start] + 0);
            tmp2 = _mm_load_pd((double*) &top[start] + 2);
            tmp3 = _mm_load_pd((double*) &top[start] + 4);
            tmp4 = _mm_load_pd((double*) &top[start] + 6);

            cell1 = _mm_add_pd(cell1, tmp1);
            cell2 = _mm_add_pd(cell2, tmp2);
            cell3 = _mm_add_pd(cell3, tmp3);
            cell4 = _mm_add_pd(cell4, tmp4);

            tmp1 = _mm_load_pd((double*) &center[start] + 0);
            tmp2 = _mm_load_pd((double*) &center[start] + 2);
            tmp3 = _mm_load_pd((double*) &center[start] + 4);
            tmp4 = _mm_load_pd((double*) &center[start] + 6);

            cell1 = _mm_add_pd(cell1, tmp1);
            cell2 = _mm_add_pd(cell2, tmp2);
            cell3 = _mm_add_pd(cell3, tmp3);
            cell4 = _mm_add_pd(cell4, tmp4);

            tmp1 = _mm_loadu_pd((double*) &bottom[start] + 1);
            tmp2 = _mm_loadu_pd((double*) &bottom[start] + 3);
            tmp3 = _mm_loadu_pd((double*) &bottom[start] + 5);
            tmp4 = _mm_loadu_pd((double*) &bottom[start] + 7);

            cell1 = _mm_add_pd(cell1, tmp0);
            cell2 = _mm_add_pd(cell2, tmp1);
            cell3 = _mm_add_pd(cell3, tmp2);
            cell4 = _mm_add_pd(cell4, tmp3);

            cell1 = _mm_add_pd(cell1, tmp1);
            cell2 = _mm_add_pd(cell2, tmp2);
            cell3 = _mm_add_pd(cell3, tmp3);
            cell4 = _mm_add_pd(cell4, tmp4);

            tmp0 = tmp4;

            tmp1 = _mm_load_pd((double*) &center[start] + 0);
            tmp2 = _mm_load_pd((double*) &center[start] + 2);
            tmp3 = _mm_load_pd((double*) &center[start] + 4);
            tmp4 = _mm_load_pd((double*) &center[start] + 6);

            cell1 = _mm_add_pd(cell1, tmp1);
            cell2 = _mm_add_pd(cell2, tmp2);
            cell3 = _mm_add_pd(cell3, tmp3);
            cell4 = _mm_add_pd(cell4, tmp4);

            tmp1 = _mm_load_pd((double*) &left[start] + 0);
            tmp2 = _mm_load_pd((double*) &left[start] + 2);
            tmp3 = _mm_load_pd((double*) &left[start] + 4);
            tmp4 = _mm_load_pd((double*) &left[start] + 6);

            cell1 = _mm_add_pd(cell1, tmp1);
            cell2 = _mm_add_pd(cell2, tmp2);
            cell3 = _mm_add_pd(cell3, tmp3);
            cell4 = _mm_add_pd(cell4, tmp4);

            cell1 = _mm_mul_pd(cell1, xFactor);
            cell2 = _mm_mul_pd(cell2, xFactor);
            cell3 = _mm_mul_pd(cell3, xFactor);
            cell4 = _mm_mul_pd(cell4, xFactor);

            _mm_store_pd((double*) &target[start] + 0, cell1);
            _mm_store_pd((double*) &target[start] + 2, cell2);
            _mm_store_pd((double*) &target[start] + 4, cell3);
            _mm_store_pd((double*) &target[start] + 6, cell4);
        }
    }

    void kernel(
        double *back,
        double *up,
        double *same,
        double *down,
        double *front,
        double *store,
        const int& n)
    {
        // kernelSimple(back, up, same, down, front, store, n);
        // kernelSSEBack(back, up, same, down, front, store, n);
        // kernelSSEForward(back, up, same, down, front, store, n);
        // kernelSSEForwardWithMid(back, up, same, down, front, store, n);
        kernelLib(back, up, same, down, front, store, n);
    }

    typedef DisplacedGrid<Cell, Topologies::Cube<3>::Topology> Matrix;

    long long getUTtime()
    {
        timeval t;
        gettimeofday(&t, 0);
        return (long long)t.tv_sec * 1000000 + t.tv_usec;
    }

    void initi(Matrix *target, int n)
    {
        for (int z = 0; z < n; ++z) 
            for (int y = 0; y < n; ++y) 
                for (int x = 0; x < n; ++x) 
                    (*target)[Coord<3>(x, y, z)].temp = x * y * z;
    }

    Region<3> region;

    void update(Matrix *source, Matrix *target, int n)
    {
        for (Region<3>::StreakIterator i = region.beginStreak(); 
             i != region.endStreak(); 
             ++i) {
            UpdateFunctor<Cell>()(
                *i,
                source,
                target,
                0);
        }
    }

    void evaluate(int n, long repeats, long long uTime)
    {
        double seconds = 1.0 * uTime / 1000 / 1000;
        double lups = (n - 2) * (n - 2) * (n - 2) * repeats / seconds;
        double glups = lups  / 1000 / 1000 / 1000;
        double gflops = 6.0 * glups;
        std::cout << n << " " << seconds << " " << glups << " " << gflops << "\n";
    }

    void benchmark(int n, long repeats)
    {
        Matrix *mat1 = new Matrix(CoordBox<3>(Coord<3>(), Coord<3>(n, n, n)));
        Matrix *mat2 = new Matrix(CoordBox<3>(Coord<3>(), Coord<3>(n, n, n)));
        initi(mat1, n);

        region.clear();
        for (int z = 1; z < n - 1; ++z) 
            for (int y = 1; y < n - 1; ++y) 
                region << Streak<3>(Coord<3>(0, y, z), 256);

        long long tStart = getUTtime();

        for (long r = 0; r < repeats; ++r) {
            update(mat1, mat2, n);
            std::swap(mat1, mat2);
        }
    
        long long tEnd = getUTtime();
        long long delta = tEnd - tStart;
        std::cout << "delta: " << delta << "\n";

        delete mat1;
        delete mat2;

        evaluate(n, repeats, tEnd - tStart);
    }


    void testPerf()
    {
        // fixme: add performance test targets
        benchmark(256, 100);
    }

private:
    LoadBalancer *balancer;
    MonolithicSimulator<TestCell<2> > *referenceSim;
    StripingSimulator<TestCell<2> > *testSim;
    Initializer<TestCell<2> > *init;
};

};
