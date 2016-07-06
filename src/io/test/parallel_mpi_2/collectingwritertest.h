#include <libgeodecomp/io/collectingwriter.h>
#include <libgeodecomp/io/testinitializer.h>
#include <libgeodecomp/io/memorywriter.h>
#include <libgeodecomp/loadbalancer/noopbalancer.h>
#include <libgeodecomp/loadbalancer/randombalancer.h>
#include <libgeodecomp/misc/testhelper.h>
#include <libgeodecomp/parallelization/stripingsimulator.h>

#include <boost/shared_ptr.hpp>

using namespace LibGeoDecomp;

namespace LibGeoDecomp {

class CollectingWriterTest : public CxxTest::TestSuite
{
public:
    void setUp()
    {
        TestInitializer<TestCell<3> > *init = new TestInitializer<TestCell<3> >();

        LoadBalancer *balancer = MPILayer().rank()? 0 : new RandomBalancer;
        sim.reset(new StripingSimulator<TestCell<3> >(init, balancer));

        writer = 0;
        if (MPILayer().rank() == 0) {
            writer = new MemoryWriter<TestCell<3> >(3);
        }

        sim->addWriter(new CollectingWriter<TestCell<3> >(writer, 0));
    }

    void tearDown()
    {
        sim.reset();
    }

    void testBasic()
    {
        sim->run();

        if (MPILayer().rank() == 0) {
            int size = writer->getGrids().size();
            unsigned cycle = 0;

            for (int i = 0; i < (size - 1); ++i) {
                cycle = APITraits::SelectNanoSteps<TestCell<3> >::VALUE * i * 3;

                TS_ASSERT_TEST_GRID(MemoryWriter<TestCell<3> >::GridType, writer->getGrids()[i], cycle);
            }

            cycle = APITraits::SelectNanoSteps<TestCell<3> >::VALUE * TestInitializer<TestCell<3> >().maxSteps();

            // check the last grid with the same cycle counter as the
            // simulator will notify the writer of this time step
            // twice (WRITER_STEP_FINISHED and WRITER_ALL_DONE)
            TS_ASSERT_TEST_GRID(MemoryWriter<TestCell<3> >::GridType, writer->getGrids()[size - 1], cycle);
        }
    }

    void testSoA()
    {
        TestInitializer<TestCellSoA> *init = new TestInitializer<TestCellSoA>();

        LoadBalancer *balancer = MPILayer().rank()? 0 : new RandomBalancer;
        StripingSimulator<TestCellSoA> sim(init, balancer);

        MemoryWriter<TestCellSoA> *writer = 0;
        if (MPILayer().rank() == 0) {
            writer = new MemoryWriter<TestCellSoA>(3);
        }

        sim.addWriter(new CollectingWriter<TestCellSoA>(writer, 0));
        sim.run();

        if (MPILayer().rank() == 0) {
            int size = writer->getGrids().size();
            unsigned cycle = 0;

            for (int i = 0; i < (size - 1); ++i) {
                cycle = APITraits::SelectNanoSteps<TestCellSoA>::VALUE * i * 3;

                TS_ASSERT_TEST_GRID(MemoryWriter<TestCellSoA>::GridType, writer->getGrids()[i], cycle);
            }

            cycle = APITraits::SelectNanoSteps<TestCellSoA>::VALUE * TestInitializer<TestCellSoA>().maxSteps();

            // check the last grid with the same cycle counter as the
            // simulator will notify the writer of this time step
            // twice (WRITER_STEP_FINISHED and WRITER_ALL_DONE)
            TS_ASSERT_TEST_GRID(MemoryWriter<TestCellSoA>::GridType, writer->getGrids()[size - 1], cycle);
        }
    }

private:
    boost::shared_ptr<StripingSimulator<TestCell<3> > > sim;
    MemoryWriter<TestCell<3> > *writer;
};

}
