#include <boost/shared_ptr.hpp>

#include <libgeodecomp/io/collectingwriter.h>
#include <libgeodecomp/io/testinitializer.h>
#include <libgeodecomp/io/memorywriter.h>
#include <libgeodecomp/loadbalancer/randombalancer.h>
#include <libgeodecomp/misc/testhelper.h>
#include <libgeodecomp/parallelization/stripingsimulator.h>

using namespace LibGeoDecomp; 

namespace LibGeoDecomp {

class CollectingWriterTest : public CxxTest::TestSuite 
{
public:
    void setUp()
    {  
        TestInitializer<TestCell<3> > *init = getInit();
        Coord<3> dimensions(init->gridDimensions());

        LoadBalancer *balancer = MPILayer().rank()? 0 : new RandomBalancer;
        sim.reset(new StripingSimulator<TestCell<3> >(init, balancer));
        
        if (MPILayer().rank() == 0) {
            writer = new MemoryWriter<TestCell<3> >(1);
        } else {
            writer = 0;
        }

        sim->addWriter(new CollectingWriter<TestCell<3> >(writer)); 
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
                cycle = TestCell<3>::nanoSteps() * i;
                TS_ASSERT_TEST_GRID(MemoryWriter<TestCell<3> >::GridType, writer->getGrids()[i], cycle);
            }

            // check the last grid with the same cycle counter as the
            // simulator will notify the writer of this time step
            // twice (WRITER_STEP_FINISHED and WRITER_ALL_DONE)
            TS_ASSERT_TEST_GRID(MemoryWriter<TestCell<3> >::GridType, writer->getGrids()[size - 1], cycle);
        }
    }

private:
    boost::shared_ptr<StripingSimulator<TestCell<3> > > sim;
    MemoryWriter<TestCell<3> > *writer;

    TestInitializer<TestCell<3> > *getInit()
    {
        return new TestInitializer<TestCell<3> >();
    }
};

}
