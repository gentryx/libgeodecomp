#include <boost/assign/std/vector.hpp>
#include <boost/shared_ptr.hpp>
#include <cxxtest/TestSuite.h>

#include <libgeodecomp/io/mockwriter.h>
#include <libgeodecomp/io/parallelmemorywriter.h>
#include <libgeodecomp/io/testinitializer.h>
#include <libgeodecomp/misc/testcell.h>
#include <libgeodecomp/misc/testhelper.h>
#include <libgeodecomp/parallelization/hiparsimulator.h>
#include <libgeodecomp/parallelization/hiparsimulator/partitions/stripingpartition.h>

using namespace boost::assign; 
using namespace LibGeoDecomp; 
using namespace HiParSimulator; 

namespace LibGeoDecomp {
namespace HiParSimulator {

class HiParSimulatorTest : public CxxTest::TestSuite
{
public:
    // fixme: rename types a la "MyFoobar" to "FoobarType"
    typedef HiParSimulator<TestCell<2>, StripingPartition<2> > SimulatorType;
    typedef ParallelMemoryWriter<TestCell<2> > MemoryWriterType;

    void setUp()
    {
        int width = 11;
        int height = 21;
        dim = Coord<2>(width, height);
        maxSteps = 200;
        firstStep = 20;
        TestInitializer<2> *init = new TestInitializer<2>(
            dim, maxSteps, firstStep);
        
        outputPeriod = 1;
        loadBalancingPeriod = 31;
        ghostzZoneWidth = 10;
        s.reset(new SimulatorType(
                    init, 0, loadBalancingPeriod, ghostzZoneWidth));
        mockWriter = new MockWriter(&*s);
        memoryWriter = new MemoryWriterType(&*s, outputPeriod);
    }

    void tearDown()
    {
        s.reset();        
    }

    void testStep()
    {
        s->step();

        std::string expectedEvents = "initialized()\ninitialized()\nstepFinished(step=21)\nstepFinished(step=21)\n";

        TS_ASSERT_EQUALS(expectedEvents, mockWriter->events());
        
        SuperVector<unsigned> actualSteps;
        SuperVector<unsigned> expectedSteps;
        expectedSteps += 20, 21;

        MemoryWriterType::GridMap grids = memoryWriter->getGrids();
        for (MemoryWriterType::GridMap::iterator iter = grids.begin(); iter != grids.end(); ++iter) {
            actualSteps << iter->first;
            int globalNanoStep = iter->first * TestCell<2>::nanoSteps();
            TS_ASSERT_TEST_GRID(
                MemoryWriterType::GridType, iter->second, globalNanoStep);
        }

        TS_ASSERT_EQUALS(expectedSteps, actualSteps);
    }

    void testRun()
    {
        s->run();

        std::string expectedEvents = "initialized()\ninitialized()\n";
        for (int t = 21; t < 200; ++t) {
            std::stringstream addend;
            addend << "stepFinished(step=" << t << ")\n";
            expectedEvents += addend.str() + addend.str();
        }
        expectedEvents += "allDone()\nallDone()\n";
        TS_ASSERT_EQUALS(expectedEvents, mockWriter->events());

        for (int t = 20; t <= 200; ++t) {
            int globalNanoStep = t * TestCell<2>::nanoSteps();
            MemoryWriterType::GridMap grids = memoryWriter->getGrids();
            TS_ASSERT_TEST_GRID(
                MemoryWriterType::GridType, 
                grids[t], 
                globalNanoStep);
            TS_ASSERT_EQUALS(dim, grids[t].getDimensions());
        }
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
