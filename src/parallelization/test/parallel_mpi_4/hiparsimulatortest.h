#include <boost/shared_ptr.hpp>
#include <cxxtest/TestSuite.h>
#include <sstream>

#include "../../../io/mockwriter.h"
#include "../../../io/parallelmemorywriter.h"
#include "../../../io/testinitializer.h"
#include "../../../misc/testcell.h"
#include "../../../misc/testhelper.h"
#include "../../hiparsimulator.h"
#include "../../hiparsimulator/partitions/zcurvepartition.h"

using namespace LibGeoDecomp; 
using namespace HiParSimulator; 

namespace LibGeoDecomp {
namespace HiParSimulator {

class HiParSimulatorTest : public CxxTest::TestSuite
{
public:
    typedef HiParSimulator<TestCell<2>, ZCurvePartition<2> > SimulatorType;
    typedef ParallelMemoryWriter<TestCell<2> > MemoryWriterType;

    void setUp()
    {
        int width = 131;
        int height = 241;
        dim = Coord<2>(width, height);
        maxSteps = 1500;
        firstStep = 20;
        firstCycle = firstStep * TestCell<2>::nanoSteps();
        TestInitializer<2> *init = new TestInitializer<2>(
            dim, maxSteps, firstStep);

        outputPeriod = 4;
        loadBalancingPeriod = 31;
        ghostZoneWidth = 10;
        s.reset(new SimulatorType(
                    init, 0, loadBalancingPeriod, ghostZoneWidth));
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
        s->step();
        s->step();
        s->step();

        std::stringstream expectedEvents;
        expectedEvents << "initialized()\ninitialized()\n";
        for (int t = 21; t < 25; t += 1) {
            expectedEvents << "stepFinished(step=" << t << ")\n"
                           << "stepFinished(step=" << t << ")\n";
        }
        TS_ASSERT_EQUALS(expectedEvents.str(), mockWriter->events());

        for (int t = 20; t < 25; t += outputPeriod) {
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
    unsigned firstCycle;
    unsigned outputPeriod;
    unsigned loadBalancingPeriod;
    unsigned ghostZoneWidth;
    MockWriter *mockWriter;
    MemoryWriterType *memoryWriter;
};

};
};
