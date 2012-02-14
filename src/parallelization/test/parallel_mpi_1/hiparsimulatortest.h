#include <boost/shared_ptr.hpp>
#include <cxxtest/TestSuite.h>

#include <libgeodecomp/io/mockwriter.h>
#include <libgeodecomp/io/testinitializer.h>
#include <libgeodecomp/misc/testcell.h>
#include <libgeodecomp/misc/testhelper.h>
#include <libgeodecomp/parallelization/hiparsimulator.h>
#include <libgeodecomp/parallelization/hiparsimulator/partitions/stripingpartition.h>

using namespace LibGeoDecomp; 
using namespace HiParSimulator; 

namespace LibGeoDecomp {
namespace HiParSimulator {

class HiParSimulatorTest : public CxxTest::TestSuite
{
public:

    void setUp()
    {
        width = 11;
        height = 21;
        maxSteps = 200;
        firstStep = 20;
        TestInitializer<2> *init = new TestInitializer<2>(
            Coord<2>(width, height), maxSteps, firstStep);
        
        outputPeriod = 17;
        loadBalancingPeriod = 31;
        ghostzZoneWidth = 10;
        s.reset(new HiParSimulator<TestCell<2>, StripingPartition<2> >(
                    init, 0, outputPeriod, loadBalancingPeriod, ghostzZoneWidth));
        // fixme
        // mockWriter = new MockWriter(&(*s));
    }

    void tearDown()
    {
        s.reset();        
    }

    void testResetEvents()
    {
        // fixme
    //     s->resetEvents();
    //     EventMap expectedEvents;
    //     for (int repeat = 2; repeat < 6; ++repeat)
    //         expectedEvents[repeat * outputPeriod * 
    //                        TestCell<2>::nanoSteps()].insert(OUTPUT);
    //     for (int repeat = 1; repeat < 5; ++repeat)
    //         expectedEvents[repeat * loadBalancingPeriod * 
    //                        TestCell<2>::nanoSteps()].insert(LOAD_BALANCING);
    //     expectedEvents[maxSteps * TestCell<2>::nanoSteps()].insert(
    //         SIMULATION_END);
    //     TS_ASSERT_EQUALS(s->events, expectedEvents);
    // }


    // void testResetSimulation()
    // {
    //     // fixme:
    //     // TS_ASSERT_TEST_GRID(
    //     //     Grid<TestCell<2> >, *s->getGrid(),   
    //     //     firstStep * TestCell<2>::nanoSteps() + firstNanoStep);
    //     TS_ASSERT_EQUALS(
    //         s->nanoStepCounter, 
    //         firstStep * TestCell<2>::nanoSteps());
    // }

    // // fixme: check which object (hiparsimulator, steppers,
    // // partitionmanager, loadbalancer...) owns what and who's
    // // responsible for deallocation
    // void testNanoStepEventHandling()
    // {
    //     // 2 (output + loadbalancing) * 4 (eventRepetitionHorizon) 
    //     // + 1 (end of simulation)
    //     TS_ASSERT_EQUALS(s->events.size(), 2 * 4 + 1);
    //     s->nanoStep((maxSteps - 100) * TestCell<2>::nanoSteps());
    //     TS_ASSERT_EQUALS(s->events.size(), 2 * 4 + 1);

    //     std::ostringstream expectedIO;
    //     expectedIO << "initialized()\n";
    //     // fixme: TS_ASSERT_EQUALS(expectedIO.str(), mockWriter->events());
    }

private:
    boost::shared_ptr<HiParSimulator<TestCell<2>, StripingPartition<2> > > s;
    unsigned width;
    unsigned height;
    unsigned maxSteps;
    unsigned firstStep;
    unsigned outputPeriod;
    unsigned loadBalancingPeriod;
    unsigned ghostzZoneWidth;
    MockWriter *mockWriter;
};

};
};
