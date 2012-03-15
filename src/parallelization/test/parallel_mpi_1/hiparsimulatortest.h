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
    // fixme: rename types a la "MyFoobar" to "FoobarType"
    typedef HiParSimulator<TestCell<2>, StripingPartition<2> > MySimulator;

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
        s.reset(new MySimulator(
                    init, 0, loadBalancingPeriod, ghostzZoneWidth));
        mockWriter = new MockWriter(&(*s));
    }

    void tearDown()
    {
        s.reset();        
    }

    void testStep()
    {
        // const MySimulator::GridType *grid;
        // const Region<2> *validRegion;

        // s->getGridFragment(&grid, &validRegion);
        // std::cout << "got " << (*grid)[Coord<2>(5, 5)] << "\n";

        s->step();
        // s->step();
        // s->step();
        std::cout << "-----------got events : " << mockWriter->events() << "\n";
    }

    void testCallsToWriter()
    {
    //     std::cout << "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX\n";
        // s->run();
        std::cout << "-----------got events : " << mockWriter->events() << "\n";
    }

private:
    boost::shared_ptr<MySimulator > s;
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
