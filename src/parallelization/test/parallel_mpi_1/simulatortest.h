#include <cxxtest/TestSuite.h>
#include <boost/shared_ptr.hpp>
#include <libgeodecomp/misc/testcell.h>
#include <libgeodecomp/io/mockinitializer.h>
#include <libgeodecomp/io/mockwriter.h>
#include <libgeodecomp/parallelization/mocksimulator.h>
#include <libgeodecomp/parallelization/simulator.h>

using namespace LibGeoDecomp;

namespace LibGeoDecomp {

class SimulatorTest : public CxxTest::TestSuite
{
public:
    void testDestruction()
    {
        MockInitializer::events = "";
        MockSimulator::events = "";
        MockWriter::staticEvents = "";
        {
            MockSimulator sim(new MockInitializer);
            sim.addWriter(new MockWriter());
            sim.addWriter(new MockWriter());
            sim.addWriter(new MockWriter());
        }
        TS_ASSERT_EQUALS("created, configString: ''\ndeleted\n", MockInitializer::events);
        TS_ASSERT_EQUALS("deleted\n", MockSimulator::events);
        TS_ASSERT_EQUALS("deleted\ndeleted\ndeleted\n", MockWriter::staticEvents);
    }

};

}
