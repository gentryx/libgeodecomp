#include <libgeodecomp/io/mockinitializer.h>
#include <libgeodecomp/io/mockwriter.h>
#include <libgeodecomp/misc/testcell.h>
#include <libgeodecomp/parallelization/mocksimulator.h>
#include <libgeodecomp/parallelization/simulator.h>

#include <cxxtest/TestSuite.h>
#include <boost/shared_ptr.hpp>

using namespace LibGeoDecomp;

namespace LibGeoDecomp {

class SimulatorTest : public CxxTest::TestSuite
{
public:
    void testDestruction()
    {
        MockInitializer::events = "";
        MockSimulator::events = "";
        boost::shared_ptr<MockWriter<>::EventsStore> events(new MockWriter<>::EventsStore);
        {
            MockSimulator sim(new MockInitializer);
            sim.addWriter(new MockWriter<>(events));
            sim.addWriter(new MockWriter<>(events));
            sim.addWriter(new MockWriter<>(events));
        }
        TS_ASSERT_EQUALS("created, configString: ''\ndeleted\n", MockInitializer::events);
        TS_ASSERT_EQUALS("deleted\n", MockSimulator::events);
        TS_ASSERT_EQUALS(3, events->end() - events->begin());
        TS_ASSERT_EQUALS(-1, (*events)[0].step);
        TS_ASSERT_EQUALS(-1, (*events)[1].step);
        TS_ASSERT_EQUALS(-1, (*events)[2].step);
    }

};

}
