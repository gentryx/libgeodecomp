#include <cxxtest/TestSuite.h>
#include <boost/shared_ptr.hpp>
#include "../../../misc/testcell.h"
#include "../../../io/mockinitializer.h"
#include "../../../io/mockwriter.h"
#include "../../mocksimulator.h"
#include "../../simulator.h"

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
            new MockWriter(&sim);
            new MockWriter(&sim);
            new MockWriter(&sim);
        }
        TS_ASSERT_EQUALS("created, configString: ''\ndeleted\n", MockInitializer::events);
        TS_ASSERT_EQUALS("deleted\n", MockSimulator::events);
        TS_ASSERT_EQUALS("deleted\ndeleted\ndeleted\n", MockWriter::staticEvents);        
    }

};

};
