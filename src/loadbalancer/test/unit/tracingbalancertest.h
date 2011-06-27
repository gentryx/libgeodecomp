#include <sstream>
#include <cxxtest/TestSuite.h>
#include "../../tracingbalancer.h"
#include "../../mockbalancer.h"

using namespace LibGeoDecomp; 

namespace LibGeoDecomp {

class TracingBalancerTest : public CxxTest::TestSuite
{
public:

    void setUp()
    {
        _loads = UVec(3);
        _loads[0] = 47;
        _loads[1] = 11;
        _loads[2] = 9;

        _relLoads = DVec(3);
        _relLoads[0] = 0.2;
        _relLoads[1] = 0.3;
        _relLoads[2] = 0.4;       
    }


    void testInterfaceToOtherBalancer()
    {        
        std::ostringstream output;
        TracingBalancer b(new MockBalancer, output);
        std::string expected = "balance() " + _loads.toString() + " " + 
            _relLoads.toString() + "\n";
        std::string longExpected = expected + expected + expected + expected;
        
        TS_ASSERT_EQUALS("", MockBalancer::events);
        TS_ASSERT_EQUALS(_loads, b.balance(_loads, _relLoads));
        TS_ASSERT_EQUALS(expected, MockBalancer::events);               
        TS_ASSERT_EQUALS(_loads, b.balance(_loads, _relLoads));
        TS_ASSERT_EQUALS(_loads, b.balance(_loads, _relLoads));
        TS_ASSERT_EQUALS(_loads, b.balance(_loads, _relLoads));
        TS_ASSERT_EQUALS(longExpected, MockBalancer::events);
    }


    void testDeleteBalancer()
    {
        MockBalancer::events = "";
        {
            TracingBalancer(new MockBalancer()); 
        }
        TS_ASSERT_EQUALS("deleted\n", MockBalancer::events);
    }


    void testTrace()
    {        
        std::ostringstream output;
        TracingBalancer b(new MockBalancer, output);
        std::string expected = std::string() + 
            "TracingBalancer::balance()\n" + 
            "  currentLoads: " + _loads.toString() + "\n" +
            "  relativeLoads: " + _relLoads.toString() + "\n";
        std::string longExpected = expected + expected + expected + expected;
        
        TS_ASSERT_EQUALS("", MockBalancer::events);
        TS_ASSERT_EQUALS(_loads, b.balance(_loads, _relLoads));
        TS_ASSERT_EQUALS(expected, output.str());
        TS_ASSERT_EQUALS(_loads, b.balance(_loads, _relLoads));
        TS_ASSERT_EQUALS(_loads, b.balance(_loads, _relLoads));
        TS_ASSERT_EQUALS(_loads, b.balance(_loads, _relLoads));
        TS_ASSERT_EQUALS(longExpected, output.str());
    }


private:
    UVec _loads;
    DVec _relLoads;
};

};
