#include <sstream>
#include <cxxtest/TestSuite.h>
#include <libgeodecomp/loadbalancer/tracingbalancer.h>
#include <libgeodecomp/loadbalancer/mockbalancer.h>

using namespace LibGeoDecomp; 

namespace LibGeoDecomp {

class TracingBalancerTest : public CxxTest::TestSuite
{
public:

    void setUp()
    {
        _loads = TracingBalancer::WeightVec(3);
        _loads[0] = 47;
        _loads[1] = 11;
        _loads[2] = 9;

        _relLoads = TracingBalancer::LoadVec(3);
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
            "  weights: " + _loads.toString() + "\n" +
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
    TracingBalancer::WeightVec _loads;
    TracingBalancer::LoadVec _relLoads;
};

};
