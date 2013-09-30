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
        loads = TracingBalancer::WeightVec(3);
        loads[0] = 47;
        loads[1] = 11;
        loads[2] = 9;

        relLoads = TracingBalancer::LoadVec(3);
        relLoads[0] = 0.2;
        relLoads[1] = 0.3;
        relLoads[2] = 0.4;
    }


    void testInterfaceToOtherBalancer()
    {
        std::ostringstream output;
        TracingBalancer b(new MockBalancer, output);
        std::string expected = "balance() " + toString(loads) + " " +
            toString(relLoads) + "\n";
        std::string longExpected = expected + expected + expected + expected;

        TS_ASSERT_EQUALS("", MockBalancer::events);
        TS_ASSERT_EQUALS(loads, b.balance(loads, relLoads));
        TS_ASSERT_EQUALS(expected, MockBalancer::events);
        TS_ASSERT_EQUALS(loads, b.balance(loads, relLoads));
        TS_ASSERT_EQUALS(loads, b.balance(loads, relLoads));
        TS_ASSERT_EQUALS(loads, b.balance(loads, relLoads));
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
            "  weights: " + toString(loads) + "\n" +
            "  relativeLoads: " + toString(relLoads) + "\n";
        std::string longExpected = expected + expected + expected + expected;

        TS_ASSERT_EQUALS("", MockBalancer::events);
        TS_ASSERT_EQUALS(loads, b.balance(loads, relLoads));
        TS_ASSERT_EQUALS(expected, output.str());
        TS_ASSERT_EQUALS(loads, b.balance(loads, relLoads));
        TS_ASSERT_EQUALS(loads, b.balance(loads, relLoads));
        TS_ASSERT_EQUALS(loads, b.balance(loads, relLoads));
        TS_ASSERT_EQUALS(longExpected, output.str());
    }


private:
    TracingBalancer::WeightVec loads;
    TracingBalancer::LoadVec relLoads;
};

};
