#include <cxxtest/TestSuite.h>
#include <libgeodecomp/loadbalancer/biasbalancer.h>
#include <libgeodecomp/loadbalancer/mockbalancer.h>

using namespace LibGeoDecomp;

namespace LibGeoDecomp {

class BiasBalancerTest : public CxxTest::TestSuite
{
public:

    void setUp()
    {
        loads = BiasBalancer::WeightVec(3);
        loads[0] = 47;
        loads[1] = 11;
        loads[2] = 9;

        relLoads = BiasBalancer::LoadVec(3);
        relLoads[0] = 0.2;
        relLoads[1] = 0.3;
        relLoads[2] = 0.4;
    }


    void testDestabilization()
    {
        BiasBalancer b(0);
        BiasBalancer::WeightVec expected(3, 0);
        expected[0] = 67;

        TS_ASSERT_EQUALS(expected, b.balance(loads, relLoads));
    }


    void testDeleteBalancer()
    {
        MockBalancer::events = "";
        {
            BiasBalancer(new MockBalancer());
        }
        TS_ASSERT_EQUALS("deleted\n", MockBalancer::events);
    }


    void testInterfaceToOtherBalancer()
    {
        BiasBalancer b(new MockBalancer);
        std::string expected = "balance() " + toString(loads) + " " +
            toString(relLoads) + "\n";

        TS_ASSERT_EQUALS("", MockBalancer::events);
        b.balance(loads, relLoads);
        TS_ASSERT_EQUALS("", MockBalancer::events);
        TS_ASSERT_EQUALS(loads, b.balance(loads, relLoads));
        TS_ASSERT_EQUALS(expected, MockBalancer::events);
        TS_ASSERT_EQUALS(loads, b.balance(loads, relLoads));
        TS_ASSERT_EQUALS(loads, b.balance(loads, relLoads));
        TS_ASSERT_EQUALS(loads, b.balance(loads, relLoads));
    }

private:
    BiasBalancer::WeightVec loads;
    BiasBalancer::LoadVec relLoads;
};

};
