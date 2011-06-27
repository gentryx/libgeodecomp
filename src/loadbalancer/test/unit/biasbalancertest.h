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
        _loads = UVec(3);
        _loads[0] = 47;
        _loads[1] = 11;
        _loads[2] = 9;

        _relLoads = DVec(3);
        _relLoads[0] = 0.2;
        _relLoads[1] = 0.3;
        _relLoads[2] = 0.4;       
    }


    void testDestabilization()
    {
        BiasBalancer b(0);
        UVec expected(3, 0);
        expected[0] = 67;

        TS_ASSERT_EQUALS(expected, b.balance(_loads, _relLoads));
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
        std::string expected = "balance() " + _loads.toString() + " " + 
            _relLoads.toString() + "\n";
        
        TS_ASSERT_EQUALS("", MockBalancer::events);
        b.balance(_loads, _relLoads);
        TS_ASSERT_EQUALS("", MockBalancer::events);
        TS_ASSERT_EQUALS(_loads, b.balance(_loads, _relLoads));
        TS_ASSERT_EQUALS(expected, MockBalancer::events);               
        TS_ASSERT_EQUALS(_loads, b.balance(_loads, _relLoads));
        TS_ASSERT_EQUALS(_loads, b.balance(_loads, _relLoads));
        TS_ASSERT_EQUALS(_loads, b.balance(_loads, _relLoads));
    }

private:
    UVec _loads;
    DVec _relLoads;
};

};
