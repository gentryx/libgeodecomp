#include <cxxtest/TestSuite.h>
#include "../../noopbalancer.h"

using namespace LibGeoDecomp; 

namespace LibGeoDecomp {

class NoOpBalancerTest : public CxxTest::TestSuite
{
public:

    void testEcho()
    {
        NoOpBalancer b;
        NoOpBalancer::WeightVec loads(3);
        loads[0] = 47;
        loads[1] = 11;
        loads[2] = 9;

        TS_ASSERT_EQUALS(loads, b.balance(loads, NoOpBalancer::LoadVec()));
    }
};

}
