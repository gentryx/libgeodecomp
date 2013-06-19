#include <cxxtest/TestSuite.h>
#include <libgeodecomp/loadbalancer/randombalancer.h>

using namespace LibGeoDecomp;

namespace LibGeoDecomp {

class RandomBalancerTest : public CxxTest::TestSuite
{
public:

    void testEcho()
    {
        RandomBalancer b;
        RandomBalancer::WeightVec loads(4);
        loads[0] = 0;
        loads[1] = 8;
        loads[2] = 15;
        loads[3] = 1;

        RandomBalancer::WeightVec actualA = b.balance(loads, RandomBalancer::LoadVec());
        RandomBalancer::WeightVec actualB = b.balance(loads, RandomBalancer::LoadVec());
        TS_ASSERT_DIFFERS(actualA, actualB);

        TS_ASSERT_EQUALS((unsigned)4,  actualA.size());
        TS_ASSERT_EQUALS((unsigned)4,  actualB.size());
        TS_ASSERT_EQUALS((unsigned)24, actualA.sum());
        TS_ASSERT_EQUALS((unsigned)24, actualB.sum());
    }
};

};
