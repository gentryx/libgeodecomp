#include <cxxtest/TestSuite.h>
#include <hpx/hpx.hpp>
#include <hpx/hpx_init.hpp>
#include <hpx/runtime/serialization/set.hpp>
#include <libgeodecomp/communication/hpxserialization.h>
#include <libgeodecomp/loadbalancer/tracingbalancer.h>

using namespace LibGeoDecomp;

namespace HPXSerializationTest {

class TestBalancer : public LoadBalancer
{
public:
    using LoadBalancer::WeightVec;
    using LoadBalancer::LoadVec;

    TestBalancer(int factor = 0) :
        factor(factor)
    {}

    virtual WeightVec balance(const WeightVec& weights, const LoadVec& relativeLoads)
    {
        WeightVec ret(weights.size());

        for (std::size_t i = 0; i < weights.size(); ++i) {
            ret[i] = factor * weights[i];
        }

        return ret;
    }

    int factor;
};

template <typename Archive>
void serialize(Archive & archive, TestBalancer& balancer, unsigned)
{
    archive & balancer.factor;
}

}

HPX_SERIALIZATION_REGISTER_CLASS(HPXSerializationTest::TestBalancer);

namespace LibGeoDecomp {

class TracingBalancerTest : public CxxTest::TestSuite
{
public:

    void testSerialization()
    {
        std::ostringstream voidStream;
        TracingBalancer balancer1(new HPXSerializationTest::TestBalancer(4711));
        TracingBalancer balancer2(0, voidStream);

        std::vector<char> buffer;
        hpx::serialization::output_archive outputArchive(buffer);

        outputArchive << balancer1;

        hpx::serialization::input_archive inputArchive(buffer);
        inputArchive >> balancer2;

        LoadBalancer::WeightVec weights1;
        weights1 << 1
                 << 10
                 << 100;

        LoadBalancer::WeightVec weights2;
        LoadBalancer::LoadVec loads;

        weights2 = balancer2.balance(weights1, loads);
        TS_ASSERT_EQUALS(weights2.size(), 3);
        TS_ASSERT_EQUALS(weights2[0], 4711);
        TS_ASSERT_EQUALS(weights2[1], 47110);
        TS_ASSERT_EQUALS(weights2[2], 471100);
    }

};

}
