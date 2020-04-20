#include <cxxtest/TestSuite.h>
#include <libgeodecomp/misc/testhelper.h>
#include <libgeodecomp/loadbalancer/loadbalancer.h>

using namespace LibGeoDecomp;

namespace LibGeoDecomp {

class TestLoadBalancer : public LoadBalancer
{
    typedef LoadBalancer::LoadVec LoadVec;
    typedef LoadBalancer::WeightVec WeightVec;

    virtual WeightVec balance(const WeightVec& weights, const LoadVec& relativeLoads) {
        // intentionally a NOP since we're only testing LoadBalancer::initialWeights() here.
        return WeightVec();
    }

};

class LoadBalancerTest : public CxxTest::TestSuite
{
public:
    void testInitialWeights1()
    {
        TestLoadBalancer balancer;

        std::vector<double> speeds;
        speeds << 0.5
               << 0.5
               << 0.5;

        std::vector<std::size_t> weights;
        weights = balancer.initialWeights(9, speeds);

        std::vector<std::size_t> expectedWeights;
        expectedWeights << 3
                        << 3
                        << 3;

        TS_ASSERT_EQUALS(expectedWeights, weights);
    }

    void testInitialWeights2()
    {
        TestLoadBalancer balancer;

        std::vector<double> speeds;
        speeds << 0.5
               << 0.5
               << 0.5;
        std::vector<std::size_t> weights;
        weights = balancer.initialWeights(8, speeds);

        std::vector<std::size_t> expectedWeights;
        expectedWeights << 2
                        << 3
                        << 3;

        TS_ASSERT_EQUALS(expectedWeights, weights);
    }

    void testInitialWeights3()
    {
        TestLoadBalancer balancer;

        std::vector<double> speeds;
        speeds << 0.5
               << 0.5
               << 0.5;
        std::vector<std::size_t> weights;
        weights = balancer.initialWeights(7, speeds);

        std::vector<std::size_t> expectedWeights;
        expectedWeights << 2
                        << 2
                        << 3;

        TS_ASSERT_EQUALS(expectedWeights, weights);
    }

    void testInitialWeights4()
    {
        TestLoadBalancer balancer;

        std::vector<double> speeds;
        speeds << 0.5
               << 0.5
               << 0.5;
        std::vector<std::size_t> weights;
        weights = balancer.initialWeights(6, speeds);

        std::vector<std::size_t> expectedWeights;
        expectedWeights << 2
                        << 2
                        << 2;

        TS_ASSERT_EQUALS(expectedWeights, weights);
    }

    void testInitialWeights5()
    {
        TestLoadBalancer balancer;

        // intentionally left empty
        std::vector<double> speeds;
        TS_ASSERT_THROWS(balancer.initialWeights(6, speeds), std::invalid_argument&);
    }
};

}
