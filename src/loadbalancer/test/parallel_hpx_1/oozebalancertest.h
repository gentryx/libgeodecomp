#include <cxxtest/TestSuite.h>
#include <hpx/hpx.hpp>
#include <hpx/hpx_init.hpp>
#include <hpx/runtime/serialization/set.hpp>
#include <libgeodecomp/communication/hpxserializationwrapper.h>
#include <libgeodecomp/loadbalancer/tracingbalancer.h>

using namespace LibGeoDecomp;

namespace LibGeoDecomp {

class OozeBalancerTest : public CxxTest::TestSuite
{
public:

    void testSerializationByValue()
    {
        OozeBalancer balancer1(0.65);
        OozeBalancer balancer2(0.22);

        std::vector<char> buffer;
        hpx::serialization::output_archive outputArchive(buffer);

        outputArchive << balancer1;

        hpx::serialization::input_archive inputArchive(buffer);
        inputArchive >> balancer2;

        std::vector<std::size_t> weights;
        weights << 10
                << 10
                << 20;

        std::vector<double> loads;
        loads << 1.5
              << 0.8
              << 0.1;

        std::vector<std::size_t> expectedWeights;
        expectedWeights <<  6
                        <<  8
                        << 26;

        TS_ASSERT_EQUALS(expectedWeights, balancer2.balance(weights, loads));
    }

    void testSerializationViaSmartPointer()
    {
        boost::shared_ptr<LoadBalancer> balancer1(new OozeBalancer(0.23));
        boost::shared_ptr<LoadBalancer> balancer2(new OozeBalancer(0.34));

        std::vector<char> buffer;
        hpx::serialization::output_archive outputArchive(buffer);

        outputArchive << balancer1;

        hpx::serialization::input_archive inputArchive(buffer);
        inputArchive >> balancer2;

        std::vector<std::size_t> weights;
        weights << 10
                << 10
                << 20;

        std::vector<double> loads;
        loads << 0.5
              << 0.8
              << 0.2;

        std::vector<std::size_t> expectedWeights;
        expectedWeights << 10
                        <<  9
                        << 21;

        TS_ASSERT_EQUALS(expectedWeights, balancer2->balance(weights, loads));
    }
};

}
