#ifndef LIBGEODECOMP_LOADBALANCER_TRACINGBALANCER_H
#define LIBGEODECOMP_LOADBALANCER_TRACINGBALANCER_H

#include <libgeodecomp/loadbalancer/loadbalancer.h>
#include <libgeodecomp/misc/sharedptr.h>
#include <iostream>

namespace LibGeoDecomp {

// Hardwire this warning to off as MSVC would otherwise complain about
// an assignment operator missing -- which is clearly there:
#ifdef _MSC_BUILD
#pragma warning( push )
#pragma warning( disable : 4626 )
#endif

/**
 * This class relies on another LoadBalancer to do the job, but is
 * able to pass debug output.
 */
class HPX_COMPONENT_EXPORT TracingBalancer : public LoadBalancer
{
public:
    explicit TracingBalancer(
        LoadBalancer *balancer,
        std::ostream& stream = std::cout) :
        balancer(balancer),
        stream(stream)
    {}

#ifdef LIBGEODECOMP_WITH_CPP14
    inline TracingBalancer(const TracingBalancer& other) = default;
    inline TracingBalancer(TracingBalancer&& other) = default;
#endif

    virtual WeightVec balance(const WeightVec& weights, const LoadVec& relativeLoads)
    {
        stream << "TracingBalancer::balance()\n"
               << "  weights: " << weights << "\n"
               << "  relativeLoads: " << relativeLoads << "\n";
        return balancer->balance(weights, relativeLoads);
    }

private:
    SharedPtr<LoadBalancer>::Type balancer;
    std::ostream& stream;
};

#ifdef _MSC_BUILD
#pragma warning( pop )
#endif

}

#endif
