#ifndef LIBGEODECOMP_LOADBALANCER_TRACINGBALANCER_H
#define LIBGEODECOMP_LOADBALANCER_TRACINGBALANCER_H

#include <libgeodecomp/loadbalancer/loadbalancer.h>
#include <libgeodecomp/misc/sharedptr.h>
#include <iostream>

namespace LibGeoDecomp {

/**
 * This class relies on another LoadBalancer to do the job, but is
 * able to pass debug output.
 */
class TracingBalancer : public LoadBalancer
{
public:
    explicit TracingBalancer(
        LoadBalancer *balancer,
        std::ostream& stream = std::cout) :
        balancer(balancer),
        stream(stream)
    {}

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

}

#endif
