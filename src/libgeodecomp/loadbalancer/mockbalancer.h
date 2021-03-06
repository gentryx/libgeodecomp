#ifndef LIBGEODECOMP_LOADBALANCER_MOCKBALANCER_H
#define LIBGEODECOMP_LOADBALANCER_MOCKBALANCER_H

#include <libgeodecomp/loadbalancer/loadbalancer.h>

namespace LibGeoDecomp {

/**
 * MockBalancer is a non-operative test class which will record all
 * calls made to its public interface.
 */
class MockBalancer : public LoadBalancer
{
public:
    MockBalancer();
    virtual ~MockBalancer();
    virtual WeightVec balance(const WeightVec& weights, const LoadVec& relativeLoads);

    static std::string events;
};

}

#endif
