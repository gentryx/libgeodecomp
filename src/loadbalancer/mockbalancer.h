#ifndef LIBGEODECOMP_LOADBALANCER_MOCKBALANCER_H
#define LIBGEODECOMP_LOADBALANCER_MOCKBALANCER_H

#include <libgeodecomp/loadbalancer/loadbalancer.h>

namespace LibGeoDecomp {

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
