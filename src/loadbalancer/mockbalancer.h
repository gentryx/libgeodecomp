#ifndef _libgeodecomp_loadbalancer_mockbalancer_h_
#define _libgeodecomp_loadbalancer_mockbalancer_h_

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
