#ifndef LIBGEODECOMP_LOADBALANCER_RANDOMBALANCER_H
#define LIBGEODECOMP_LOADBALANCER_RANDOMBALANCER_H
#include <libgeodecomp/loadbalancer/loadbalancer.h>

namespace LibGeoDecomp {

/**
 * This class is for testing purposes. It's meant to return valid, but
 * randomized work loads. This behaviour can be used as a stress test
 * for parallel simulators.
 */
class RandomBalancer : public LoadBalancer
{
public:
    virtual WeightVec balance(const WeightVec& weights, const LoadVec& relativeLoads);
};

}

#endif
