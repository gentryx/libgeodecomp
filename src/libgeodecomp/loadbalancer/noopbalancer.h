#ifndef LIBGEODECOMP_LOADBALANCER_NOOPBALANCER_H
#define LIBGEODECOMP_LOADBALANCER_NOOPBALANCER_H

#include <libgeodecomp/loadbalancer/loadbalancer.h>

namespace LibGeoDecomp {

/**
 * This class is for testing purposes and will not not modify the
 * given work loads.
 */
class HPX_COMPONENT_EXPORT NoOpBalancer : public LoadBalancer
{
public:
    virtual WeightVec balance(const WeightVec& weights, const LoadVec& /* relativeLoads */)
    {
        return weights;
    }
};

}

#endif
