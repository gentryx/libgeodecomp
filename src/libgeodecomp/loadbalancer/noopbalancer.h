#ifndef LIBGEODECOMP_LOADBALANCER_NOOPBALANCER_H
#define LIBGEODECOMP_LOADBALANCER_NOOPBALANCER_H

#include <libgeodecomp/loadbalancer/loadbalancer.h>

namespace LibGeoDecomp {

// Hardwire this warning to off as MSVC would otherwise complain about
// inline functions not being included in object files:
#ifdef _MSC_BUILD
#pragma warning( push )
#pragma warning( disable : 4514 )
#endif

/**
 * This class is for testing purposes and will not not modify the
 * given work loads.
 */
class NoOpBalancer : public LoadBalancer
{
public:
    virtual WeightVec balance(const WeightVec& weights, const LoadVec& /* relativeLoads */)
    {
        return weights;
    }
};

#ifdef _MSC_BUILD
#pragma warning( pop )
#endif

}

#endif
