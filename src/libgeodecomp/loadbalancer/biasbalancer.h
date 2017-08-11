#ifndef LIBGEODECOMP_LOADBALANCER_BIASBALANCER_H
#define LIBGEODECOMP_LOADBALANCER_BIASBALANCER_H

#include <libgeodecomp/loadbalancer/loadbalancer.h>
#include <libgeodecomp/misc/sharedptr.h>


namespace LibGeoDecomp {

#ifdef _MSC_BUILD
#pragma warning( push )
#pragma warning( disable : 4820 )
#endif

/**
 * This class is for testing purposes. It's meant to create an
 * unbalanced load distribution first and then hand over to the
 * balancer specified in the constructor.
 */
class BiasBalancer : public LoadBalancer
{
public:
    explicit BiasBalancer(LoadBalancer *balancer);
    virtual WeightVec balance(const WeightVec& weights, const LoadVec& relativeLoads);

private:
    bool pristine;
    SharedPtr<LoadBalancer>::Type balancer;

    WeightVec loadOnOneNodeOnly(WeightVec weights) const;
};

#ifdef _MSC_BUILD
#pragma warning( pop )
#endif

}

#endif
