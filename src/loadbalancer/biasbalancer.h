#ifndef LIBGEODECOMP_LOADBALANCER_BIASBALANCER_H
#define LIBGEODECOMP_LOADBALANCER_BIASBALANCER_H

#include <libgeodecomp/loadbalancer/loadbalancer.h>
#include <boost/shared_ptr.hpp>

namespace LibGeoDecomp {

/**
 * This class is for testing purposes. It's meant to create an
 * unbalanced load distribution first and then hand over to the
 * balancer specified in the constructor.
 */
class BiasBalancer : public LoadBalancer
{
public:
    BiasBalancer(LoadBalancer *balancer);
    virtual WeightVec balance(const WeightVec& weights, const LoadVec& relativeLoads);

private:
    bool _pristine;
    boost::shared_ptr<LoadBalancer> _balancer;

    WeightVec oneNodeOnly(WeightVec weights) const;
};

}

#endif
