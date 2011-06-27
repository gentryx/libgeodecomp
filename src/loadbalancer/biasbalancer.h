#ifndef _libgeodecomp_loadbalancer_biasbalancer_h_
#define _libgeodecomp_loadbalancer_biasbalancer_h_

#include <boost/shared_ptr.hpp>
#include <libgeodecomp/loadbalancer/loadbalancer.h>

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
    virtual UVec balance(const UVec& currentLoads, const DVec& relativeLoads);

private:
    bool _pristine;
    boost::shared_ptr<LoadBalancer> _balancer;

    UVec oneNodeOnly(UVec currentLoads) const;
};

};

#endif
