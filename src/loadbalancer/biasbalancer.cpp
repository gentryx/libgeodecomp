#include <libgeodecomp/loadbalancer/biasbalancer.h>

namespace LibGeoDecomp {

BiasBalancer::BiasBalancer(LoadBalancer *balancer) : 
    _pristine(true),
    _balancer(balancer)
{
}


UVec BiasBalancer::oneNodeOnly(UVec currentLoads) const
{
    unsigned sum = 0;
    for (unsigned i = 0; i < currentLoads.size(); i++) {
        sum += currentLoads[i];
    }
    
    UVec ret(currentLoads.size(), 0);
    ret[0] = sum;
    return ret;
}


UVec BiasBalancer::balance(const UVec& currentLoads, const DVec& relativeLoads)
{
    if (_pristine) {
        _pristine = false;
        return oneNodeOnly(currentLoads);
    } else {
        return _balancer->balance(currentLoads, relativeLoads);
    }
}

};
