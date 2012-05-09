#include <libgeodecomp/loadbalancer/biasbalancer.h>

namespace LibGeoDecomp {

BiasBalancer::BiasBalancer(LoadBalancer *balancer) : 
    _pristine(true),
    _balancer(balancer)
{
}


BiasBalancer::WeightVec BiasBalancer::oneNodeOnly(WeightVec weights) const
{
    long sum = 0;
    for (unsigned i = 0; i < weights.size(); i++) {
        sum += weights[i];
    }
    
    WeightVec ret(weights.size(), 0);
    ret[0] = sum;
    return ret;
}


BiasBalancer::WeightVec BiasBalancer::balance(
    const BiasBalancer::WeightVec& weights, 
    const BiasBalancer::LoadVec& relativeLoads)
{
    if (_pristine) {
        _pristine = false;
        return oneNodeOnly(weights);
    } else {
        return _balancer->balance(weights, relativeLoads);
    }
}

}
