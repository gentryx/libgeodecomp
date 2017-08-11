// Hardwire this warning to off as MSVC would otherwise complain about
// inline functions not being included in object files:
#ifdef _MSC_BUILD
#pragma warning( push )
#pragma warning( disable : 4514 4548 )
#endif

#include <libgeodecomp/loadbalancer/biasbalancer.h>
#include <libgeodecomp/misc/stdcontaineroverloads.h>

#ifdef _MSC_BUILD
#pragma warning( pop )
#endif

namespace LibGeoDecomp {

BiasBalancer::BiasBalancer(LoadBalancer *balancer) :
    pristine(true),
    balancer(balancer)
{
}


BiasBalancer::WeightVec BiasBalancer::loadOnOneNodeOnly(WeightVec weights) const
{
    WeightVec ret(weights.size(), 0);
    ret[0] = sum(weights);

    return ret;
}


BiasBalancer::WeightVec BiasBalancer::balance(
    const BiasBalancer::WeightVec& weights,
    const BiasBalancer::LoadVec& relativeLoads)
{
    if (pristine) {
        pristine = false;
        return loadOnOneNodeOnly(weights);
    } else {
        return balancer->balance(weights, relativeLoads);
    }
}

}

#ifdef _MSC_BUILD
#pragma warning( disable : 4710 )
#endif
