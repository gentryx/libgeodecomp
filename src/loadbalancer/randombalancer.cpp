#include <libgeodecomp/loadbalancer/randombalancer.h>
#include <libgeodecomp/misc/random.h>

namespace LibGeoDecomp {

RandomBalancer::WeightVec RandomBalancer::balance(
    const RandomBalancer::WeightVec& weights,
    const RandomBalancer::LoadVec& /* unused */)
{
    WeightVec ret(weights.size());
    LoadVec randomBase(weights.size());

    // independent random fill
    for (unsigned i = 0; i < randomBase.size(); i++) {
        randomBase[i] = Random::genDouble(1.0);
    }

    // calc. scaling wheights
    double randSum = 0;
    long loadsSum = 0;
    for (unsigned i = 0; i < ret.size(); i++) {
        randSum += randomBase[i];
        loadsSum += weights[i];
    }

    // scaled fill & calc. remainder
    long remainder = loadsSum;
    for (unsigned i = 0; i < ret.size(); i++) {
        ret[i] = (long)(randomBase[i] * loadsSum / randSum);
        remainder -= ret[i];
    }

    // scatter remainder
    for (unsigned i = remainder; i > 0; i--) {
        ret[Random::genUnsigned(ret.size())]++;
    }

    return ret;
}

}
