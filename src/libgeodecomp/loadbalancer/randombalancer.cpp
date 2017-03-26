// Hardwire this warning to off as MSVC would otherwise complain about
// inline functions not being included in object files:
#ifdef _MSC_BUILD
#pragma warning( push )
#pragma warning( disable : 4514 )
#endif

#include <libgeodecomp/loadbalancer/randombalancer.h>
#include <libgeodecomp/misc/random.h>

#ifdef _MSC_BUILD
#pragma warning( pop )
#endif

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
    std::size_t loadsSum = 0;
    for (unsigned i = 0; i < ret.size(); i++) {
        randSum += randomBase[i];
        loadsSum += weights[i];
    }

    // scaled fill & calc. remainder
    std::size_t remainder = loadsSum;
    for (unsigned i = 0; i < ret.size(); i++) {
        ret[i] = static_cast<std::size_t>(randomBase[i] * loadsSum / randSum);
        remainder -= ret[i];
    }

    // scatter remainder
    for (unsigned i = remainder; i > 0; i--) {
        ret[Random::genUnsigned(ret.size())]++;
    }

    return ret;
}

}

#ifdef _MSC_BUILD
#pragma warning( disable : 4710 )
#endif
