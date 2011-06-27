#include <libgeodecomp/misc/random.h>
#include <libgeodecomp/loadbalancer/randombalancer.h>

namespace LibGeoDecomp {

UVec RandomBalancer::balance(const UVec& currentLoads, const DVec&)
{
    UVec ret(currentLoads.size());
    DVec randomBase(currentLoads.size());

    // independent random fill
    for (unsigned i = 0; i < randomBase.size(); i++)
        randomBase[i] = Random::gen_d(1.0);
    // calc. scaling wheights
    double randSum = 0;
    unsigned loadsSum = 0;
    for (unsigned i = 0; i < ret.size(); i++) {
        randSum += randomBase[i];
        loadsSum += currentLoads[i];
    }
    // scaled fill & calc. remainder
    unsigned remainder = loadsSum;
    for (unsigned i = 0; i < ret.size(); i++) {
        ret[i] = (unsigned) (randomBase[i] * loadsSum / randSum);
        remainder -= ret[i];
    }
    // scatter remainder
    for (unsigned i = remainder; i > 0; i--) 
        ret[Random::gen_u(ret.size())]++;

    return ret;
}

};
