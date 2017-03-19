// Hardwire this warning to off as MSVC would otherwise complain about
// inline functions not being included in object files:
#ifdef _MSC_BUILD
#pragma warning( push )
#pragma warning( disable : 4514 )
#endif

#include <libgeodecomp/loadbalancer/mockbalancer.h>

#ifdef _MSC_BUILD
#pragma warning( pop )
#endif

namespace LibGeoDecomp {

MockBalancer::MockBalancer()
{
    events = "";
}

MockBalancer::~MockBalancer()
{
    events += "deleted\n";
}

MockBalancer::WeightVec MockBalancer::balance(
    const MockBalancer::WeightVec& weights,
    const MockBalancer::LoadVec& relativeLoads)
{
    std::stringstream buf;
    buf << "balance() " << weights << " " << relativeLoads << "\n";
    events += buf.str();
    return weights;
}

std::string MockBalancer::events;

}

#ifdef _MSC_BUILD
#pragma warning( disable : 4710 )
#endif
