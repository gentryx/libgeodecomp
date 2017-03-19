#include <libgeodecomp/loadbalancer/mockbalancer.h>

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
