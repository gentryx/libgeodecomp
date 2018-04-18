#include <libgeodecomp/loadbalancer/mockbalancer.h>
#include <libflatarray/macros.hpp>

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

LIBFLATARRAY_DISABLE_SYSTEM_HEADER_WARNINGS_EOF
