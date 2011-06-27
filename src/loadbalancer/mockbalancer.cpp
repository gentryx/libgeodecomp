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

UVec MockBalancer::balance(const UVec& currentLoads, const DVec& relativeLoads)
{
    events += "balance() " + currentLoads.toString() + " " + 
        relativeLoads.toString() + "\n";    
    return currentLoads;
}

std::string MockBalancer::events;

};
