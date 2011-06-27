#ifndef _libgeodecomp_loadbalancer_mockbalancer_h_
#define _libgeodecomp_loadbalancer_mockbalancer_h_

#include <libgeodecomp/loadbalancer/loadbalancer.h>

namespace LibGeoDecomp {

class MockBalancer : public LoadBalancer
{
public:
    MockBalancer();
    virtual ~MockBalancer();
    virtual UVec balance(const UVec& currentLoads, const DVec& relativeLoads);

    static std::string events;
};

};

#endif
