#ifndef _libgeodecomp_loadbalancer_tracingbalancer_h_
#define _libgeodecomp_loadbalancer_tracingbalancer_h_

#include <iostream>
#include <boost/shared_ptr.hpp>
#include <libgeodecomp/loadbalancer/loadbalancer.h>

namespace LibGeoDecomp {

/**
 * This class relies on another LoadBalancer to do the job, but is
 * able to pass debug output.
 */
class TracingBalancer : public LoadBalancer
{
public:
    TracingBalancer(LoadBalancer *balancer, std::ostream& stream = std::cout) :
        _stream(stream) 
    {
        _balancer.reset(balancer);
    }

    virtual UVec balance(const UVec& currentLoads, const DVec& relativeLoads)
    {
        _stream << "TracingBalancer::balance()\n";
        _stream << "  currentLoads: " << currentLoads << "\n";
        _stream << "  relativeLoads: " << relativeLoads << "\n";
        
        return _balancer->balance(currentLoads, relativeLoads);
    }

private:
    boost::shared_ptr<LoadBalancer> _balancer;
    std::ostream& _stream;
};

};

#endif
