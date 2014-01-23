#ifndef LIBGEODECOMP_LOADBALANCER_TRACINGBALANCER_H
#define LIBGEODECOMP_LOADBALANCER_TRACINGBALANCER_H

#include <libgeodecomp/loadbalancer/loadbalancer.h>

#include <boost/shared_ptr.hpp>
#include <iostream>

namespace LibGeoDecomp {

/**
 * This class relies on another LoadBalancer to do the job, but is
 * able to pass debug output.
 */
class TracingBalancer : public LoadBalancer
{
public:
    friend class Serialization;

    TracingBalancer(LoadBalancer *balancer, std::ostream& stream = std::cout) :
        balancer(balancer),
        stream(stream)
    {
    }

    virtual WeightVec balance(const WeightVec& weights, const LoadVec& relativeLoads)
    {
        stream << "TracingBalancer::balance()\n";
        stream << "  weights: " << weights << "\n";
        stream << "  relativeLoads: " << relativeLoads << "\n";

        return balancer->balance(weights, relativeLoads);
    }

private:
    boost::shared_ptr<LoadBalancer> balancer;
    std::ostream& stream;
};

}

namespace boost {
namespace serialization {

template<class Archive, typename CELL_TYPE>
inline void load_construct_data(
    Archive& archive, LibGeoDecomp::TracingBalancer *object, const unsigned version)
{
    ::new(object)LibGeoDecomp::TracingBalancer(0);
    serialize(archive, *object, version);
}

}
}

#endif
