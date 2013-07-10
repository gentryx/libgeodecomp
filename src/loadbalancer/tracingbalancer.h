#ifndef LIBGEODECOMP_LOADBALANCER_TRACINGBALANCER_H
#define LIBGEODECOMP_LOADBALANCER_TRACINGBALANCER_H

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
#ifdef LIBGEODECOMP_FEATURE_BOOST_SERIALIZATION
    friend class boost::serialization::access;
#endif

    TracingBalancer() :
        stream(std::cout)
    {
    }

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

#ifdef LIBGEODECOMP_FEATURE_BOOST_SERIALIZATION
    template<typename ARCHIVE>
    void serialize(ARCHIVE& ar, unsigned)
    {
        ar & boost::serialization::base_object<LoadBalancer>(*this);
        ar & balancer;
    }
#endif
};

}

#endif
