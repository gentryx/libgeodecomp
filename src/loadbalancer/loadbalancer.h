#ifndef LIBGEODECOMP_LOADBALANCER_LOADBALANCER_H
#define LIBGEODECOMP_LOADBALANCER_LOADBALANCER_H

#include <libgeodecomp/misc/supervector.h>

namespace LibGeoDecomp {

class LoadBalancer
{
public:
    typedef SuperVector<long> WeightVec;
    typedef SuperVector<double> LoadVec;

    virtual ~LoadBalancer() {}

    /**
     * Given the current workload distribution @a weights
     * and the work time / wall clock time ratio @a relativeLoads for
     * each node, return a new, possibly better distribution "newLoads".
     *
     * Wall clock time is the sum of the work time and the waiting
     * time during which a node is blocking on communication to other
     * nodes.
     *
     * NOTE: The sum of the elements in weights and the return
     * value "newLoads" has to match, as the underlying assumption is,
     * that this sum is the number of smallest, atomic work items that
     * can be exchanged between to nodes. More formally:
     *
     * \f[
     * \sum_{i=0}^{i<n} \mbox{weights}[i] = \sum_{i=0}^{i<n} \mbox{newLoads}[i] \qquad
     * \mbox{where:}\quad n = |\mbox{weights}| = |\mbox{newLoads}|
     * \f]
     */
    virtual WeightVec balance(const WeightVec& weights, const LoadVec& relativeLoads) = 0;
};

}

#endif
