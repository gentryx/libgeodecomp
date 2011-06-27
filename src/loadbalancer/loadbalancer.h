#ifndef _libgeodecomp_loadbalancer_loadbalancer_h_
#define _libgeodecomp_loadbalancer_loadbalancer_h_

#include <libgeodecomp/misc/commontypedefs.h>

namespace LibGeoDecomp {

class LoadBalancer
{
public:
    virtual ~LoadBalancer() {}

    /**
     * Given the current workload distribution @a currentLoads
     * and the work time / wall clock time ratio @a relativeLoads for
     * each node, return a new, possibly better distribution "newLoads". 
     * 
     * Wall clock time is the sum of the work time and the waiting
     * time during which a node is blocking on communication to other
     * nodes.
     *
     * NOTE: The sum of the elements in currentLoads and the return
     * value "newLoads" has to match, as the underlying assumption is,
     * that this sum is the number of smallest, atomic work items that
     * can be exchanged between to nodes. More formally:
     *
     * \f[
     * \sum_{i=0}^{i<n} \mbox{currentLoads}[i] = \sum_{i=0}^{i<n} \mbox{newLoads}[i] \qquad
     * \mbox{where:}\quad n = |\mbox{currentLoads}| = |\mbox{newLoads}|   
     * \f]
     */
    virtual UVec balance(const UVec& currentLoads, const DVec& relativeLoads) = 0;
};

};

#endif
