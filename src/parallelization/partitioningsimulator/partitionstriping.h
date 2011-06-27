#include <libgeodecomp/config.h>
#ifdef LIBGEODECOMP_FEATURE_MPI
#ifndef _libgeodecomp_parallelization_partitioningsimulator_partitionstriping_h_
#define _libgeodecomp_parallelization_partitioningsimulator_partitionstriping_h_

#include <libgeodecomp/parallelization/partitioningsimulator/partition.h>

namespace LibGeoDecomp {

/**
 * This class implements a partitioning into vertical stripes
 **/
class PartitionStriping : public Partition
{
public:
    PartitionStriping(const CoordBox<2>& rect, const unsigned& numPartitions);
    CoordBox<2> coordsForNode(const unsigned& node) const;
    CoordBox<2> rectForNodes(const Nodes& nodes) const;
    unsigned nodeForCoord(const Coord<2>& coord) const;
    Nodes getNodes() const;

    void copyFromBroadcast(MPILayer& mpilayer, const unsigned& root);

private:
    unsigned _stripeWidth;
    unsigned _leftOver;
};

};

#endif
#endif
