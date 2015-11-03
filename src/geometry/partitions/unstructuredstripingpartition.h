#ifndef LIBGEODECOMP_GEOMETRY_PARTITIONS_UNSTRUCTUREDSTRIPINGPARTITION_H
#define LIBGEODECOMP_GEOMETRY_PARTITIONS_UNSTRUCTUREDSTRIPINGPARTITION_H

#include <libgeodecomp/config.h>
#include <libgeodecomp/geometry/adjacency.h>
#include <libgeodecomp/geometry/partitions/partition.h>

namespace LibGeoDecomp {

class UnstructuredStripingPartition : public Partition<1>
{
public:
    using Partition<1>::startOffsets;
    using Partition<1>::weights;

    explicit UnstructuredStripingPartition(
            const Coord<1> origin,
            const Coord<1> /* unused: dimensions */,
            const long offset,
            const std::vector<std::size_t>& weights) :
        Partition<1>(origin.x() + offset, weights)
    {}

    Region<1> getRegion(const std::size_t node) const override
    {
        Region<1> region;
        region << Streak<1>(Coord<1>(startOffsets[node + 0]), startOffsets[node + 1]);
        return region;
    }
};

}

#endif

