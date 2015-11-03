#ifndef LIBGEODECOMP_GEOMETRY_PARTITIONS_UNSTRUCTUREDSTRIPINGPARTITION_H
#define LIBGEODECOMP_GEOMETRY_PARTITIONS_UNSTRUCTUREDSTRIPINGPARTITION_H

#include <libgeodecomp/config.h>
#include <libgeodecomp/geometry/partitions/partition.h>
#include <libgeodecomp/geometry/adjacency.h>

#ifdef LIBGEODECOMP_WITH_MPI

#include <mpi.h>

namespace LibGeoDecomp
{

template<int X, int Y>
class UnstructuredStripingPartition : public Partition<1>
{
public:
    using Partition<1>::startOffsets;
    using Partition<1>::weights;

    explicit UnstructuredStripingPartition(
            const Coord<1>& origin,
            const Coord<1>& dimensions,
            const long offset,
            const std::vector<std::size_t>& weights) :
        Partition<1>(offset, weights),
        nodes(weights.size())
    {
    }

    Region<1> getRegion(const std::size_t node) const override
    {
        Coord<2> origin(0, 0);
        Coord<2> dimensions(X, Y);

        Coord<2> from = origin + dimensions * (float)node / (float)nodes;
        Coord<2> to = origin + dimensions * (float)(node + 1) / (float)nodes;

        Region<1> region;

        for (int x = 0; x < X; ++x) {
            for (int y = from.y(); y < to.y(); ++y) {
                region << Coord<1> (y * X + x);
            }
        }

        return region;
    }

private:
    int nodes;

};

}

#endif
#endif

