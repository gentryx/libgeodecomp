#ifndef LIBGEODECOMP_GEOMETRY_PARTITIONS_CHECKERBOARDINGPARTITION_H
#define LIBGEODECOMP_GEOMETRY_PARTITIONS_CHECKERBOARDINGPARTITION_H

#include <libgeodecomp/geometry/partitions/partition.h>

namespace LibGeoDecomp {

template<int DIM>
class CheckerboardingPartition : public Partition<DIM>
{
public:
    inline CheckerboardingPartition(
        const Coord<DIM>& origin = Coord<DIM>(),
        const Coord<DIM>& dimensions = Coord<DIM>(),
        const long& offset = 0,
        const std::vector<size_t>& weights = std::vector<std::size_t>(2)) :
        Partition<DIM>(offset, weights),
        origin(origin),
        dimensions(dimensions)
    {}

    Region<DIM> getRegion(const std::size_t node) const
    {
        const unsigned int dim0Nodes = sqrt(weights.size());
        unsigned int dim1Nodes = dim0Nodes;
        const unsigned int remain = (weights.size() - dim0Nodes * dim1Nodes);
        if(remain != 0){
            dim1Nodes += (remain / dim0Nodes);
        }

        const unsigned long dim0Box = dimensions[0]/dim0Nodes;
        const unsigned long dim1Box = dimensions[1]/dim1Nodes;
        unsigned long remain0 = 0;
        unsigned long remain1 = 0;
        if(node/dim1Nodes == (dim0Nodes-1)){
            remain0 =  dimensions[0]%dim0Nodes;
        }
        if(node%dim1Nodes == (dim1Nodes-1)){
            remain1 = dimensions[1]%dim0Nodes;
        }

        Region<DIM> r;
        r << CoordBox<DIM>(
                           Coord<DIM>(node/dim1Nodes * dim0Box, node%dim1Nodes * dim1Box),
                           Coord<DIM>(dim0Box + remain0, dim1Box + remain1));
        return r;
    }

private:
    using Partition<DIM>::startOffsets;
    using Partition<DIM>::weights;

    Coord<DIM> origin;
    Coord<DIM> dimensions;
};

}

#endif
