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
        // @dominik: implement me
        return Region<DIM>();
    }

private:
    Coord<DIM> origin;
    Coord<DIM> dimensions;
};

}

#endif
