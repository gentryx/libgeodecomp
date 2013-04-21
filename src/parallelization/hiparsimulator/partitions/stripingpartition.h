#ifndef LIBGEODECOMP_PARALLELIZATION_HIPARSIMULATOR_PARTITIONS_STRIPINGPARTITION_H
#define LIBGEODECOMP_PARALLELIZATION_HIPARSIMULATOR_PARTITIONS_STRIPINGPARTITION_H

#include <sstream>
#include <libgeodecomp/misc/coord.h>
#include <libgeodecomp/misc/coordbox.h>
#include <libgeodecomp/parallelization/hiparsimulator/partitions/spacefillingcurve.h>

namespace LibGeoDecomp {
namespace HiParSimulator {

template<int DIMENSIONS>
class StripingPartition : public SpaceFillingCurve<DIMENSIONS>
{
    friend class StripingPartitionTest;
public:
    const static int DIM = DIMENSIONS;
    typedef typename CoordBox<DIM>::Iterator Iterator;

    StripingPartition(
        const Coord<DIM>& _origin=Coord<DIM>(), 
        const Coord<DIM>& _dimensions=Coord<DIM>(),
        const long& offset=0,
        const SuperVector<long>& weights=SuperVector<long>(2)) :
        SpaceFillingCurve<DIM>(offset, weights),
        origin(_origin),
        dimensions(_dimensions)
    {
        if (dimensions.prod() == 0) {
            // set all dimensions to 1 except for the last one to
            // avoid division by 0 in operator[]
            dimensions = Coord<DIM>::diagonal(1);
            dimensions[DIM - 1] = 0;
        }
    }

    Iterator begin() const 
    {
        return Iterator(origin, origin, dimensions);
    }

    Iterator end() const 
    {
        Coord<DIM> endOffset;
        endOffset[DIM - 1] = dimensions[DIM - 1];
        return Iterator(origin, origin + endOffset, dimensions);
    }

    inline Region<DIM> getRegion(const long& node) const 
    {
        return Region<DIM>(
            (*this)[startOffsets[node + 0]], 
            (*this)[startOffsets[node + 1]]);
    }

    Iterator operator[](const unsigned& pos) const
    {
        Coord<DIM> cursor = IndexToCoord<DIM>()(pos, dimensions) + origin;
        return Iterator(origin, cursor, dimensions);
    }


private:
    using SpaceFillingCurve<DIMENSIONS>::startOffsets;

    Coord<DIM> origin;
    Coord<DIM> dimensions;
};

}
}

template<typename _CharT, typename _Traits, int _Dim>
std::basic_ostream<_CharT, _Traits>&
operator<<(std::basic_ostream<_CharT, _Traits>& __os,
           const typename LibGeoDecomp::HiParSimulator::StripingPartition<_Dim>::Iterator& iter)
{
    __os << iter.toString();
    return __os;
}

#endif
