#ifndef LIBGEODECOMP_PARALLELIZATION_HIPARSIMULATOR_PARTITIONS_SPACEFILLINGCURVE_H
#define LIBGEODECOMP_PARALLELIZATION_HIPARSIMULATOR_PARTITIONS_SPACEFILLINGCURVE_H

#include <libgeodecomp/misc/coord.h>
#include <libgeodecomp/misc/region.h>
#include <libgeodecomp/parallelization/hiparsimulator/partitions/partition.h>

namespace LibGeoDecomp {
namespace HiParSimulator {

enum SpaceFillingCurveSublevelState {TRIVIAL, CACHED};

template<int DIM>
class SpaceFillingCurve : public Partition<DIM>
{
public:

    class Iterator
    {
    public:
        inline Iterator(const Coord<DIM>& _origin, const bool& _endReached) :
            origin(_origin),
            cursor(_origin),
            endReached(_endReached)
        {}

        inline bool operator==(const Iterator& other) const
        {
            return endReached == other.endReached && cursor == other.cursor;
        }

        inline bool operator!=(const Iterator& other) const
        {
            return !(*this == other);
        }

        inline const Coord<DIM>& operator*() const
        {
            return cursor;
        }

        inline const Coord<DIM> *operator->() const
        {
            return &cursor;
        }

        static inline bool hasTrivialDimensions(const Coord<DIM>& dimensions)
        {
            int prod = dimensions.prod();
            int sum = dimensions.sum();
            return ((prod == 0) || (prod == (sum - DIM + 1)));
        }

    protected:
        Coord<DIM> origin;
        Coord<DIM> cursor;
        bool endReached;
        SpaceFillingCurveSublevelState sublevelState;
    };

    inline SpaceFillingCurve(
        const long& offset,
        const std::vector<std::size_t>& weights) :
        Partition<DIM>(offset, weights)
    {}
};

}
}

#endif
