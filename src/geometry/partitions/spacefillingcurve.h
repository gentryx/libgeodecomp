#ifndef LIBGEODECOMP_GEOMETRY_PARTITIONS_SPACEFILLINGCURVE_H
#define LIBGEODECOMP_GEOMETRY_PARTITIONS_SPACEFILLINGCURVE_H

#include <libgeodecomp/geometry/coord.h>
#include <libgeodecomp/geometry/region.h>
#include <libgeodecomp/geometry/partitions/partition.h>

namespace LibGeoDecomp {

enum SpaceFillingCurveSublevelState {TRIVIAL, CACHED};

template<int DIM>
class SpaceFillingCurve : public Partition<DIM>
{
public:

    class Iterator
    {
    public:
        inline Iterator(const Coord<DIM>& origin, const bool& endReached) :
            origin(origin),
            cursor(origin),
            endReached(endReached)
        {}

        inline bool operator==(const Iterator& other) const
        {
            return (endReached == other.endReached) && (cursor == other.cursor);
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


#endif
