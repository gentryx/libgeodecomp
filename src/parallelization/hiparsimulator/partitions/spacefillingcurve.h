#ifndef _libgeodecomp_parallelization_hiparsimulator_partitions_spacefillingcurve_h_
#define _libgeodecomp_parallelization_hiparsimulator_partitions_spacefillingcurve_h_

#include <libgeodecomp/misc/coord.h>

namespace LibGeoDecomp {
namespace HiParSimulator {

enum SpaceFillingCurveSublevelState {TRIVIAL, CACHED};

template<int DIM>
class SpaceFillingCurve
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
};

}
}

#endif
