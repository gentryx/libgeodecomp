#ifndef LIBGEODECOMP_STORAGE_FIXEDNEIGHBORHOOD_H
#define LIBGEODECOMP_STORAGE_FIXEDNEIGHBORHOOD_H

#include <libflatarray/flat_array.hpp>
#include <libgeodecomp/geometry/fixedcoord.h>

namespace LibGeoDecomp {

namespace FixedNeighborhoodHelpers {

template<int DIM, int DISTANCE>
class NormalizeCoord
{
public:
};

}

/**
 * Similar to LinePointerNeighborhood, this class serves as a proxy
 * for cells to acccess their neighbors during update. In contrast to
 * CoordMap and LinePointerNeighborhood, class requires grid
 * dimensions to be known at compile time. The benefit is that we can
 * significantly reduce runtime overhead.
 */
template<
    typename CELL,
    typename TOPOLOGY,
    int DIM_X,
    int DIM_Y,
    int DIM_Z,
    int INDEX,
    template<typename CELL2, int DIM_X2, int DIM_Y2, int DIM_Z2, int INDEX2> class SOA_ACCESSOR = LibFlatArray::soa_accessor>
class FixedNeighborhood
{
public:
    typedef SOA_ACCESSOR<CELL, DIM_X, DIM_Y, DIM_Z, INDEX> SoAAccessor;

    __host__ __device__
    FixedNeighborhood(SoAAccessor& accessor) :
        accessor(accessor)
    {}

    template<int X, int Y, int Z>
    __host__ __device__
    const SOA_ACCESSOR<CELL, LIBFLATARRAY_PARAMS> operator[](FixedCoord<X, Y, Z>) const
    {
        return accessor[LibFlatArray::coord<X, Y, Z>()];
    }

    void operator>>(CELL& cell) const
    {
        cell << accessor;
    }

    __host__ __device__
    inline
    int& index()
    {
        return accessor.index;
    }

    __host__ __device__
    inline
    const int& index() const
    {
        return accessor.index;
    }

private:
    SoAAccessor& accessor;
};

}

#endif
