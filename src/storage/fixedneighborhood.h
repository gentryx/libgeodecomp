#ifndef LIBGEODECOMP_STORAGE_FIXEDNEIGHBORHOOD_H
#define LIBGEODECOMP_STORAGE_FIXEDNEIGHBORHOOD_H

#include <libflatarray/flat_array.hpp>
#include <libgeodecomp/geometry/fixedcoord.h>

namespace LibGeoDecomp {

/**
 * Similar to LinePointerNeighborhood, this class serves as a proxy
 * for cells to acccess their neighbors during update. In contrast to
 * CoordMap and LinePointerNeighborhood, class requires grid
 * dimensions to be known at compile time. The benefit is that we can
 * significantly reduce runtime overhead.
 */
template<
    typename CELL,
    long DIM_X,
    long DIM_Y,
    long DIM_Z,
    long INDEX,
    // X axis
    long OFFSET_WEST = -1,
    long OFFSET_EAST = 1,
    // Y axis
    long OFFSET_TOP = -1,
    long OFFSET_BOTTOM = 1,
    // Z axis
    long OFFSET_SOUTH = -1,
    long OFFSET_NORTH = 1,
    template<
        typename CELL2,
        long DIM_X2,
        long DIM_Y2,
        long DIM_Z2,
        long INDEX2> class SOA_ACCESSOR_IN = LibFlatArray::soa_accessor,
    template<
        typename CELL3,
        long DIM_X3,
        long DIM_Y3,
        long DIM_Z3,
        long INDEX3> class SOA_ACCESSOR_OUT = LibFlatArray::soa_accessor_light

>
class FixedNeighborhood
{
public:
    typedef SOA_ACCESSOR_IN< CELL, DIM_X, DIM_Y, DIM_Z, INDEX> SoAAccessorIn;
    typedef CELL Cell;

    __host__ __device__
    explicit FixedNeighborhood(SoAAccessorIn& accessor) :
        accessor(accessor)
    {}

    template<int X, int Y, int Z>
    __host__ __device__
    const SOA_ACCESSOR_OUT<CELL, LIBFLATARRAY_PARAMS> operator[](FixedCoord<X, Y, Z>) const
    {
        return accessor[LibFlatArray::coord<
                        ((X < 0) ? (OFFSET_WEST   + X + 1) : 0) +
                        ((X > 0) ? (OFFSET_EAST   + X - 1) : 0),
                        ((Y < 0) ? (OFFSET_TOP    + Y + 1) : 0) +
                        ((Y > 0) ? (OFFSET_BOTTOM + Y - 1) : 0),
                        ((Z < 0) ? (OFFSET_SOUTH  + Z + 1) : 0) +
                        ((Z > 0) ? (OFFSET_NORTH  + Z - 1) : 0)>()];
    }

    void operator>>(CELL& cell) const
    {
        cell << accessor;
    }

    __host__ __device__
    inline
    long& index()
    {
        return accessor.index;
    }

    __host__ __device__
    inline
    const long& index() const
    {
        return accessor.index;
    }

private:
    SoAAccessorIn& accessor;
};

}

#endif
