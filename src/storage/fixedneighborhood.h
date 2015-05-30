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
    explicit FixedNeighborhood(
        SoAAccessorIn& accessor,
        long& tempIndex,
        // X axis
        const long& offsetWest = 0,
        const long& offsetEast = 0,
        // Y axis
        const long& offsetTop = 0,
        const long& offsetBottom = 0,
        // Z axis
        const long& offsetSouth = 0,
        const long& offsetNorth = 0) :
        accessor(accessor),
        tempIndex(tempIndex),
        offsetWest(offsetWest),
        offsetEast(offsetEast),
        offsetTop(offsetTop),
        offsetBottom(offsetBottom),
        offsetSouth(offsetSouth),
        offsetNorth(offsetNorth)
    {}

    __host__ __device__
    template<int X, int Y, int Z>
    const SOA_ACCESSOR_OUT<CELL, LIBFLATARRAY_PARAMS_FULL(X, Y, Z, DIM_X, DIM_Y, DIM_Z, INDEX)>
    operator[](FixedCoord<X, Y, Z>) const
    {
        typedef SOA_ACCESSOR_OUT<CELL, LIBFLATARRAY_PARAMS_FULL(X, Y, Z, DIM_X, DIM_Y, DIM_Z, INDEX)> ACCESSOR;
        ACCESSOR tempAccessor = accessor[LibFlatArray::coord<X, Y, Z>()];
        //fixme: use gen_index from ACCESSOR?
        tempIndex =
            *tempAccessor.get_index() +
            (((X < 0) ? offsetWest                   : 0) +
             ((X > 0) ? offsetEast                   : 0)) +
            (((Y < 0) ? offsetTop    * DIM_X         : 0) +
             ((Y > 0) ? offsetBottom * DIM_X         : 0)) +
            (((Z < 0) ? offsetSouth  * DIM_X * DIM_Y : 0) +
             ((Z > 0) ? offsetNorth  * DIM_X * DIM_Y : 0));

        return ACCESSOR(tempAccessor.get_data(), tempIndex);
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

    __host__ __device__
    inline
    void operator+=(const long offset)
    {
        accessor += offset;
    }

private:
    SoAAccessorIn& accessor;
    long& tempIndex;
    const long& offsetWest;
    const long& offsetEast;
    const long& offsetTop;
    const long& offsetBottom;
    const long& offsetSouth;
    const long& offsetNorth;
};

}

#endif
