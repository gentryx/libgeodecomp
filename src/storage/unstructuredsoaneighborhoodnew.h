#ifndef LIBGEODECOMP_STORAGE_UNSTRUCTUREDSOANEIGHBORHOODNEW_H
#define LIBGEODECOMP_STORAGE_UNSTRUCTUREDSOANEIGHBORHOODNEW_H

#include <libgeodecomp/config.h>
#ifdef LIBGEODECOMP_WITH_CPP14

#include <libflatarray/soa_accessor.hpp>

namespace LibGeoDecomp {

/**
 * Neighborhood which is used for hoodNew in updateLineX().
 * Provides access to member pointers of the new grid.
 */
template<typename CELL, long DIM_X, long DIM_Y, long DIM_Z, long INDEX>
class UnstructuredSoANeighborhoodNew
{
public:
    using SoAAccessor = LibFlatArray::soa_accessor<CELL, DIM_X, DIM_Y, DIM_Z, INDEX>;

    inline explicit
    UnstructuredSoANeighborhoodNew(SoAAccessor *acc) :
        accessor(acc)
    {}

    inline
    SoAAccessor *operator->() const
    {
        return accessor;
    }

    inline
    UnstructuredSoANeighborhoodNew& operator++()
    {
        ++(*accessor);
        return *this;
    }

    inline
    UnstructuredSoANeighborhoodNew& operator+=(int i)
    {
        *accessor += i;
        return *this;
    }

    inline
    int index() const
    {
        return accessor->index();
    }

    inline
    void operator<<(const CELL& cell)
    {
        (*accessor) << cell;
    }

private:
    SoAAccessor *accessor;      /**< accessor to new grid */
};

}

#endif
#endif
