#ifndef LIBGEODECOMP_STORAGE_UNSTRUCTUREDLOOPPEELER_H
#define LIBGEODECOMP_STORAGE_UNSTRUCTUREDLOOPPEELER_H

namespace LibGeoDecomp {

namespace UnstructuredLoopPeelerHelpers {

/**
 * This is a wrapper for UnstructuredSoANeighborhood to do scalar
 * updates (required for loop peeling if a streak is not aligned on
 * chunk boundaries).
 */
template<typename HOOD>
class WrappedNeighborhood
{
public:
    typedef typename HOOD::ScalarIterator Iterator;
    typedef typename HOOD::SoAAccessor SoAAccessor;

    inline
    WrappedNeighborhood(HOOD& hood) :
        hood(hood)
    {}

    inline
    Iterator begin()
    {
        return hood.beginScalar();
    }

    inline
    Iterator end()
    {
        return hood.endScalar();
    }

    inline
    void operator++()
    {
        hood.incIntraChunkOffset();
    }

    inline
    const SoAAccessor *operator->() const
    {
        return hood.operator->();
    }

    inline
    WrappedNeighborhood& weights(std::size_t matrixID = 0)
    {
        hood.weights(matrixID);
        return *this;
    }

private:
    HOOD& hood;
};

}

template<typename SHORT_VEC_TYPE, typename COUNTER_TYPE1, typename COUNTER_TYPE2, typename HOOD_OLD, typename LAMBDA>
static
void unstructuredLoopPeeler(COUNTER_TYPE1 *counter, const COUNTER_TYPE2& end, HOOD_OLD& hoodOld, const LAMBDA& lambda)
{
    typedef SHORT_VEC_TYPE lfa_local_short_vec;
    typedef typename LibFlatArray::detail::flat_array::
        sibling_short_vec_switch<SHORT_VEC_TYPE, 1>::VALUE
        lfa_local_scalar;

    COUNTER_TYPE1 nextStop = *counter;
    COUNTER_TYPE1 remainder = *counter & (SHORT_VEC_TYPE::ARITY - 1);
    if (remainder != 0) {
        nextStop += SHORT_VEC_TYPE::ARITY - remainder;
    }
    COUNTER_TYPE1 lastStop = end - (end & (SHORT_VEC_TYPE::ARITY - 1));

    typedef UnstructuredLoopPeelerHelpers::WrappedNeighborhood<HOOD_OLD> WrappedHood;
    WrappedHood wrappedHood(hoodOld);

    lambda(lfa_local_scalar(),    counter, nextStop, wrappedHood);
    lambda(lfa_local_short_vec(), counter, lastStop, hoodOld);
    lambda(lfa_local_scalar(),    counter, end,      wrappedHood);
}


}

#endif
