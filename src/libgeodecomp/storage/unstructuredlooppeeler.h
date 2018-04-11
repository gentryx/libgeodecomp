#ifndef LIBGEODECOMP_STORAGE_UNSTRUCTUREDLOOPPEELER_H
#define LIBGEODECOMP_STORAGE_UNSTRUCTUREDLOOPPEELER_H

namespace LibGeoDecomp {

namespace UnstructuredLoopPeelerHelpers {

#ifdef _MSC_BUILD
#pragma warning( push )
#pragma warning( disable : 4626 5027 )
#endif

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

#ifdef _MSC_BUILD
#pragma warning( pop )
#endif

}

/**
 * This function does the housekeeping required for vectorized update
 * routines for unstructured grid codes. It does so by peeling the
 * initial and trailing loop iterations of a Streak, which are not
 * aligned on chunk boundaries, and calls the user-supplied functor
 * (or lambda) back to execute those in a scalar fashion. The main
 * part of the Streak is executed vectorized.
 *
 * Contrary to the loop peeler for structured grids, this loop peeler
 * also needs to rebind hoodOld, the proxy object to access the old
 * grid, as iteration here works on chunks. See the SoA-enabled tests
 * of UnstructuredUpdateFunctor for for how to use this.
 */
template<typename SHORT_VEC_TYPE, typename COUNTER_TYPE1, typename COUNTER_TYPE2, typename HOOD_OLD, typename LAMBDA>
static
void unstructuredLoopPeeler(COUNTER_TYPE1 *counter, const COUNTER_TYPE2& end, HOOD_OLD& hoodOld, const LAMBDA& lambda)
{
    typedef SHORT_VEC_TYPE lfa_local_short_vec;
    typedef typename LibFlatArray::detail::flat_array::
        sibling_short_vec_switch<SHORT_VEC_TYPE, 1>::VALUE
        lfa_local_scalar;

    const COUNTER_TYPE1 arityMinusOne = static_cast<COUNTER_TYPE1>(SHORT_VEC_TYPE::ARITY - 1);
    COUNTER_TYPE1 nextStop = *counter;
    COUNTER_TYPE1 remainder = *counter & arityMinusOne;
    if (remainder != 0) {
        nextStop += SHORT_VEC_TYPE::ARITY - remainder;
    }
    COUNTER_TYPE1 lastStop = end - (end & arityMinusOne);

    typedef UnstructuredLoopPeelerHelpers::WrappedNeighborhood<HOOD_OLD> WrappedHood;
    WrappedHood wrappedHood(hoodOld);

    lambda(lfa_local_scalar(),    counter, nextStop, wrappedHood);
    lambda(lfa_local_short_vec(), counter, lastStop, hoodOld);
    lambda(lfa_local_scalar(),    counter, end,      wrappedHood);
}


}

#endif
