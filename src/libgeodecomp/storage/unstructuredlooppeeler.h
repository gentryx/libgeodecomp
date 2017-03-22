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
    typedef SHORT_VEC_TYPE LgdLocalShortVec;
    typedef typename LibFlatArray::detail::flat_array::
        sibling_short_vec_switch<SHORT_VEC_TYPE, 1>::VALUE
        LgdLocalScalar;

    COUNTER_TYPE1 nextStop = *counter;
    COUNTER_TYPE1 remainder = *counter & (SHORT_VEC_TYPE::ARITY - 1);
    if (remainder != 0) {
        nextStop += SHORT_VEC_TYPE::ARITY - remainder;
    }
    COUNTER_TYPE1 lastStop = end - (end & (SHORT_VEC_TYPE::ARITY - 1));

    typedef UnstructuredLoopPeelerHelpers::WrappedNeighborhood<HOOD_OLD> WrappedHood;
    WrappedHood wrappedHood(hoodOld);

    lambda(LgdLocalScalar(),   counter, nextStop, wrappedHood);
    lambda(LgdLocalShortVec(), counter, lastStop, hoodOld);
    lambda(LgdLocalScalar(),   counter, end,      wrappedHood);
}

/**
 * Quick and dirty stand-in until we can have CUDA builds with C++14 enabled.
 */
#define UNSTRUCTURED_LOOP_PEELER(SHORT_VEC_TYPE, COUNTER_TYPE, COUNTER, END, HOOD_OLD_TYPE, HOOD_OLD, LAMBDA, ...) \
    {                                                                   \
        typedef SHORT_VEC_TYPE LgdLocalShortVec;                        \
        typedef typename LibFlatArray::detail::flat_array::             \
            sibling_short_vec_switch<SHORT_VEC_TYPE, 1>::VALUE          \
            LgdLocalScalar;                                             \
                                                                        \
        COUNTER_TYPE nextStop = *COUNTER;                               \
        COUNTER_TYPE remainder = *COUNTER & (SHORT_VEC_TYPE::ARITY - 1); \
        if (remainder != 0) {                                           \
            nextStop += SHORT_VEC_TYPE::ARITY - remainder;              \
        }                                                               \
        COUNTER_TYPE lastStop = END - (END & (SHORT_VEC_TYPE::ARITY - 1)); \
                                                                        \
        typedef UnstructuredLoopPeelerHelpers::WrappedNeighborhood<HOOD_OLD_TYPE> WrappedHood; \
        WrappedHood wrappedHood(HOOD_OLD);                              \
                                                                        \
        LAMBDA(LgdLocalScalar(),   COUNTER, nextStop, wrappedHood, __VA_ARGS__); \
        LAMBDA(LgdLocalShortVec(), COUNTER, lastStop, HOOD_OLD,    __VA_ARGS__); \
        LAMBDA(LgdLocalScalar(),   COUNTER, END,      wrappedHood, __VA_ARGS__); \
    }

}

#endif
