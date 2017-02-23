#ifndef LIBGEODECOMP_STORAGE_DEFAULTFILTERFACTORY_H
#define LIBGEODECOMP_STORAGE_DEFAULTFILTERFACTORY_H

#include <libgeodecomp/misc/sharedptr.h>
#include <libgeodecomp/storage/defaultarrayfilter.h>
#include <libgeodecomp/storage/defaultcudafilter.h>
#include <libgeodecomp/storage/defaultcudaarrayfilter.h>
#include <libgeodecomp/storage/defaultfilter.h>

namespace LibGeoDecomp {

template<bool CUDA =
#ifdef __CUDACC__
         true
#else
         false
#endif
>
class DefaultFilterFactory;

#ifdef __CUDACC__
template<>
class DefaultFilterFactory<true>
{
public:
    template<typename CELL, typename MEMBER>
    inline
    typename SharedPtr<FilterBase<CELL> >::Type make() const
    {
        return makeShared(new DefaultCUDAFilter<CELL, MEMBER, MEMBER>);
    }

    template<typename CELL, typename MEMBER, int ARITY>
    inline
    typename SharedPtr<FilterBase<CELL> >::Type make() const
    {
        return makeShared(new DefaultCUDAArrayFilter<CELL, MEMBER, MEMBER, ARITY>);
    }
};
#endif

/**
 * Type switch that determines whether to instantiate CUDA-enabled
 * filters or not. This is tricky because the same header may be
 * compiled with nvcc as well as with g++/icpc etc. and should give
 * the correct result every time.
 */
template<>
class DefaultFilterFactory<false>
{
public:
    template<typename CELL, typename MEMBER>
    inline
    typename SharedPtr<FilterBase<CELL> >::Type make() const
    {
        return makeShared(new DefaultFilter<CELL, MEMBER, MEMBER>);
    }

    template<typename CELL, typename MEMBER, int ARITY>
    inline
    typename SharedPtr<FilterBase<CELL> >::Type make() const
    {
        return makeShared(new DefaultArrayFilter<CELL, MEMBER, MEMBER, ARITY>);
    }
};

}


#endif
