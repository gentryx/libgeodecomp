#ifndef LIBGEODECOMP_STORAGE_DEFAULTFILTERFACTORY_H
#define LIBGEODECOMP_STORAGE_DEFAULTFILTERFACTORY_H

#include <libgeodecomp/misc/sharedptr.h>
#include <libgeodecomp/storage/defaultarrayfilter.h>
#include <libgeodecomp/storage/defaultcudafilter.h>
#include <libgeodecomp/storage/defaultcudaarrayfilter.h>
#include <libgeodecomp/storage/defaultfilter.h>

namespace LibGeoDecomp {

/**
 *
 */
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
        return typename SharedPtr<FilterBase<CELL> >::Type(new DefaultCUDAFilter<CELL, MEMBER, MEMBER>);
    }

    template<typename CELL, typename MEMBER, int ARITY>
    inline
    typename SharedPtr<FilterBase<CELL> >::Type make() const
    {
        return typename SharedPtr<FilterBase<CELL> >::Type(new DefaultCUDAArrayFilter<CELL, MEMBER, MEMBER, ARITY>);
    }
};
#endif

template<>
class DefaultFilterFactory<false>
{
public:
    template<typename CELL, typename MEMBER>
    inline
    typename SharedPtr<FilterBase<CELL> >::Type make() const
    {
        return typename SharedPtr<FilterBase<CELL> >::Type(new DefaultFilter<CELL, MEMBER, MEMBER>);
    }

    template<typename CELL, typename MEMBER, int ARITY>
    inline
    typename SharedPtr<FilterBase<CELL> >::Type make() const
    {
        return typename SharedPtr<FilterBase<CELL> >::Type(new DefaultArrayFilter<CELL, MEMBER, MEMBER, ARITY>);
    }
};

}


#endif
