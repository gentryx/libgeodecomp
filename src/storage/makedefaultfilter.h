#ifndef LIBGEODECOMP_STORAGE_MAKEDEFAULTFILTER_H
#define LIBGEODECOMP_STORAGE_MAKEDEFAULTFILTER_H

#include <libgeodecomp/misc/sharedptr.h>
#include <libgeodecomp/storage/defaultarrayfilter.h>
#include <libgeodecomp/storage/defaultcudafilter.h>
#include <libgeodecomp/storage/defaultcudaarrayfilter.h>
#include <libgeodecomp/storage/defaultfilter.h>

namespace LibGeoDecomp {

template<typename CELL, typename MEMBER>
inline
typename SharedPtr<FilterBase<CELL> >::Type makeDefaultFilter()
{
#ifdef __CUDACC__
    return typename SharedPtr<FilterBase<CELL> >::Type(new DefaultCUDAFilter<CELL, MEMBER, MEMBER>);
#else
    return typename SharedPtr<FilterBase<CELL> >::Type(new DefaultFilter<CELL, MEMBER, MEMBER>);
#endif
}

template<typename CELL, typename MEMBER, int ARITY>
inline
typename SharedPtr<FilterBase<CELL> >::Type makeDefaultFilter()
{
#ifdef __CUDACC__
    return typename SharedPtr<FilterBase<CELL> >::Type(new DefaultCUDAArrayFilter<CELL, MEMBER, MEMBER, ARITY>);
#else
    return typename SharedPtr<FilterBase<CELL> >::Type(new DefaultArrayFilter<CELL, MEMBER, MEMBER, ARITY>);
#endif
}


}


#endif
