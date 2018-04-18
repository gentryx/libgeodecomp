#ifndef LIBGEODECOMP_MISC_SHAREDPTR_H
#define LIBGEODECOMP_MISC_SHAREDPTR_H

#include <libflatarray/macros.hpp>

LIBFLATARRAY_DISABLE_SYSTEM_HEADER_WARNINGS_PRE
#ifdef LIBGEODECOMP_WITH_BOOST_SHARED_PTR
#include <boost/shared_ptr.hpp>
#else
#include <memory>
#endif
LIBFLATARRAY_DISABLE_SYSTEM_HEADER_WARNINGS_POST

namespace LibGeoDecomp {

/**
 * Selector for shared pointer implementations (std::shared_ptr vs.
 * boost::shared_ptr).
 */
template<typename CARGO>
class SharedPtr
{
public:
#ifdef LIBGEODECOMP_WITH_BOOST_SHARED_PTR
    typedef boost::shared_ptr<CARGO> Type;
#else
    typedef std::shared_ptr<CARGO> Type;
#endif
};

template<typename CARGO>
typename SharedPtr<CARGO>::Type makeShared(CARGO *pointer)
{
    typedef typename SharedPtr<CARGO>::Type SharedPointer;
    return SharedPointer(pointer);
}

}

#endif
