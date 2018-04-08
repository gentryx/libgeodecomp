#ifndef LIBGEODECOMP_COMMUNICATION_HPXSERIALIZATIONWRAPPER_H
#define LIBGEODECOMP_COMMUNICATION_HPXSERIALIZATIONWRAPPER_H

#include <libgeodecomp/config.h>
#ifdef LIBGEODECOMP_WITH_HPX

/**
 * including hpxserialization.h only works if other headers are pulled
 * in as well. Repeating those on all sites is tedious. Instead we
 * just pull in this header.
 */
#include <libgeodecomp/communication/hpxserialization.h>

// Kill warning 4514 in system headers
#ifdef _MSC_BUILD
#pragma warning( push )
#pragma warning( disable : 4514 )
#endif

#include <hpx/runtime/serialization/map.hpp>
#include <hpx/runtime/serialization/serialize.hpp>
#include <hpx/runtime/serialization/shared_ptr.hpp>
#include <hpx/runtime/serialization/vector.hpp>

#ifdef _MSC_BUILD
#pragma warning( pop )
#endif

#endif

#endif
