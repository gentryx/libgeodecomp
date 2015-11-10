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
#include <hpx/runtime/serialization/map.hpp>
#include <hpx/runtime/serialization/serialize.hpp>
#include <hpx/runtime/serialization/shared_ptr.hpp>
#include <hpx/runtime/serialization/vector.hpp>

#endif

#endif
