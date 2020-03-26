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
#include <hpx/serialization/map.hpp>
#include <hpx/serialization/serialize.hpp>
#include <hpx/serialization/shared_ptr.hpp>
#include <hpx/serialization/vector.hpp>

#endif

#endif
