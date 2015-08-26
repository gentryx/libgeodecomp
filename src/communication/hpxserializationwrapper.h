#ifndef LIBGEODECOMP_HPXSERIALIZATIONWRAPPER_H
#define LIBGEODECOMP_HPXSERIALIZATIONWRAPPER_H

#include <libgeodecomp/config.h>
#ifdef LIBGEODECOMP_WITH_HPX

/**
 * including hpxserialization.h only works if other headers are pulled
 * in as well. Repeating those on all sites is tedious. Instead we
 * just pull in this header.
 */
// #include <hpx/runtime/serialization/serialization_fwd.hpp>
// #include <hpx/runtime/serialization/serialization_chunk.hpp>
// #include <hpx/runtime/serialization/input_container.hpp>
// #include <hpx/runtime/serialization/container.hpp>
// #include <hpx/runtime/serialization/base_object.hpp>
// #include <hpx/runtime/serialization/set.hpp>
#include <libgeodecomp/communication/hpxserialization.h>
#include <hpx/runtime/serialization/serialize.hpp>
#include <hpx/runtime/serialization/shared_ptr.hpp>

#endif

#endif
