#ifndef LIBGEODECOMP_COMMUNICATION_SERIALIZATIONHELPERS_H
#define LIBGEODECOMP_COMMUNICATION_SERIALIZATIONHELPERS_H

#include<libgeodecomp/config.h>
#ifdef LIBGEODECOMP_WITH_BOOST_SERIALIZATION

#include <libgeodecomp/communication/serialization.h>

namespace boost {
namespace serialization {

/**
 * These functions can only be defined after
 * LibGeoDecomp::Serialization was defined (otherwise the
 * specializations of serialize() were not awailable).
 */

template<class Archive, typename CELL_TYPE>
inline void load_construct_data(
    Archive& archive, LibGeoDecomp::SiloWriter<CELL_TYPE> *object, const unsigned version)
{
    ::new(object)LibGeoDecomp::SiloWriter<CELL_TYPE>("dummy", 1);
    serialize(archive, *object, version);
}

template<class Archive>
inline void load_construct_data(
    Archive& archive, LibGeoDecomp::TracingBalancer *object, const unsigned version)
{
    ::new(object)LibGeoDecomp::TracingBalancer(0);
    serialize(archive, *object, version);
}

}
}

#endif

#endif
