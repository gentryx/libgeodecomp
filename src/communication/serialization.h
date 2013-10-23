#include<libgeodecomp/config.h>
#ifdef LIBGEODECOMP_FEATURE_BOOST_SERIALIZATION
#ifndef LIBGEODECOMP_SERIALIZATION_H
#define LIBGEODECOMP_SERIALIZATION_H

#include <libgeodecomp/geometry/coord.h>
#include <libgeodecomp/geometry/coord.h>
#include <libgeodecomp/geometry/coord.h>
#include <libgeodecomp/io/writer.h>

namespace LibGeoDecomp {
class Serialization
{
public:
    template<typename ARCHIVE>
    inline
    static void serialize(ARCHIVE& archive, LibGeoDecomp::Coord<1 >& object, const unsigned /*version*/)
    {
        archive & object.c;
    }


    template<typename ARCHIVE>
    inline
    static void serialize(ARCHIVE& archive, LibGeoDecomp::Coord<2 >& object, const unsigned /*version*/)
    {
        archive & object.c;
    }


    template<typename ARCHIVE>
    inline
    static void serialize(ARCHIVE& archive, LibGeoDecomp::Coord<3 >& object, const unsigned /*version*/)
    {
        archive & object.c;
    }


    template<typename ARCHIVE, typename CELL_TYPE>
    inline
    static void serialize(ARCHIVE& archive, LibGeoDecomp::Writer<CELL_TYPE>& object, const unsigned /*version*/)
    {
        archive & object.period;
        archive & object.prefix;
    }


};
}

namespace boost {
namespace serialization {

using namespace LibGeoDecomp;

template<class ARCHIVE>
void serialize(ARCHIVE& archive, LibGeoDecomp::Coord<1 >& object, const unsigned version)
{
    Serialization::serialize(archive, object, version);
}

template<class ARCHIVE>
void serialize(ARCHIVE& archive, LibGeoDecomp::Coord<2 >& object, const unsigned version)
{
    Serialization::serialize(archive, object, version);
}

template<class ARCHIVE>
void serialize(ARCHIVE& archive, LibGeoDecomp::Coord<3 >& object, const unsigned version)
{
    Serialization::serialize(archive, object, version);
}

template<class ARCHIVE, typename CELL_TYPE>
void serialize(ARCHIVE& archive, LibGeoDecomp::Writer<CELL_TYPE>& object, const unsigned version)
{
    Serialization::serialize(archive, object, version);
}


}
}

#endif

#endif
