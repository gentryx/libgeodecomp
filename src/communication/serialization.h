#include<libgeodecomp/config.h>
#ifdef LIBGEODECOMP_FEATURE_BOOST_SERIALIZATION
#ifndef LIBGEODECOMP_SERIALIZATION_H
#define LIBGEODECOMP_SERIALIZATION_H

#include <libgeodecomp/misc/chronometer.h>
#include <libgeodecomp/geometry/coord.h>
#include <libgeodecomp/geometry/coord.h>
#include <libgeodecomp/geometry/coord.h>
#include <libgeodecomp/geometry/coordbox.h>
#include <libgeodecomp/storage/fixedarray.h>
#include <libgeodecomp/geometry/floatcoord.h>
#include <libgeodecomp/geometry/floatcoord.h>
#include <libgeodecomp/geometry/floatcoord.h>
#include <libgeodecomp/io/hpxwritercollector.h>
#include <libgeodecomp/io/initializer.h>
#include <libgeodecomp/loadbalancer/loadbalancer.h>
#include <libgeodecomp/loadbalancer/oozebalancer.h>
#include <libgeodecomp/io/parallelwriter.h>
#include <libgeodecomp/geometry/region.h>
#include <libgeodecomp/io/serialbovwriter.h>
#include <libgeodecomp/io/simpleinitializer.h>
#include <libgeodecomp/io/steerer.h>
#include <libgeodecomp/geometry/streak.h>
#include <libgeodecomp/loadbalancer/tracingbalancer.h>
#include <libgeodecomp/io/tracingwriter.h>
#include <libgeodecomp/io/writer.h>

namespace LibGeoDecomp {
class Serialization
{
public:
    template<typename ARCHIVE>
    inline
    static void serialize(ARCHIVE& archive, LibGeoDecomp::Chronometer& object, const unsigned /*version*/)
    {
        archive & object.totalTimes;
    }

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

    template<typename ARCHIVE, int DIM>
    inline
    static void serialize(ARCHIVE& archive, LibGeoDecomp::CoordBox<DIM>& object, const unsigned /*version*/)
    {
        archive & object.dimensions;
        archive & object.origin;
    }

    template<typename ARCHIVE, typename T, int SIZE>
    inline
    static void serialize(ARCHIVE& archive, LibGeoDecomp::FixedArray<T, SIZE>& object, const unsigned /*version*/)
    {
        archive & object.elements;
        archive & object.store;
    }

    template<typename ARCHIVE>
    inline
    static void serialize(ARCHIVE& archive, LibGeoDecomp::FloatCoord<1 >& object, const unsigned /*version*/)
    {
        archive & object.c;
    }

    template<typename ARCHIVE>
    inline
    static void serialize(ARCHIVE& archive, LibGeoDecomp::FloatCoord<2 >& object, const unsigned /*version*/)
    {
        archive & object.c;
    }

    template<typename ARCHIVE>
    inline
    static void serialize(ARCHIVE& archive, LibGeoDecomp::FloatCoord<3 >& object, const unsigned /*version*/)
    {
        archive & object.c;
    }

    template<typename ARCHIVE, typename CELL_TYPE, typename CONVERTER>
    inline
    static void serialize(ARCHIVE& archive, LibGeoDecomp::HpxWriterCollector<CELL_TYPE, CONVERTER>& object, const unsigned /*version*/)
    {
        archive & boost::serialization::base_object<LibGeoDecomp::ParallelWriter<CELL_TYPE > >(object);
        archive & object.sink;
    }

    template<typename ARCHIVE, typename CELL>
    inline
    static void serialize(ARCHIVE& archive, LibGeoDecomp::Initializer<CELL>& object, const unsigned /*version*/)
    {
    }

    template<typename ARCHIVE>
    inline
    static void serialize(ARCHIVE& archive, LibGeoDecomp::LoadBalancer& object, const unsigned /*version*/)
    {
    }

    template<typename ARCHIVE>
    inline
    static void serialize(ARCHIVE& archive, LibGeoDecomp::OozeBalancer& object, const unsigned /*version*/)
    {
        archive & boost::serialization::base_object<LibGeoDecomp::LoadBalancer >(object);
        archive & object.newLoadWeight;
    }

    template<typename ARCHIVE, typename CELL_TYPE>
    inline
    static void serialize(ARCHIVE& archive, LibGeoDecomp::ParallelWriter<CELL_TYPE>& object, const unsigned /*version*/)
    {
        archive & object.period;
        archive & object.prefix;
        archive & object.region;
    }

    template<typename ARCHIVE, int DIM>
    inline
    static void serialize(ARCHIVE& archive, LibGeoDecomp::Region<DIM>& object, const unsigned /*version*/)
    {
        archive & object.geometryCacheTainted;
        archive & object.indices;
        archive & object.myBoundingBox;
        archive & object.mySize;
    }

    template<typename ARCHIVE, typename CELL_TYPE, typename SELECTOR_TYPE>
    inline
    static void serialize(ARCHIVE& archive, LibGeoDecomp::SerialBOVWriter<CELL_TYPE, SELECTOR_TYPE>& object, const unsigned /*version*/)
    {
        archive & boost::serialization::base_object<LibGeoDecomp::Writer<CELL_TYPE > >(object);
        archive & object.brickletDim;
    }

    template<typename ARCHIVE, typename CELL_TYPE>
    inline
    static void serialize(ARCHIVE& archive, LibGeoDecomp::SimpleInitializer<CELL_TYPE>& object, const unsigned /*version*/)
    {
        archive & boost::serialization::base_object<LibGeoDecomp::Initializer<CELL_TYPE > >(object);
        archive & object.dimensions;
        archive & object.steps;
    }

    template<typename ARCHIVE, typename CELL_TYPE>
    inline
    static void serialize(ARCHIVE& archive, LibGeoDecomp::Steerer<CELL_TYPE>& object, const unsigned /*version*/)
    {
        archive & object.period;
        archive & object.region;
    }

    template<typename ARCHIVE, int DIM>
    inline
    static void serialize(ARCHIVE& archive, LibGeoDecomp::Streak<DIM>& object, const unsigned /*version*/)
    {
        archive & object.endX;
        archive & object.origin;
    }

    template<typename ARCHIVE>
    inline
    static void serialize(ARCHIVE& archive, LibGeoDecomp::TracingBalancer& object, const unsigned /*version*/)
    {
        archive & boost::serialization::base_object<LibGeoDecomp::LoadBalancer >(object);
        archive & object.balancer;
        archive & object.stream;
    }

    template<typename ARCHIVE, typename CELL_TYPE>
    inline
    static void serialize(ARCHIVE& archive, LibGeoDecomp::TracingWriter<CELL_TYPE>& object, const unsigned /*version*/)
    {
        archive & boost::serialization::base_object<LibGeoDecomp::ParallelWriter<CELL_TYPE > >(object);
        archive & boost::serialization::base_object<LibGeoDecomp::Writer<CELL_TYPE > >(object);
        archive & object.lastStep;
        archive & object.maxSteps;
        archive & object.outputRank;
        archive & object.startTime;
        archive & object.stream;
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
void serialize(ARCHIVE& archive, LibGeoDecomp::Chronometer& object, const unsigned version)
{
    Serialization::serialize(archive, object, version);
}

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

template<class ARCHIVE, int DIM>
void serialize(ARCHIVE& archive, LibGeoDecomp::CoordBox<DIM>& object, const unsigned version)
{
    Serialization::serialize(archive, object, version);
}

template<class ARCHIVE, typename T, int SIZE>
void serialize(ARCHIVE& archive, LibGeoDecomp::FixedArray<T, SIZE>& object, const unsigned version)
{
    Serialization::serialize(archive, object, version);
}

template<class ARCHIVE>
void serialize(ARCHIVE& archive, LibGeoDecomp::FloatCoord<1 >& object, const unsigned version)
{
    Serialization::serialize(archive, object, version);
}

template<class ARCHIVE>
void serialize(ARCHIVE& archive, LibGeoDecomp::FloatCoord<2 >& object, const unsigned version)
{
    Serialization::serialize(archive, object, version);
}

template<class ARCHIVE>
void serialize(ARCHIVE& archive, LibGeoDecomp::FloatCoord<3 >& object, const unsigned version)
{
    Serialization::serialize(archive, object, version);
}

template<class ARCHIVE, typename CELL_TYPE, typename CONVERTER>
void serialize(ARCHIVE& archive, LibGeoDecomp::HpxWriterCollector<CELL_TYPE, CONVERTER>& object, const unsigned version)
{
    Serialization::serialize(archive, object, version);
}

template<class ARCHIVE, typename CELL>
void serialize(ARCHIVE& archive, LibGeoDecomp::Initializer<CELL>& object, const unsigned version)
{
    Serialization::serialize(archive, object, version);
}

template<class ARCHIVE>
void serialize(ARCHIVE& archive, LibGeoDecomp::LoadBalancer& object, const unsigned version)
{
    Serialization::serialize(archive, object, version);
}

template<class ARCHIVE>
void serialize(ARCHIVE& archive, LibGeoDecomp::OozeBalancer& object, const unsigned version)
{
    Serialization::serialize(archive, object, version);
}

template<class ARCHIVE, typename CELL_TYPE>
void serialize(ARCHIVE& archive, LibGeoDecomp::ParallelWriter<CELL_TYPE>& object, const unsigned version)
{
    Serialization::serialize(archive, object, version);
}

template<class ARCHIVE, int DIM>
void serialize(ARCHIVE& archive, LibGeoDecomp::Region<DIM>& object, const unsigned version)
{
    Serialization::serialize(archive, object, version);
}

template<class ARCHIVE, typename CELL_TYPE, typename SELECTOR_TYPE>
void serialize(ARCHIVE& archive, LibGeoDecomp::SerialBOVWriter<CELL_TYPE, SELECTOR_TYPE>& object, const unsigned version)
{
    Serialization::serialize(archive, object, version);
}

template<class ARCHIVE, typename CELL_TYPE>
void serialize(ARCHIVE& archive, LibGeoDecomp::SimpleInitializer<CELL_TYPE>& object, const unsigned version)
{
    Serialization::serialize(archive, object, version);
}

template<class ARCHIVE, typename CELL_TYPE>
void serialize(ARCHIVE& archive, LibGeoDecomp::Steerer<CELL_TYPE>& object, const unsigned version)
{
    Serialization::serialize(archive, object, version);
}

template<class ARCHIVE, int DIM>
void serialize(ARCHIVE& archive, LibGeoDecomp::Streak<DIM>& object, const unsigned version)
{
    Serialization::serialize(archive, object, version);
}

template<class ARCHIVE>
void serialize(ARCHIVE& archive, LibGeoDecomp::TracingBalancer& object, const unsigned version)
{
    Serialization::serialize(archive, object, version);
}

template<class ARCHIVE, typename CELL_TYPE>
void serialize(ARCHIVE& archive, LibGeoDecomp::TracingWriter<CELL_TYPE>& object, const unsigned version)
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
