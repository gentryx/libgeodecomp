#include<libgeodecomp/config.h>
#ifdef LIBGEODECOMP_WITH_BOOST_SERIALIZATION
#ifndef LIBGEODECOMP_SERIALIZATION_H
#define LIBGEODECOMP_SERIALIZATION_H

#include <storage/arrayfilter.h>
#include <misc/chronometer.h>
#include <misc/clonable.h>
#include <geometry/coord.h>
#include <geometry/coord.h>
#include <geometry/coord.h>
#include <geometry/coordbox.h>
#include <storage/defaultarrayfilter.h>
#include <storage/defaultfilter.h>
#include <storage/filter.h>
#include <storage/filterbase.h>
#include <storage/fixedarray.h>
#include <geometry/floatcoord.h>
#include <geometry/floatcoord.h>
#include <geometry/floatcoord.h>
#include <io/hpxwritercollector.h>
#include <io/initializer.h>
#include <loadbalancer/loadbalancer.h>
#include <misc/nonpodtestcell.h>
#include <loadbalancer/oozebalancer.h>
#include <io/parallelwriter.h>
#include <geometry/region.h>
#include <io/serialbovwriter.h>
#include <io/silowriter.h>
#include <storage/simplearrayfilter.h>
#include <storage/simplefilter.h>
#include <io/simpleinitializer.h>
#include <io/steerer.h>
#include <geometry/streak.h>
#include <loadbalancer/tracingbalancer.h>
#include <io/tracingwriter.h>
#include <io/writer.h>

namespace LibGeoDecomp {
class Serialization
{
public:
    template<typename ARCHIVE, typename CELL, typename MEMBER, typename EXTERNAL, int ARITY>
    inline
    static void serialize(ARCHIVE& archive, LibGeoDecomp::ArrayFilter<CELL, MEMBER, EXTERNAL, ARITY>& object, const unsigned /*version*/)
    {
        archive & boost::serialization::base_object<LibGeoDecomp::FilterBase<CELL > >(object);
    }

    template<typename ARCHIVE>
    inline
    static void serialize(ARCHIVE& archive, LibGeoDecomp::Chronometer& object, const unsigned /*version*/)
    {
        archive & object.totalTimes;
    }

    template<typename ARCHIVE, typename BASE, typename IMPLEMENTATION>
    inline
    static void serialize(ARCHIVE& archive, LibGeoDecomp::Clonable<BASE, IMPLEMENTATION>& object, const unsigned /*version*/)
    {
        archive & boost::serialization::base_object<BASE >(object);
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

    template<typename ARCHIVE, typename CELL, typename MEMBER, typename EXTERNAL, int ARITY>
    inline
    static void serialize(ARCHIVE& archive, LibGeoDecomp::DefaultArrayFilter<CELL, MEMBER, EXTERNAL, ARITY>& object, const unsigned /*version*/)
    {
        archive & boost::serialization::base_object<LibGeoDecomp::ArrayFilter<CELL, MEMBER, EXTERNAL, ARITY > >(object);
    }

    template<typename ARCHIVE, typename CELL, typename MEMBER, typename EXTERNAL>
    inline
    static void serialize(ARCHIVE& archive, LibGeoDecomp::DefaultFilter<CELL, MEMBER, EXTERNAL>& object, const unsigned /*version*/)
    {
        archive & boost::serialization::base_object<LibGeoDecomp::Filter<CELL, MEMBER, EXTERNAL > >(object);
    }

    template<typename ARCHIVE, typename CELL, typename MEMBER, typename EXTERNAL>
    inline
    static void serialize(ARCHIVE& archive, LibGeoDecomp::Filter<CELL, MEMBER, EXTERNAL>& object, const unsigned /*version*/)
    {
        archive & boost::serialization::base_object<LibGeoDecomp::FilterBase<CELL > >(object);
    }

    template<typename ARCHIVE, typename CELL>
    inline
    static void serialize(ARCHIVE& archive, LibGeoDecomp::FilterBase<CELL>& object, const unsigned /*version*/)
    {
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

    template<typename ARCHIVE, typename CELL_TYPE>
    inline
    static void serialize(ARCHIVE& archive, LibGeoDecomp::HpxWriterCollector<CELL_TYPE>& object, const unsigned /*version*/)
    {
        archive & boost::serialization::base_object<LibGeoDecomp::Clonable<ParallelWriter<CELL_TYPE >, HpxWriterCollector<CELL_TYPE > > >(object);
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
    static void serialize(ARCHIVE& archive, LibGeoDecomp::NonPoDTestCell& object, const unsigned /*version*/)
    {
        archive & object.coord;
        archive & object.cycleCounter;
        archive & object.missingNeighbors;
        archive & object.seenNeighbors;
        archive & object.simSpace;
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

    template<typename ARCHIVE, typename CELL_TYPE>
    inline
    static void serialize(ARCHIVE& archive, LibGeoDecomp::SerialBOVWriter<CELL_TYPE>& object, const unsigned /*version*/)
    {
        archive & boost::serialization::base_object<LibGeoDecomp::Clonable<Writer<CELL_TYPE >, SerialBOVWriter<CELL_TYPE > > >(object);
        archive & object.brickletDim;
        archive & object.selector;
    }

    template<typename ARCHIVE, typename CELL>
    inline
    static void serialize(ARCHIVE& archive, LibGeoDecomp::SiloWriter<CELL>& object, const unsigned /*version*/)
    {
        archive & boost::serialization::base_object<LibGeoDecomp::Clonable<Writer<CELL >, SiloWriter<CELL > > >(object);
        archive & object.cellSelectors;
        archive & object.coords;
        archive & object.databaseType;
        archive & object.elementTypes;
        archive & object.nodeList;
        archive & object.pointMeshLabel;
        archive & object.pointMeshSelectors;
        archive & object.region;
        archive & object.regularGridLabel;
        archive & object.shapeCounts;
        archive & object.shapeSizes;
        archive & object.unstructuredGridSelectors;
        archive & object.unstructuredMeshLabel;
        archive & object.variableData;
    }

    template<typename ARCHIVE, typename CELL, typename MEMBER, typename EXTERNAL, int ARITY>
    inline
    static void serialize(ARCHIVE& archive, LibGeoDecomp::SimpleArrayFilter<CELL, MEMBER, EXTERNAL, ARITY>& object, const unsigned /*version*/)
    {
        archive & boost::serialization::base_object<LibGeoDecomp::ArrayFilter<CELL, MEMBER, EXTERNAL, ARITY > >(object);
    }

    template<typename ARCHIVE, typename CELL, typename MEMBER, typename EXTERNAL>
    inline
    static void serialize(ARCHIVE& archive, LibGeoDecomp::SimpleFilter<CELL, MEMBER, EXTERNAL>& object, const unsigned /*version*/)
    {
        archive & boost::serialization::base_object<LibGeoDecomp::Filter<CELL, MEMBER, EXTERNAL > >(object);
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
        archive & boost::serialization::base_object<LibGeoDecomp::Clonable<ParallelWriter<CELL_TYPE >, TracingWriter<CELL_TYPE > > >(object);
        archive & boost::serialization::base_object<LibGeoDecomp::Clonable<Writer<CELL_TYPE >, TracingWriter<CELL_TYPE > > >(object);
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

template<class ARCHIVE, typename CELL, typename MEMBER, typename EXTERNAL, int ARITY>
void serialize(ARCHIVE& archive, LibGeoDecomp::ArrayFilter<CELL, MEMBER, EXTERNAL, ARITY>& object, const unsigned version)
{
    Serialization::serialize(archive, object, version);
}

template<class ARCHIVE>
void serialize(ARCHIVE& archive, LibGeoDecomp::Chronometer& object, const unsigned version)
{
    Serialization::serialize(archive, object, version);
}

template<class ARCHIVE, typename BASE, typename IMPLEMENTATION>
void serialize(ARCHIVE& archive, LibGeoDecomp::Clonable<BASE, IMPLEMENTATION>& object, const unsigned version)
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

template<class ARCHIVE, typename CELL, typename MEMBER, typename EXTERNAL, int ARITY>
void serialize(ARCHIVE& archive, LibGeoDecomp::DefaultArrayFilter<CELL, MEMBER, EXTERNAL, ARITY>& object, const unsigned version)
{
    Serialization::serialize(archive, object, version);
}

template<class ARCHIVE, typename CELL, typename MEMBER, typename EXTERNAL>
void serialize(ARCHIVE& archive, LibGeoDecomp::DefaultFilter<CELL, MEMBER, EXTERNAL>& object, const unsigned version)
{
    Serialization::serialize(archive, object, version);
}

template<class ARCHIVE, typename CELL, typename MEMBER, typename EXTERNAL>
void serialize(ARCHIVE& archive, LibGeoDecomp::Filter<CELL, MEMBER, EXTERNAL>& object, const unsigned version)
{
    Serialization::serialize(archive, object, version);
}

template<class ARCHIVE, typename CELL>
void serialize(ARCHIVE& archive, LibGeoDecomp::FilterBase<CELL>& object, const unsigned version)
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

template<class ARCHIVE, typename CELL_TYPE>
void serialize(ARCHIVE& archive, LibGeoDecomp::HpxWriterCollector<CELL_TYPE>& object, const unsigned version)
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
void serialize(ARCHIVE& archive, LibGeoDecomp::NonPoDTestCell& object, const unsigned version)
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

template<class ARCHIVE, typename CELL_TYPE>
void serialize(ARCHIVE& archive, LibGeoDecomp::SerialBOVWriter<CELL_TYPE>& object, const unsigned version)
{
    Serialization::serialize(archive, object, version);
}

template<class ARCHIVE, typename CELL>
void serialize(ARCHIVE& archive, LibGeoDecomp::SiloWriter<CELL>& object, const unsigned version)
{
    Serialization::serialize(archive, object, version);
}

template<class ARCHIVE, typename CELL, typename MEMBER, typename EXTERNAL, int ARITY>
void serialize(ARCHIVE& archive, LibGeoDecomp::SimpleArrayFilter<CELL, MEMBER, EXTERNAL, ARITY>& object, const unsigned version)
{
    Serialization::serialize(archive, object, version);
}

template<class ARCHIVE, typename CELL, typename MEMBER, typename EXTERNAL>
void serialize(ARCHIVE& archive, LibGeoDecomp::SimpleFilter<CELL, MEMBER, EXTERNAL>& object, const unsigned version)
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
