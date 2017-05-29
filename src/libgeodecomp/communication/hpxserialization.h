#include<libgeodecomp/config.h>
#ifdef LIBGEODECOMP_WITH_HPX
#ifndef LIBGEODECOMP_HPXSERIALIZATION_H
#define LIBGEODECOMP_HPXSERIALIZATION_H

#include <libgeodecomp/misc/chronometer.h>
#include <libgeodecomp/misc/color.h>
#include <libgeodecomp/geometry/coord.h>
#include <libgeodecomp/geometry/coord.h>
#include <libgeodecomp/geometry/coord.h>
#include <libgeodecomp/geometry/coordbox.h>
#include <libgeodecomp/storage/fixedarray.h>
#include <libgeodecomp/geometry/floatcoord.h>
#include <libgeodecomp/geometry/floatcoord.h>
#include <libgeodecomp/geometry/floatcoord.h>
#include <libgeodecomp/misc/nonpodtestcell.h>
#include <libgeodecomp/misc/palette.h>
#include <libgeodecomp/misc/quickpalette.h>
#include <libgeodecomp/geometry/region.h>
#include <libgeodecomp/geometry/streak.h>
#include <libgeodecomp/misc/testcell.h>
#include <libgeodecomp/misc/unstructuredtestcell.h>

namespace LibGeoDecomp {
class HPXSerialization
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
    static void serialize(ARCHIVE& archive, LibGeoDecomp::Color& object, const unsigned /*version*/)
    {
        archive & object.rgb;
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

    template<typename ARCHIVE, typename VALUE>
    inline
    static void serialize(ARCHIVE& archive, LibGeoDecomp::Palette<VALUE>& object, const unsigned /*version*/)
    {
        archive & object.colors;
    }

    template<typename ARCHIVE, typename VALUE>
    inline
    static void serialize(ARCHIVE& archive, LibGeoDecomp::QuickPalette<VALUE>& object, const unsigned /*version*/)
    {
        archive & object.mark0;
        archive & object.mark1;
        archive & object.mark2;
        archive & object.mark3;
        archive & object.mark4;
        archive & object.mult;
    }

    template<typename ARCHIVE, int DIMENSIONS>
    inline
    static void serialize(ARCHIVE& archive, LibGeoDecomp::Region<DIMENSIONS>& object, const unsigned /*version*/)
    {
        archive & object.geometryCacheTainted;
        archive & object.indices;
        archive & object.myBoundingBox;
        archive & object.mySize;
    }

    template<typename ARCHIVE, int DIM>
    inline
    static void serialize(ARCHIVE& archive, LibGeoDecomp::Streak<DIM>& object, const unsigned /*version*/)
    {
        archive & object.endX;
        archive & object.origin;
    }

    template<typename ARCHIVE, int DIM, typename STENCIL, typename TOPOLOGY, typename ADDITIONAL_API, typename OUTPUT>
    inline
    static void serialize(ARCHIVE& archive, LibGeoDecomp::TestCell<DIM, STENCIL, TOPOLOGY, ADDITIONAL_API, OUTPUT>& object, const unsigned /*version*/)
    {
        archive & object.cycleCounter;
        archive & object.dimensions;
        archive & object.isEdgeCell;
        archive & object.isValid;
        archive & object.pos;
        archive & object.testValue;
    }

    template<typename ARCHIVE, typename ADDITIONAL_API, typename OUTPUT>
    inline
    static void serialize(ARCHIVE& archive, LibGeoDecomp::UnstructuredTestCell<ADDITIONAL_API, OUTPUT>& object, const unsigned /*version*/)
    {
        archive & object.cycleCounter;
        archive & object.expectedNeighborWeights;
        archive & object.id;
        archive & object.isEdgeCell;
        archive & object.isValid;
    }

};
}



namespace hpx {
namespace serialization {

using namespace LibGeoDecomp;

template<class ARCHIVE>
void serialize(ARCHIVE& archive, LibGeoDecomp::Chronometer& object, const unsigned version)
{
    HPXSerialization::serialize(archive, object, version);
}

template<class ARCHIVE>
void serialize(ARCHIVE& archive, LibGeoDecomp::Color& object, const unsigned version)
{
    HPXSerialization::serialize(archive, object, version);
}

template<class ARCHIVE>
void serialize(ARCHIVE& archive, LibGeoDecomp::Coord<1 >& object, const unsigned version)
{
    HPXSerialization::serialize(archive, object, version);
}

template<class ARCHIVE>
void serialize(ARCHIVE& archive, LibGeoDecomp::Coord<2 >& object, const unsigned version)
{
    HPXSerialization::serialize(archive, object, version);
}

template<class ARCHIVE>
void serialize(ARCHIVE& archive, LibGeoDecomp::Coord<3 >& object, const unsigned version)
{
    HPXSerialization::serialize(archive, object, version);
}

template<class ARCHIVE, int DIM>
void serialize(ARCHIVE& archive, LibGeoDecomp::CoordBox<DIM>& object, const unsigned version)
{
    HPXSerialization::serialize(archive, object, version);
}

template<class ARCHIVE, typename T, int SIZE>
void serialize(ARCHIVE& archive, LibGeoDecomp::FixedArray<T, SIZE>& object, const unsigned version)
{
    HPXSerialization::serialize(archive, object, version);
}

template<class ARCHIVE>
void serialize(ARCHIVE& archive, LibGeoDecomp::FloatCoord<1 >& object, const unsigned version)
{
    HPXSerialization::serialize(archive, object, version);
}

template<class ARCHIVE>
void serialize(ARCHIVE& archive, LibGeoDecomp::FloatCoord<2 >& object, const unsigned version)
{
    HPXSerialization::serialize(archive, object, version);
}

template<class ARCHIVE>
void serialize(ARCHIVE& archive, LibGeoDecomp::FloatCoord<3 >& object, const unsigned version)
{
    HPXSerialization::serialize(archive, object, version);
}

template<class ARCHIVE>
void serialize(ARCHIVE& archive, LibGeoDecomp::NonPoDTestCell& object, const unsigned version)
{
    HPXSerialization::serialize(archive, object, version);
}

template<class ARCHIVE, typename VALUE>
void serialize(ARCHIVE& archive, LibGeoDecomp::Palette<VALUE>& object, const unsigned version)
{
    HPXSerialization::serialize(archive, object, version);
}

template<class ARCHIVE, typename VALUE>
void serialize(ARCHIVE& archive, LibGeoDecomp::QuickPalette<VALUE>& object, const unsigned version)
{
    HPXSerialization::serialize(archive, object, version);
}

template<class ARCHIVE, int DIMENSIONS>
void serialize(ARCHIVE& archive, LibGeoDecomp::Region<DIMENSIONS>& object, const unsigned version)
{
    HPXSerialization::serialize(archive, object, version);
}

template<class ARCHIVE, int DIM>
void serialize(ARCHIVE& archive, LibGeoDecomp::Streak<DIM>& object, const unsigned version)
{
    HPXSerialization::serialize(archive, object, version);
}

template<class ARCHIVE, int DIM, typename STENCIL, typename TOPOLOGY, typename ADDITIONAL_API, typename OUTPUT>
void serialize(ARCHIVE& archive, LibGeoDecomp::TestCell<DIM, STENCIL, TOPOLOGY, ADDITIONAL_API, OUTPUT>& object, const unsigned version)
{
    HPXSerialization::serialize(archive, object, version);
}

template<class ARCHIVE, typename ADDITIONAL_API, typename OUTPUT>
void serialize(ARCHIVE& archive, LibGeoDecomp::UnstructuredTestCell<ADDITIONAL_API, OUTPUT>& object, const unsigned version)
{
    HPXSerialization::serialize(archive, object, version);
}


}
}

#endif

#endif
