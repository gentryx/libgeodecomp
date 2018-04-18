#include <libgeodecomp/geometry/partitions/hindexingpartition.h>

namespace LibGeoDecomp {

unsigned HIndexingPartition::triangleTransitions[4][4] = {
    {0u, 1u, 2u, 0u},
    {1u, 3u, 0u, 1u},
    {2u, 0u, 3u, 2u},
    {3u, 2u, 1u, 3u}
};

Coord<2> HIndexingPartition::maxCachedDimensions;

SharedPtr<HIndexingPartition::CacheType>::Type HIndexingPartition::triangleCoordsCache;

std::map<std::pair<Coord<2>, unsigned>, unsigned> HIndexingPartition::triangleLengthCache;

#ifdef _MSC_BUILD
#pragma warning( push )
#pragma warning( disable : 4710 4711 )
#endif

bool HIndexingPartition::cachesInitialized = HIndexingPartition::fillCaches();

#ifdef _MSC_BUILD
#pragma warning( pop )
#endif

}

LIBFLATARRAY_DISABLE_SYSTEM_HEADER_WARNINGS_EOF
