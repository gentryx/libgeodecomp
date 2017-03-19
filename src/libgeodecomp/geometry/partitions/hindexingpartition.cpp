// Hardwire this warning to off as MSVC would otherwise complain about
// inline functions not being included in object files:
#ifdef _MSC_BUILD
#pragma warning( push )
#pragma warning( disable : 4514 )
#endif

#include <libgeodecomp/geometry/partitions/hindexingpartition.h>

#ifdef _MSC_BUILD
#pragma warning( pop )
#endif

namespace LibGeoDecomp {

int HIndexingPartition::triangleTransitions[4][4] = {
    {0,1,2,0},
    {1,3,0,1},
    {2,0,3,2},
    {3,2,1,3}
};

Coord<2> HIndexingPartition::maxCachedDimensions;

SharedPtr<HIndexingPartition::CacheType>::Type HIndexingPartition::triangleCoordsCache;

std::map<std::pair<Coord<2>, unsigned>, unsigned> HIndexingPartition::triangleLengthCache;

bool HIndexingPartition::cachesInitialized = HIndexingPartition::fillCaches();

}

#ifdef _MSC_BUILD
#pragma warning( disable : 4710 )
#endif
