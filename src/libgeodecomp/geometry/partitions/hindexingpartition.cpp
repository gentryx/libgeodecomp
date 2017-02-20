#include <libgeodecomp/geometry/partitions/hindexingpartition.h>

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
