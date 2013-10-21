#include <libgeodecomp/geometry/partitions/hindexingpartition.h>

namespace LibGeoDecomp {

int HIndexingPartition::triangleTransitions[4][4] = {
    {0,1,2,0},
    {1,3,0,1},
    {2,0,3,2},
    {3,2,1,3}
};

Coord<2> HIndexingPartition::maxCachedDimensions;

boost::shared_ptr<boost::multi_array<std::vector<Coord<2> >, 3> > HIndexingPartition::triangleCoordsCache;

std::map<std::pair<Coord<2>, unsigned>, unsigned> HIndexingPartition::triangleLengthCache;

bool HIndexingPartition::cachesInitialized = HIndexingPartition::fillCaches();

}
