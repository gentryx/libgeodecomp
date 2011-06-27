#include <libgeodecomp/parallelization/hiparsimulator/partitions/hindexingpartition.h>

namespace LibGeoDecomp {
namespace HiParSimulator {

int HIndexingPartition::triangleTransitions[4][4] = {
    {0,1,2,0},
    {1,3,0,1},
    {2,0,3,2},
    {3,2,1,3}
};

Coord<2> HIndexingPartition::maxCachedDimensions;

boost::shared_ptr<boost::multi_array<SuperVector<Coord<2> >, 3> > HIndexingPartition::triangleCoordsCache;

SuperMap<std::pair<Coord<2>, unsigned>, unsigned> HIndexingPartition::triangleLengthCache;

bool HIndexingPartition::cachesInitialized = HIndexingPartition::fillCaches();

}
}
