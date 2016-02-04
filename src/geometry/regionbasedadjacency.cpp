#include <libgeodecomp/geometry/regionbasedadjacency.h>
#include <libgeodecomp/geometry/coordbox.h>
#include <libgeodecomp/geometry/region.h>

namespace LibGeoDecomp
{

RegionBasedAdjacency::RegionBasedAdjacency() :
    region(new Region<2>())
{

}

void RegionBasedAdjacency::insert(int from, int to)
{
    (*region) << Coord<2>(to, from);
}

void RegionBasedAdjacency::insert(int from, std::vector<int> to)
{
    std::sort(to.begin(), to.end());
    Region<2> buf;
    for (std::vector<int>::const_iterator i = to.begin(); i != to.end(); ++i) {
        buf << Coord<2>(*i, from);
    }

    (*region) += buf;
}

void RegionBasedAdjacency::getNeighbors(int node, std::vector<int> *neighbors) const
{
    CoordBox<2> box = region->boundingBox();
    int minX = box.origin.x();
    int maxX = minX + box.dimensions.x();

    for (Region<2>::StreakIterator i = region->streakIteratorOnOrAfter(Coord<2>(minX, node));
            i != region->streakIteratorOnOrAfter(Coord<2>(maxX, node));
            ++i) {
        for (int j = i->origin.x(); j < i->endX; ++j) {
            (*neighbors) << j;
        }
    }
}

std::size_t RegionBasedAdjacency::size() const
{
    return region->size();
}

}
