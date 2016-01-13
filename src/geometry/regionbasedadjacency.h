#ifndef LIBGEODECOMP_GEOMETRY_REGIONBASEDADJACENCY_H
#define LIBGEODECOMP_GEOMETRY_REGIONBASEDADJACENCY_H

#include <libgeodecomp/geometry/region.h>
#include <libgeodecomp/misc/stdcontaineroverloads.h>

namespace LibGeoDecomp {

class RegionBasedAdjacency
{
public:
    inline void insert(int from, int to) {
        region << Coord<2>(to, from);
    }

    inline void insert(int from, std::vector<int> to) {
        std::sort(to.begin(), to.end());
        Region<2> buf;
        for (std::vector<int>::const_iterator i = to.begin(); i != to.end(); ++i) {
            buf << Coord<2>(*i, from);
        }

        region += buf;
    }

    inline void getNeighbors(int node, std::vector<int> *neighbors) {
        CoordBox<2> box = region.boundingBox();
        int minX = box.origin.x();
        int maxX = minX + box.dimensions.x();

        for (Region<2>::StreakIterator i = region.streakIteratorOnOrAfter(Coord<2>(minX, node));
             i != region.streakIteratorOnOrAfter(Coord<2>(maxX, node));
             ++i) {
            for (int j = i->origin.x(); j < i->endX; ++j) {
                (*neighbors) << j;
            }
        }
    }

private:
    Region<2> region;
};

}

#endif
