#ifndef LIBGEODECOMP_GEOMETRY_ADJACENCY_H
#define LIBGEODECOMP_GEOMETRY_ADJACENCY_H

#include <libgeodecomp/geometry/coord.h>
#include <libgeodecomp/geometry/regionbasedadjacency.h>

namespace LibGeoDecomp {

typedef RegionBasedAdjacency Adjacency;

template<typename T>
Adjacency MakeAdjacency(const std::map<Coord<2>, T> &weights)
{
    Adjacency result;

    for (auto &p : weights) {
        // ptscotch doesn't like edges from nodes to themselves
        if (p.first.x() == p.first.y()) {
            continue;
        }

        result.insert(p.first.x(), p.first.y());
        result.insert(p.first.y(), p.first.x());
    }

    return result;
}

}

#endif
