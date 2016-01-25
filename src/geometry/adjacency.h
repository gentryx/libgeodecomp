#ifndef LIBGEODECOMP_GEOMETRY_ADJACENCY_H
#define LIBGEODECOMP_GEOMETRY_ADJACENCY_H

#include <libgeodecomp/geometry/coord.h>
#include <libgeodecomp/geometry/regionbasedadjacency.h>

namespace LibGeoDecomp {

typedef RegionBasedAdjacency Adjacency;

template<typename T>
Adjacency MakeAdjacency(const std::map<Coord<2>, T>& weights)
{
    Adjacency result;

    for (typename std::map<Coord<2>, T>::const_iterator it = weights.begin();
        it != weights.end(); ++it) {

        // ptscotch doesn't like edges from nodes to themselves
        if (it->first.x() == it->first.y()) {
            continue;
        }

        result.insert(it->first.x(), it->first.y());
        result.insert(it->first.y(), it->first.x());
    }

    return result;
}

}

#endif
