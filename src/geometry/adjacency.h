#ifndef LIBGEODECOMP_GEOMETRY_ADJACENCY_H
#define LIBGEODECOMP_GEOMETRY_ADJACENCY_H

//#define USE_MAP_ADJACENCY

#ifdef USE_MAP_ADJACENCY
#include <unordered_map>
#include <vector>
#endif // USE_MAP_ADJACENCY

#include <libgeodecomp/geometry/coord.h>
#include <libgeodecomp/geometry/regionbasedadjacency.h>

namespace LibGeoDecomp {

#ifdef USE_MAP_ADJACENCY
typedef std::unordered_map<int, std::vector<int> > Adjacency;
#else
typedef RegionBasedAdjacency Adjacency;
#endif // USE_MAP_ADJACENCY


template<typename T>
Adjacency MakeAdjacency(const std::map<Coord<2>, T> &weights)
{
    Adjacency result;

    for (auto &p : weights) {
        // ptscotch doesn't like edges from nodes to themselves
        if (p.first.x() == p.first.y()) continue;
#ifdef USE_MAP_ADJACENCY
        {
            auto &others = result[p.first.x()];
            if (std::find(others.begin(), others.end(), p.first.y()) == others.end())
            {
                others.push_back(p.first.y());
            }
        }

        {
            auto &others = result[p.first.y()];
            if (std::find(others.begin(), others.end(), p.first.x()) == others.end())
            {
                others.push_back(p.first.x());
            }
        }
#else
        result.insert(p.first.x(), p.first.y());
        result.insert(p.first.y(), p.first.x());
#endif // USE_MAP_ADJACENCY


    }

    return result;
}

}

#endif
