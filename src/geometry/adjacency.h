#ifndef LIBGEODECOMP_GEOMETRY_ADJACENCY_H
#define LIBGEODECOMP_GEOMETRY_ADJACENCY_H

#include <unordered_map>
#include <vector>

#include <libgeodecomp/geometry/coord.h>

namespace LibGeoDecomp {

typedef std::unordered_map<int, std::vector<int> > Adjacency;

template<typename T>
Adjacency MakeAdjacency(const std::map<Coord<2>, T> &weights)
{
    Adjacency result;

    for(auto &p : weights)
    {
        // ptscotch doesn't like edges from nodes to themselves
        if (p.first.x() == p.first.y()) continue;

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
    }

    return result;
}

}

#endif
