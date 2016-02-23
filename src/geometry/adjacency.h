#ifndef LIBGEODECOMP_GEOMETRY_ADJACENCY_H
#define LIBGEODECOMP_GEOMETRY_ADJACENCY_H

#include <libgeodecomp/geometry/coord.h>
#include <boost/shared_ptr.hpp>
#include <boost/make_shared.hpp>

namespace LibGeoDecomp {

/**
 * This class defines an interface for classes which model edge sets
 * for directed graphs. Unstructured grids use the adjacency to define
 * their topoloy.
 */
class Adjacency
{
public:
    virtual ~Adjacency() {}

    /**
     * Insert a single edge (from, to) to the graph
     */
    virtual void insert(int from, int to) = 0;

    /**
     * Returns all x \in V with (node, x) \in E.
     */
    virtual void getNeighbors(int node, std::vector<int> *neighbors) const = 0;

    /**
     * Retrieves the number of edges in the adjacency
     */
    virtual std::size_t size() const = 0;

};

template<typename ADJACENCY, typename T>
boost::shared_ptr<ADJACENCY> MakeAdjacency(const std::map<Coord<2>, T>& weights)
{
    boost::shared_ptr<ADJACENCY> result = boost::make_shared<ADJACENCY>();

    for (typename std::map<Coord<2>, T>::const_iterator it = weights.begin();
        it != weights.end(); ++it) {

        // ptscotch doesn't like edges from nodes to themselves
        if (it->first.x() == it->first.y()) {
            continue;
        }

        result->insert(it->first.x(), it->first.y());
        result->insert(it->first.y(), it->first.x());
    }

    return result;
}

}

#endif
