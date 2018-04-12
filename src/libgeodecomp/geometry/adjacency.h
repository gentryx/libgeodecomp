#ifndef LIBGEODECOMP_GEOMETRY_ADJACENCY_H
#define LIBGEODECOMP_GEOMETRY_ADJACENCY_H

#include <libgeodecomp/geometry/coord.h>
#include <libgeodecomp/misc/sharedptr.h>

// Kill some warnings in system headers
#ifdef _MSC_BUILD
#pragma warning( push )
#pragma warning( disable : 4514 4710 4711 )
#endif

#include <map>
#include <vector>

#ifdef _MSC_BUILD
#pragma warning( pop )
#endif

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
typename SharedPtr<ADJACENCY>::Type MakeAdjacency(const std::vector<std::pair<Coord<2>, T> >& weights)
{
    typename SharedPtr<ADJACENCY>::Type result(new ADJACENCY);
    typedef std::vector<std::pair<Coord<2>, T> > SparseMatrix;

    for (typename SparseMatrix::const_iterator it = weights.begin(); it != weights.end(); ++it) {
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
