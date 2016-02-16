#ifndef LIBGEODECOMP_GEOMETRY_REGIONBASEDADJACENCY_H
#define LIBGEODECOMP_GEOMETRY_REGIONBASEDADJACENCY_H

#include <libgeodecomp/misc/stdcontaineroverloads.h>
#include <libgeodecomp/geometry/adjacency.h>
#include <boost/shared_ptr.hpp>

namespace LibGeoDecomp {

template<int DIM>
class Region;

/**
 * This class can store the adjacency list/matrix of a directed graph.
 * It performs a run-length compression for efficient storage.
 *
 * Storage complexity for a graph that comprises n nodes and a total
 * of m edges: O((n + m) * 2 * sizeof(int)).
 *
 * Insert complexity for ordered inserts is constant (O(1). Ordered
 * inserts require to be inserted edges (a, b) to satisfy for the
 * current graph (V, E) two conditions:
 *
 * 1. a >= c \forall c \in V with \exists d \in V: (x, y) \in E) and
 *
 * 2. b >= c \forall c \in V with (a, c) \in E.
 *
 * For random order inserts runtime complexity is linear in the number
 * of edges (O(m)). We could get O(log()) here with a previous
 * std::map-based implementation of Region, but storage constants as
 * well as constants for linear merges were 10x worse. That was a show
 * stopper.
 */
class RegionBasedAdjacency : public Adjacency
{
public:
    RegionBasedAdjacency();
    RegionBasedAdjacency(const RegionBasedAdjacency &other);
    RegionBasedAdjacency &operator=(const RegionBasedAdjacency &other);
    virtual ~RegionBasedAdjacency() {}

#ifdef LIBGEODECOMP_WITH_CPP14
    RegionBasedAdjacency(RegionBasedAdjacency &&other) = default;
    RegionBasedAdjacency &operator=(RegionBasedAdjacency &&other) = default;
#endif // LIBGEODECOMP_WITH_CPP14

    /**
     * Insert a single edge (from, to) to the graph
     */
    void insert(int from, int to);

    /**
     * Insert all pairs (from, x) with x \in to, to the graph. We sort
     * parameter to to ensure fast, linear inserts.
     */
    void insert(int from, std::vector<int> to);

    /**
     * Returns all x \in V with (node, x) \in E.
     */
    void getNeighbors(int node, std::vector<int> *neighbors) const;

    /**
     * Retrieves the number of edges in the adjacency
     */
    std::size_t size() const;

    const Region<2>& getRegion() const
    {
        return *region;
    }

private:
    boost::shared_ptr<Region<2> > region;
};

}

#endif
