#ifndef LIBGEODECOMP_GEOMETRY_REGIONBASEDADJACENCY_H
#define LIBGEODECOMP_GEOMETRY_REGIONBASEDADJACENCY_H

#include <libgeodecomp/geometry/region.h>
#include <libgeodecomp/misc/stdcontaineroverloads.h>

namespace LibGeoDecomp {

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
class RegionBasedAdjacency
{
public:
    /**
     * Insert a single edge (from, to) to the graph
     */
    inline void insert(int from, int to) {
        region << Coord<2>(to, from);
    }

    /**
     * Insert all pairs (from, x) with x \in to, to the graph. We sort
     * parameter to to ensure fast, linear inserts.
     */
    inline void insert(int from, std::vector<int> to) {
        std::sort(to.begin(), to.end());
        Region<2> buf;
        for (std::vector<int>::const_iterator i = to.begin(); i != to.end(); ++i) {
            buf << Coord<2>(*i, from);
        }

        region += buf;
    }

    /**
     * Returns all x \in V with (node, x) \in E.
     */
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
