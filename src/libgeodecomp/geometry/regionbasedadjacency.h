/**
 * Copyright 2016-2017 Andreas Schäfer
 * Copyright 2016 Konstantin Kronfeldner
 *
 * Distributed under the Boost Software License, Version 1.0. (See accompanying
 * file LICENSE or copy at http://www.boost.org/LICENSE_1_0.txt)
 */

#ifndef LIBGEODECOMP_GEOMETRY_REGIONBASEDADJACENCY_H
#define LIBGEODECOMP_GEOMETRY_REGIONBASEDADJACENCY_H

#include <libgeodecomp/geometry/adjacency.h>
#include <libgeodecomp/geometry/region.h>
#include <libgeodecomp/misc/limits.h>
#include <libgeodecomp/misc/stdcontaineroverloads.h>

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
    friend class RegionBasedAdjacencyTest;

    explicit RegionBasedAdjacency(std::size_t maxSize = (std::size_t(1) << 30)) :
        regions(1),
        limits(1, Limits<int>::getMax()),
        maxSize(maxSize)
    {}

    /**
     * Insert a single edge (from, to) to the graph
     */
    void insert(int from, int to)
    {
        std::size_t myIndex = index(from);
        regions[myIndex] << Coord<2>(to, from);

        splitIfTooLarge(myIndex);
    }

    /**
     * Insert all pairs (from, x) with x \in to, to the graph. We sort
     * parameter to to ensure fast, linear inserts.
     */
    void insert(int from, std::vector<int> to)
    {
        std::sort(to.begin(), to.end());
        Region<2> buf;
        for (std::vector<int>::const_iterator i = to.begin(); i != to.end(); ++i) {
            buf << Coord<2>(*i, from);
        }

        std::size_t myIndex = index(from);
        regions[myIndex] += buf;

        splitIfTooLarge(myIndex);
    }

    /**
     * Returns all x \in V with (node, x) \in E.
     */
    void getNeighbors(int node, std::vector<int> *neighbors) const
    {
        std::size_t myIndex = index(node);
        CoordBox<2> box = regions[myIndex].boundingBox();
        int minX = box.origin.x();
        int maxX = minX + box.dimensions.x();

        for (Region<2>::StreakIterator i = regions[myIndex].streakIteratorOnOrAfter(Coord<2>(minX, node));
             i != regions[myIndex].streakIteratorOnOrAfter(Coord<2>(maxX, node));
             ++i) {
            for (int j = i->origin.x(); j < i->endX; ++j) {
                (*neighbors) << j;
            }
        }
    }

    /**
     * Retrieves the number of edges in the adjacency
     */
    std::size_t size() const
    {
        std::size_t sum = 0;
        for (std::vector<Region<2> >::const_iterator i = regions.begin(); i != regions.end(); ++i) {
            sum += i->size();
        }

        return sum;
    }

private:
    std::vector<Region<2> > regions;
    std::vector<int> limits;
    std::size_t maxSize;

    /**
     * split Regions which are about to overflow into two
     */
    void splitIfTooLarge(std::size_t i)
    {
        using std::swap;

        if (regions[i].numStreaks() <= maxSize) {
            return;
        }

        std::size_t yIndicesSize = regions[i].indicesSize(1);
        int newLimit = regions[i].indicesAt(1, yIndicesSize / 2)->first;

        // this swap avoids copying the data out and reduces insertion overhead
        Region<2> buf;
        swap(buf, regions[i]);

        regions.insert(regions.begin() + int(i), Region<2>());
        limits.insert(limits.begin() + int(i), newLimit);

        Region<2>::StreakIterator middle = buf.streakIteratorOnOrAfter(Coord<2>(Limits<int>::getMax(), newLimit));

        for (Region<2>::StreakIterator iter = buf.beginStreak(); iter != middle; ++iter) {
            regions[i + 0] << *iter;
        }

        for (Region<2>::StreakIterator iter = middle; iter != buf.endStreak(); ++iter) {
            regions[i + 1] << *iter;
        }
    }

    std::size_t index(int y) const
    {
        using std::lower_bound;
        return std::size_t(lower_bound(limits.begin(), limits.end(), y) - limits.begin());
    }
};

}

#endif
