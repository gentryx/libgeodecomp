#ifndef LIBGEODECOMP_STORAGE_MESHLESSADAPTER_H
#define LIBGEODECOMP_STORAGE_MESHLESSADAPTER_H

// Kill some warnings in system headers
#ifdef _MSC_BUILD
#pragma warning( push )
#pragma warning( disable : 4514 4710 4711 )
#endif

#include <list>
#include <set>

#ifdef _MSC_BUILD
#pragma warning( pop )
#endif

#include <libgeodecomp/geometry/floatcoord.h>
#include <libgeodecomp/geometry/topologies.h>
#include <libgeodecomp/misc/stdcontaineroverloads.h>
#include <libgeodecomp/storage/grid.h>

namespace LibGeoDecomp {

/**
 * A utility class which supports users in porting meshless codes to
 * LibGeoDecomp by superimposing a stencil structure, even though the
 * actual cells may be connected by an irregular graph.
 *
 * Its purpose is mostly to aid with computing and verifying the grid
 * geometry.
 */
template<class TOPOLOGY=Topologies::Torus<2>::Topology>
class MeshlessAdapter
{
public:
    friend class MeshlessAdapterTest;
    static const int DIM = TOPOLOGY::DIM;
    static const int MAX_SIZE = 300000;

    typedef std::list<std::pair<FloatCoord<DIM>, int> > CoordList;
    typedef Grid<CoordList, TOPOLOGY> CoordListGrid;
    typedef std::vector<std::pair<FloatCoord<DIM>, int> > CoordVec;
    typedef std::vector<std::vector<int> > Graph;

    /**
     * creates an MeshlessAdapter which assumes that the coordinates
     * of the cells are elementwise smaller than dimensions. The
     * stencil cells will be of size boxSize^DIMENSIONS.
     */
    inline explicit MeshlessAdapter(
        const FloatCoord<DIM>& dimensions=FloatCoord<DIM>(),
        double boxSize = 1) :
        dimensions(dimensions)
    {
        resetBoxSize(boxSize);
    }

    inline CoordListGrid grid() const
    {
        return CoordListGrid(discreteDim);
    }

    inline Coord<DIM> posToCoord(const FloatCoord<DIM>& pos) const
    {
        Coord<DIM> c;
        for (int i = 0; i < DIM; ++i) {
            // cut all overhanging coords
            c[i] = (std::min)(discreteDim[i] - 1, (int)(pos[i] * scale));
        }
        return c;
    }

    inline void insert(CoordListGrid *grid, const FloatCoord<DIM>& pos, int id) const
    {
        Coord<2> c = posToCoord(pos);
        (*grid)[c].push_back(std::make_pair(pos, id));
    }

    /**
     * checks if the grid cell containing pos or any of its neighbors
     * in its Moore neighborhood contains a vertex which is closer to
     * pos than the boxSize. May return a list of all found vertex IDs
     * if coords is set.
     */
    bool search(
        const CoordListGrid& positions,
        const FloatCoord<DIM>& pos,
        std::set<int> *coords = 0) const
    {
        bool found = false;
        Coord<DIM> center = posToCoord(pos);
        CoordBox<DIM> box(Coord<DIM>::diagonal(-1), Coord<DIM>::diagonal(3));

        for (typename CoordBox<DIM>::Iterator i = box.begin(); i != box.end(); ++i) {
            Coord<DIM> newCenter = center + *i;
            bool res = searchList(positions[newCenter], pos, coords);
            found |= res;
        }

        return found;
    }

    inline CoordVec findAllPositions(const CoordListGrid& positions) const
    {
        CoordVec ret;
        CoordBox<DIM> box = positions.boundingBox();

        for (typename CoordBox<DIM>::Iterator i = box.begin(); i != box.end(); ++i) {
            const CoordList& list = positions[*i];
            for (typename CoordList::const_iterator j = list.begin(); j != list.end(); ++j) {
                ret.push_back(*i);
            }
        }

        return ret;
    }

    double findOptimumBoxSize(
        const CoordVec& positions,
        const Graph& graph)
    {
        double upperBorder = -1;
        double lowerBorder = -1;

        Coord<DIM> upperBorderDim;
        Coord<DIM> lowerBorderDim;

        // the exponential back-off algorithm allows us to find upper
        // and lower bound for the optimal box size in log time
        if (checkBoxSize(positions, graph)) {
            upperBorder = boxSize;
            upperBorderDim = discreteDim;
            resetBoxSize(boxSize * 0.5);
            while (checkBoxSize(positions, graph))
                resetBoxSize(boxSize * 0.5);
            lowerBorder = boxSize;
            lowerBorderDim = discreteDim;
        } else {
            lowerBorder = boxSize;
            lowerBorderDim = discreteDim;
            resetBoxSize(boxSize * 2);
            while (!checkBoxSize(positions, graph))
                resetBoxSize(boxSize * 2);
            upperBorder = boxSize;
            upperBorderDim = discreteDim;
        }

        // The loop condition is not very tight. I'd rather test for
        // equality here. But the loose test is necessary to keep the
        // number of iterations low in the general case, and finite in
        // special cases where deadlocks would occurr otherwise.
        while (intervalTooLarge(lowerBorderDim, upperBorderDim)) {
            double middle = (upperBorder + lowerBorder) * 0.5;
            resetBoxSize(middle);
            if (checkBoxSize(positions, graph)) {
                upperBorder = middle;
                upperBorderDim = discreteDim;
            } else {
                lowerBorder = middle;
                lowerBorderDim = discreteDim;
            }
        }

        double maxBoxSize = 0;
        double nextLower = 0;

        // the resulting box size may lead to overhangs on the far
        // boundaries which would get added to the nearest container
        // cells. this step enlarges the box size slightly to ensure a
        // smooth distribution.
        for (int i = 0; i < DIM; ++i) {
            double current = dimensions[i] / upperBorderDim[i];
            if (current > maxBoxSize) {
                maxBoxSize = current;
                // because the loop condition above is not tight, the
                // next smaller box size might represet a valid
                // solution, too.
                nextLower = dimensions[i] / (upperBorderDim[i] + 1);
            }
        }

        resetBoxSize(nextLower);
        if (checkBoxSize(positions, graph))
            return nextLower;

        resetBoxSize(maxBoxSize);
        // this should never happen, but might in arcane cases where
        // the packaging density is much higher on the far boundaries
        // than elsewhere in the the simulation space.
        if (!checkBoxSize(positions, graph))
            throw std::logic_error("failed to determine a valid box size");
        return maxBoxSize;
    }

    bool checkBoxSize(const CoordVec& positions, const Graph& graph)
    {
        for (std::size_t i = 0; i < graph.size(); ++i) {
            for (std::vector<int>::const_iterator n = graph[i].begin();
                 n != graph[i].end(); ++n) {
                std::size_t otherIndex = static_cast<std::size_t>(*n);
                if (manhattanDistance(positions[i].first, positions[otherIndex].first) > 1) {
                    return false;
                }
            }
        }

        return true;
    }

    const Coord<DIM>& getDiscreteDim() const
    {
        return discreteDim;
    }

    std::map<std::string, double> reportFillLevels(const CoordVec& positions) const
    {
        std::map<Coord<DIM>, int> cache;
        for (typename CoordVec::const_iterator i = positions.begin(); i != positions.end(); ++i) {
            Coord<DIM> c = posToCoord(*i);
            cache[c]++;
        }

        long sum = 0;
        long emptyCells = 0;
        int lowestFill = cache[Coord<DIM>()];
        int highestFill = cache[Coord<DIM>()];
        CoordBox<DIM> box(Coord<DIM>(), discreteDim);

        for (typename CoordBox<DIM>::Iterator i = box.begin(); i != box.end(); ++i) {
            lowestFill  = (std::min)(cache[*i], lowestFill);
            highestFill = (std::max)(cache[*i], highestFill);
            sum += cache[*i];

            if (cache[*i] == 0) {
                ++emptyCells;
            }
        }

        std::map<std::string, double> ret;
        ret["emptyCells"]  = emptyCells;
        ret["averageFill"] = 1.0 * sum / discreteDim.prod();
        ret["lowestFill"]  = lowestFill;
        ret["highestFill"] = highestFill;
        return ret;
    }

    double getBoxSize() const
    {
        return boxSize;
    }

private:
    FloatCoord<DIM> dimensions;
    Coord<DIM> discreteDim;
    double scale;
    double radius2;
    double boxSize;

    void resetBoxSize(double newBoxSize)
    {
        scale = 1 / newBoxSize;
        radius2 = newBoxSize * newBoxSize;
        boxSize = newBoxSize;

        // cut the edges via floor to avoid too thin boundaries, which
        // may interfere with a Torus topology where some cells, for
        // instance on the left border would try to access their left
        // neighbors (actually on the right side of the domain), but
        // would be unable to find them since the resulting grid thin
        // boundary container cells on the right would not contain all
        // required neighbors.
        for (int i = 0; i < DIM; ++i) {
            discreteDim[i] = (std::max)(1.0, std::floor(dimensions[i] * scale));
        }

        if (discreteDim.prod() > MAX_SIZE) {
            throw std::logic_error("too many container cells are required");
        }
    }

    bool searchList(
        const CoordList& list,
        const FloatCoord<DIM>& pos,
        std::set<int> *coords = 0) const
    {
        bool found = false;

        for (typename CoordList::const_iterator iter = list.begin();
             iter != list.end();
             ++iter) {
            if (distance2(pos, iter->first) < radius2) {
                found = true;
                if (coords)
                    coords->insert(iter->second);
            }
        }

        return found;
    }

    /**
     * returns the square of the euclidean distance of a and b.
     */
    double distance2(const FloatCoord<DIM>& a, const FloatCoord<DIM>& b) const
    {
        double dist2 = 0;

        for (int i = 0; i < DIM; ++i) {
            double delta = std::abs(a[i] - b[i]);
            if (TOPOLOGY::wrapsAxis(i)) {
                delta = (std::min)(delta, dimensions[i] - delta);
            }
            dist2 += delta * delta;
        }

        return dist2;
    }

    int manhattanDistance(const FloatCoord<DIM>& a, const FloatCoord<DIM>& b) const
    {
        Coord<DIM> coordA = posToCoord(a);
        Coord<DIM> coordB = posToCoord(b);
        Coord<DIM> delta = coordA - coordB;
        int maxDist = 0;

        for (int i = 0; i < DIM; ++i) {
            int dist = std::abs(delta[i]);
            if (TOPOLOGY::wrapsAxis(i)) {
                dist = (std::min)(dist, discreteDim[i] - dist);
            }
            maxDist = (std::max)(dist, maxDist);
        }

        return maxDist;
    }

    bool intervalTooLarge(const Coord<DIM>& a, const Coord<DIM>& b)
    {
        int maxDist = 0;

        for (int i = 0; i < DIM; ++i) {
            maxDist = (std::max)(maxDist, std::abs(a[i] - b[i]));
        }

        return maxDist > 1;

    }
};

}

#endif
