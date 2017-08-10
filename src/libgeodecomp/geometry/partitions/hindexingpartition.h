#ifndef LIBGEODECOMP_GEOMETRY_PARTITIONS_HINDEXINGPARTITION_H
#define LIBGEODECOMP_GEOMETRY_PARTITIONS_HINDEXINGPARTITION_H

#include <libgeodecomp/geometry/coord.h>
#include <libgeodecomp/geometry/partitions/spacefillingcurve.h>
#include <libgeodecomp/misc/sharedptr.h>
#include <libgeodecomp/misc/stdcontaineroverloads.h>
#include <libgeodecomp/storage/grid.h>

#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace LibGeoDecomp {

// Hardwire this warning to off as MSVC would otherwise complain about
// inline functions not being included in object files:
#ifdef _MSC_BUILD
#pragma warning( push )
#pragma warning( disable : 4514 )
#endif

/**
 * This class implements the H-Indexing scheme for arbitrary
 * rectangles.
 *
 * Note:
 * - The indexing will degenerate for rectangles with bad aspect
 *   ratio.
 * - It's not true H-Indexing "from the book", but a "diagonally
 *   dominant" variant (which doesn't matter on a large scale).
 *
 * It's core abstractions is the triangle. To fill (i.e. traverse) a
 * rectangle, it will fill two right triangles. Each triangle will
 * then be broken down into four smaller triangles. The triangles are
 * distinguished by their orientation and direction of
 * traversal. There are four types:

 * (names derrived from direction of vector from Start to End,upper
 * triangles dominate the diagonal)
 *
 * triangle types:
 *
 * 0. upper right (ur)
 *
 *       XXX End
 *       XX
 *       X
 *       Start
 *
 * 1. upper left (ul)
 *
 *   End
 *       X
 *       XX
 *         Start
 *
 * 2. lower right (lr)
 *
 * Start XXX
 *        XX
 *         X
 *         End
 *
 * 3. lower left (ll)
 *
 *           Start
 *         X
 *        XX
 *       End
 *
 */
class HIndexingPartition : public SpaceFillingCurve<2>
{
    friend class HIndexingPartitionTest;

public:
    typedef std::vector<Coord<2> > CoordVector;

    class Triangle
    {
    public:
        explicit Triangle(
            unsigned type=0,
            const Coord<2>& dimensions=Coord<2>(0, 0),
            const Coord<2>& origin=Coord<2>(0, 0),
            unsigned counter=0) :
            counter(counter),
            type(type),
            dimensions(dimensions),
            origin(origin)
        {}

        std::string toString() const
        {
            std::ostringstream o;
            o << "triangle(origin: " << origin
              << " dimensions: " << dimensions
              << " type: " << type
              << " counter: " << counter << ")\n";
            return o.str();
        }

        unsigned counter;
        unsigned type;
        Coord<2> dimensions;
        Coord<2> origin;
    };

    // triangle transition table
    // (each triangle will be broken down into 4 smaller ones:
    // (type & counter -> next subtype)
    //
    // type enumeration order
    // ---- -----------------
    // 0    0120
    // 1    1301
    // 2    2032
    // 3    3213
    //
    // e.g. type 2 consist of a type 2, then 0, then 3 and finally
    // another type 2, all of only half the dimensions of the
    // original one.
    static unsigned triangleTransitions[4][4];
    // (dimensions.x(), dimensions.y(), type) -> coord sequence
    typedef Grid<CoordVector, Topologies::Cube<3>::Topology> CacheType;
    static SharedPtr<CacheType>::Type triangleCoordsCache;
    static Coord<2> maxCachedDimensions;
    static std::map<std::pair<Coord<2>, unsigned>, unsigned> triangleLengthCache;
    static bool cachesInitialized;

    class Iterator : public SpaceFillingCurve<2>::Iterator
    {
        friend class HIndexingPartitionTest;
    public:
        /**
         * Returns an iterator that will traverse the rectangle
         * specified by  _origin and @a dimensions. Traversal will
         * start at position  pos according to the SFC
         * linearization.
         */
        inline Iterator(const Coord<2>& _origin, const Coord<2>& dimensions, unsigned pos=0) :
            SpaceFillingCurve<2>::Iterator(_origin, false)
        {
            unsigned remainder = pos;
            triangleStack.push_back(Triangle(1, dimensions, origin + dimensions, (std::numeric_limits<unsigned>::max)()));
            triangleStack.push_back(Triangle(2, dimensions, origin));
            for (;;) {
                Triangle curTria = pop(triangleStack);
                unsigned length = triangleLength(curTria);
                if (length <= remainder) {
                    remainder -= length;
                } else {
                    bool repeat = traceTriangle(curTria, &remainder);
                    if (!repeat)
                        break;
                }

                if (triangleStack.empty()) {
                    *this = Iterator(_origin);
                    break;
                }
            }
            // move to start
            ++(*this);
        }

        /**
         * Returns an iterator that will traverse only the rectangle
         * of type  initType.
         */
        inline Iterator(
            const Coord<2>& origin,
            unsigned initType,
            const Coord<2>& dimensions) :
            SpaceFillingCurve<2>::Iterator(origin, false)
        {
            triangleStack.push_back(Triangle(initType, dimensions, origin));
            digDown();
            // move to start
            ++(*this);
        }

        /**
         * Returns an "end" iterator, meaning that the iterator will
         * be frozen at position  origin.
         */
        inline explicit Iterator(const Coord<2>& _origin) :
            SpaceFillingCurve<2>::Iterator(_origin, true)
        {
        }

        inline Iterator& operator++()
        {
            if (endReached) {
                return *this;
            }
            // skip to next triangle
            // (and also skip empty trivial triangles of type 1 and 3)
            while (sublevelTriangleFinished()) {
                if (triangleStack.empty()) {
                    endReached = true;
                    cursor = origin;
                    return *this;
                }

                triangleStack.back().counter++;
                digUp();
                if (endReached) {
                    return *this;
                }
                digDown();
            }
            nextSublevel();
            return *this;
        }

    private:
        std::vector<Triangle> triangleStack;
        Coord<2> cachedTriangleOrigin;
        Coord<2> *cachedTriangleCoordsIterator;
        Coord<2> *cachedTriangleCoordsEnd;
        Coord<2> trivialTriangleDirection;
        unsigned trivialTriangleType;
        unsigned trivialTriangleCounter;
        unsigned trivialTriangleLength;

        /**
         * Ripple carry bits up and clean up completed triangles
         */
        inline void digUp()
        {
            while (triangleStack.back().counter == 4) {
                triangleStack.pop_back();
                if (triangleStack.empty()) {
                    endReached = true;
                    cursor = origin;
                    return;
                }

                triangleStack.back().counter++;
            }
        }

        /**
         * Initialize lower level triangles
         */
        inline void digDown()
        {
            Triangle curTria;
            for (curTria = pop(triangleStack);
                 !hasTrivialDimensions(curTria.dimensions) && !isCached(curTria.dimensions);
                 nextSubTriangle(&curTria)) {
                triangleStack.push_back(curTria);
            }

            if (hasTrivialDimensions(curTria.dimensions)) {
                digDownTrivial(curTria);
            } else {
                digDownCached(curTria);
            }
        }

        inline void digDownCached(const Triangle& triangle, unsigned counter=0)
        {
            sublevelState = CACHED;
            cachedTriangleOrigin = triangle.origin;
            Coord<3> c(triangle.dimensions.x(), triangle.dimensions.y(), static_cast<int>(triangle.type));
            CoordVector& coords = (*triangleCoordsCache)[c];
            cachedTriangleCoordsIterator = &coords[counter];
            cachedTriangleCoordsEnd = &coords[0] + coords.size();
        }

        inline void digDownTrivial(const Triangle& triangle, unsigned counter=0)
        {
            sublevelState = TRIVIAL;
            cursor = triangle.origin;
            trivialTriangleType = triangle.type;
            trivialTriangleCounter = counter;

            // if horizontal stripe
            if (triangle.dimensions.x() > 1) {
                trivialTriangleDirection = Coord<2>(1, 0);
                trivialTriangleLength = static_cast<unsigned>(triangle.dimensions.x());
            } else {
                trivialTriangleDirection = Coord<2>(0, triangle.type == 0? -1 : 1);
                trivialTriangleLength = static_cast<unsigned>(triangle.dimensions.y());
            }
        }

        /**
         * Finds the triangle in curTria, in which the coordinate with
         * position remainder is located. Returns true if recursion is
         * necessary to check the resulting sub triangle.
         */
        inline bool traceTriangle(const Triangle& curTria, unsigned *remainder)
        {
            if (!hasTrivialDimensions(curTria.dimensions) && !isCached(curTria.dimensions)) {
                skipSubTriangles(curTria, remainder);
                return true;
            } else {
                if (hasTrivialDimensions(curTria.dimensions)) {
                    digDownTrivial(curTria, *remainder);
                } else {
                    digDownCached(curTria, *remainder);
                }
                return false;
            }
        }

        /**
         * Finds the sub triangle of curTria, in which the position on
         * the SFC remainder resides.
         */
        inline void skipSubTriangles(const Triangle& curTria, unsigned *remainder)
        {
            triangleStack.push_back(curTria);
            for (unsigned i = 0; i < 4; ++i) {
                triangleStack.back().counter = i;
                Triangle subTria = triangleStack.back();
                nextSubTriangle(&subTria);
                unsigned subTriangleLength = triangleLength(subTria);
                if (subTriangleLength <= *remainder) {
                    *remainder -= subTriangleLength;
                } else {
                    triangleStack.push_back(subTria);
                    break;
                }
            }
        }

        inline void nextSublevel()
        {
            if (sublevelState == TRIVIAL) {
                nextTrivial();
            } else {
                nextCached();
            }
        }

        inline void nextTrivial()
        {
            if (trivialTriangleCounter++ == 0) {
                if (trivialTriangleType == 0) {
                    cursor += Coord<2>(0, -1);
                }
                // nothing to do for type 2 as then origin and first
                // coord match. types 1 and 3 don't occur here.
            } else {
                cursor += trivialTriangleDirection;
            }
        }

        inline void nextCached()
        {
            cursor = cachedTriangleOrigin + *(cachedTriangleCoordsIterator++);
        }

        inline bool sublevelTriangleFinished() const
        {
            if (sublevelState == TRIVIAL) {
                return trivialTriangleFinished();
            } else {
                return cachedTriangleFinished();
            }
        }

        inline bool trivialTriangleFinished() const
        {
            // trivial triangles of type 1 and 3 don't contain any
            // coordinates as they only consist of one "diagonal" (and
            // this is dominated by type 0 and 2 triangles)
            return (trivialTriangleType % 2) || (trivialTriangleCounter >= trivialTriangleLength);
        }

        inline bool cachedTriangleFinished() const
        {
            return cachedTriangleCoordsIterator >= cachedTriangleCoordsEnd;
        }

        static inline void nextSubTriangle(Triangle *triangle)
        {
            newOriginAndDimensions(&triangle->origin, &triangle->dimensions, triangle->type, triangle->counter);
            triangle->type = triangleTransitions[triangle->type][triangle->counter];
            triangle->counter = 0;
        }

        // sub-origins/-dimensions:
        // (dimensions (x,y))
        //
        // type counter origin         dimensions
        // ---- ------- -------------- -----------------------
        // 0    0       (0, 0)
        //      1       (leftHalf, -lowerHalf)
        //      2       (0, -y)
        //      3       (leftHalf, -lowerHalf)
        //      end     (x, -y)
        // ---- ------- -------------- -----------------------
        // 1    0       (0, 0)
        //      1       (-rightHalf, -lowerHalf)
        //      2       (-x, 0)
        //      3       (-rightHalf, -lowerHalf)
        //      end     (-x, -y)
        // ---- ------- -------------- -----------------------
        // 2    0       (0, 0)
        //      1       (leftHalf, upperHalf)
        //      2       (x, 0)
        //      3       (leftHalf, upperHalf)
        //      end     (x, y)
        // ---- ------- -------------- -----------------------
        // 3    0       (0, 0)
        //      1       (-rightHalf, upperHalf)
        //      2       (0, y)
        //      3       (-rightHalf, upperHalf)
        //      end     (-x, y)
        static inline void newOriginAndDimensions(
            Coord<2> *curOri,
            Coord<2> *curDim,
            unsigned curType,
            unsigned curCounter)
        {
            int leftHalf =  curDim->x() / 2;
            int rightHalf = curDim->x() - leftHalf;
            int upperHalf = curDim->y() / 2;
            int lowerHalf = curDim->y() - upperHalf;
            Coord<2> offset0 = newOrigin(curType, curCounter,     leftHalf, rightHalf, upperHalf, lowerHalf, curDim->x(), curDim->y());
            Coord<2> offset1 = newOrigin(curType, curCounter + 1, leftHalf, rightHalf, upperHalf, lowerHalf, curDim->x(), curDim->y());
            *curOri += offset0;
            Coord<2> dim = offset1 - offset0;
            *curDim = Coord<2>(std::abs(dim.x()), std::abs(dim.y()));
        }

        static inline Coord<2> newOrigin(
            unsigned curType,
            unsigned curCounter,
            int leftHalf,
            int rightHalf,
            int upperHalf,
            int lowerHalf,
            int x,
            int y)
        {
            switch (curCounter) {
            case 0:
                return Coord<2>(0, 0);
            case 1:
            case 3:
                switch (curType) {
                case 0u:
                    return Coord<2>(leftHalf,   -lowerHalf);
                case 1u:
                    return Coord<2>(-rightHalf, -lowerHalf);
                case 2u:
                    return Coord<2>(leftHalf,   upperHalf);
                case 3u:
                    return Coord<2>(-rightHalf, upperHalf);
                }
            case 2:
                switch (curType) {
                case 0u:
                    return Coord<2>(0, -y);
                case 1u:
                    return Coord<2>(-x, 0);
                case 2u:
                    return Coord<2>(x, 0);
                case 3u:
                    return Coord<2>(0, y);
                }
            case 4: // end:
                switch (curType) {
                case 0u:
                    return Coord<2>( x, -y);
                case 1u:
                    return Coord<2>(-x, -y);
                case 2u:
                    return Coord<2>( x,  y);
                case 3u:
                    return Coord<2>(-x,  y);
                }
            }

            throw std::invalid_argument("bad curType or curCounter");
        }

        static inline Coord<2> subtriangleDimensions(const Coord<2>& dimensions, unsigned type)
        {
            Triangle t(type, dimensions);
            nextSubTriangle(&t);
            return t.dimensions;
        }

        static inline unsigned triangleLength(const Triangle& triangle)
        {
            return triangleLength(triangle.dimensions, triangle.type);
        }

        static inline unsigned triangleLength(const Coord<2>& dimensions, unsigned type)
        {
            if (hasTrivialDimensions(dimensions))
            {
                if (type == 1 || type == 3) {
                    return 0;
                } else {
                    return static_cast<std::size_t>(dimensions.prod());
                }
            } else {
                std::pair<Coord<2>, unsigned> key(dimensions, type);
                // result already cached?
                if (triangleLengthCache.count(key)) {
                    return triangleLengthCache[key];
                }

                unsigned ret = 0;
                Triangle t(type, dimensions);
                for (t.counter = 0; t.counter < 4; ++t.counter) {
                    Triangle nextTria = t;
                    nextSubTriangle(&nextTria);
                    ret += triangleLength(nextTria);
                }
                // speed-up subsequent queries by caching the result
                triangleLengthCache[key] = ret;
                return ret;
            }
        }

        inline bool isCached(const Coord<2>& dimensions) const
        {
            return dimensions.x() < maxCachedDimensions.x() && dimensions.y() < maxCachedDimensions.y();
        }
    };

    inline explicit HIndexingPartition(
        const Coord<2>& origin=Coord<2>(0, 0),
        const Coord<2>& dimensions=Coord<2>(0, 0),
        const std::size_t offset=0,
        const std::vector<std::size_t>& weights=std::vector<std::size_t>(2)) :
        SpaceFillingCurve<2>(offset, weights),
        origin(origin),
        dimensions(dimensions)
    {}

    inline Iterator begin() const
    {
        return Iterator(origin, dimensions);
    }

    inline Iterator end() const
    {
        return Iterator(origin);
    }

    inline Region<2> getRegion(const std::size_t node) const
    {
        return Region<2>(
            (*this)[startOffsets[node + 0]],
            (*this)[startOffsets[node + 1]]);
    }

    inline Iterator operator[](unsigned pos) const
    {
        return Iterator(origin, dimensions, pos);
    }

private:
    using SpaceFillingCurve<2>::startOffsets;

    Coord<2> origin;
    Coord<2> dimensions;

    static inline bool fillCaches()
    {
        // store triangles of at most maxDim in size
        Coord<2> maxDim(17, 17);
        triangleCoordsCache.reset(new CacheType(Coord<3>(maxDim.x(), maxDim.y(), 4)));

        for (int y = 2; y < maxDim.y(); ++y) {
            maxCachedDimensions = Coord<2>(y, y);
            for (int x = 2; x < maxDim.x(); ++x) {
                Coord<2> dimensions(x, y);
                for (unsigned t = 0; t < 4; ++t) {
                    CoordVector coords;
                    Iterator end(Coord<2>(0, 0));

                    for (Iterator h(Coord<2>(0, 0), t, dimensions); h != end; ++h) {
                        coords.push_back(*h);
                    }

                    Coord<3> c(dimensions.x(), dimensions.y(), static_cast<int>(t));
                    (*triangleCoordsCache)[c] = coords;
                }
            }
        }

        maxCachedDimensions = maxDim;
        return true;
    }
};

template<typename _CharT, typename _Traits>
std::basic_ostream<_CharT, _Traits>&
operator<<(std::basic_ostream<_CharT, _Traits>& __os,
           const HIndexingPartition::Triangle& tria)
{
    __os << tria.toString();
    return __os;
}

#ifdef _MSC_BUILD
#pragma warning( pop )
#endif

}

#endif
