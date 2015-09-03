#ifndef LIBGEODECOMP_GEOMETRY_REGION_H
#define LIBGEODECOMP_GEOMETRY_REGION_H

#include <libgeodecomp/geometry/adjacency.h>
#include <libgeodecomp/geometry/coordbox.h>
#include <libgeodecomp/geometry/regionstreakiterator.h>
#include <libgeodecomp/geometry/streak.h>
#include <libgeodecomp/geometry/topologies.h>
#include <libgeodecomp/misc/stdcontaineroverloads.h>

namespace LibGeoDecomp {

/**
 * Unit test class
 */
class RegionTest;

namespace RegionHelpers {

/**
 * internal helper class
 */
class RegionCommonHelper
{
public:
    static inline bool pairCompareFirst(const std::pair<int, int>& a, const std::pair<int, int>& b)
    {
        return a.first < b.first;
    }

    static inline bool pairCompareSecond(const std::pair<int, int>& a, const std::pair<int, int>& b)
    {
        return a.second < b.second;
    }

protected:
    typedef std::pair<int, int> IntPair;
    typedef std::vector<IntPair> IndexVectorType;

    inline void incRemainder(const IndexVectorType::iterator& start, const IndexVectorType::iterator& end, const int& inserts)
    {
        if (inserts == 0) {
            return;
        }

        for (IndexVectorType::iterator incrementer = start;
             incrementer != end; ++incrementer) {
            incrementer->second += inserts;
        }
    }
};

/**
 * internal helper class
 */
template<int DIM>
class ConstructStreakFromIterators
{
public:
    typedef std::pair<int, int> IntPair;
    typedef std::vector<IntPair> IndexVectorType;

    template<int STREAK_DIM>
    inline void operator()(Streak<STREAK_DIM> *streak, IndexVectorType::const_iterator *iterators, const Coord<STREAK_DIM>& offset)
    {
        ConstructStreakFromIterators<DIM - 1>()(streak, iterators, offset);
        streak->origin[DIM] = iterators[DIM]->first + offset[DIM];
    }
};

/**
 * internal helper class
 */
template<>
class ConstructStreakFromIterators<0>
{
public:
    typedef std::pair<int, int> IntPair;
    typedef std::vector<IntPair> IndexVectorType;

    template<int STREAK_DIM>
    inline void operator()(Streak<STREAK_DIM> *streak, IndexVectorType::const_iterator *iterators, const Coord<STREAK_DIM>& offset)
    {
        streak->origin[0] = iterators[0]->first  + offset[0];
        streak->endX      = iterators[0]->second + offset[0];
    }
};

/**
 * internal helper class
 */
template<int DIM>
class StreakIteratorInitSingleOffset
{
public:
    typedef std::pair<int, int> IntPair;
    typedef std::vector<IntPair> IndexVectorType;

    explicit StreakIteratorInitSingleOffset(const std::size_t& offsetIndex) :
        offsetIndex(offsetIndex)
    {}

    template<int STREAK_DIM, typename REGION>
    inline std::size_t operator()(Streak<STREAK_DIM> *streak, IndexVectorType::const_iterator *iterators, const REGION& region) const
    {
        StreakIteratorInitSingleOffset<DIM - 1> delegate(offsetIndex);
        std::size_t newOffset = delegate(streak, iterators, region);

        IndexVectorType::const_iterator upperBound =
            std::upper_bound(region.indicesBegin(DIM),
                             region.indicesEnd(DIM),
                             IntPair(0, newOffset),
                             RegionHelpers::RegionCommonHelper::pairCompareSecond);
        iterators[DIM] = upperBound - 1;
        newOffset =  iterators[DIM] - region.indicesBegin(DIM);

        return newOffset;
    }

private:
    const std::size_t& offsetIndex;
};

/**
 * internal helper class
 */
template<>
class StreakIteratorInitSingleOffset<0>
{
public:
    typedef std::pair<int, int> IntPair;
    typedef std::vector<IntPair> IndexVectorType;

    explicit StreakIteratorInitSingleOffset(const std::size_t& offsetIndex) :
        offsetIndex(offsetIndex)
    {}

    template<int STREAK_DIM, typename REGION>
    inline std::size_t operator()(Streak<STREAK_DIM> *streak, IndexVectorType::const_iterator *iterators, const REGION& region) const
    {
        iterators[0] = region.indicesBegin(0) + offsetIndex;
        return offsetIndex;
    }

private:
    const std::size_t& offsetIndex;
};

/**
 * internal helper class
 */
template<int DIM>
class StreakIteratorInitSingleOffsetWrapper
{
public:
    typedef std::pair<int, int> IntPair;
    typedef std::vector<IntPair> IndexVectorType;

    explicit StreakIteratorInitSingleOffsetWrapper(const std::size_t& offsetIndex) :
        offsetIndex(offsetIndex)
    {}

    template<int STREAK_DIM, typename REGION>
    inline void operator()(Streak<STREAK_DIM> *streak, IndexVectorType::const_iterator *iterators, const REGION& region, const Coord<STREAK_DIM>& offset) const
    {
        StreakIteratorInitSingleOffset<DIM> delegate(offsetIndex);
        delegate(streak, iterators, region);
        ConstructStreakFromIterators<DIM>()(streak, iterators, offset);
    }

private:
    const std::size_t& offsetIndex;
};

/**
 * internal helper class
 */
template<int DIM, int COORD_DIM>
class StreakIteratorInitOffsets
{
public:
    typedef std::pair<int, int> IntPair;
    typedef std::vector<IntPair> IndexVectorType;

    explicit StreakIteratorInitOffsets(const Coord<COORD_DIM>& offsets) :
        offsets(offsets)
    {}

    template<int STREAK_DIM, typename REGION>
    inline void operator()(Streak<STREAK_DIM> *streak, IndexVectorType::const_iterator *iterators, const REGION& region, const Coord<STREAK_DIM>& offset) const
    {
        iterators[DIM] = region.indicesBegin(DIM) + offsets[DIM];

        StreakIteratorInitOffsets<DIM - 1, COORD_DIM> delegate(offsets);
        delegate(streak, iterators, region, offset);
    }

private:
    const Coord<COORD_DIM>& offsets;
};

/**
 * internal helper class
 */
template<int COORD_DIM>
class StreakIteratorInitOffsets<0, COORD_DIM>
{
public:
    typedef std::pair<int, int> IntPair;
    typedef std::vector<IntPair> IndexVectorType;

    explicit StreakIteratorInitOffsets(const Coord<COORD_DIM>& offsets) :
        offsets(offsets)
    {}

    template<int STREAK_DIM, typename REGION>
    inline void operator()(Streak<STREAK_DIM> *streak, IndexVectorType::const_iterator *iterators, const REGION& region, const Coord<STREAK_DIM>& offset) const
    {
        iterators[0] = region.indicesBegin(0) + offsets[0];

        if (int(region.indicesSize(0)) > offsets[0]) {
            ConstructStreakFromIterators<STREAK_DIM - 1>()(streak, iterators, offset);
        }
    }

private:
    const Coord<COORD_DIM>& offsets;
};

/**
 * internal helper class
 */
template<int DIM>
class StreakIteratorInitBegin
{
public:
    typedef std::pair<int, int> IntPair;
    typedef std::vector<IntPair> IndexVectorType;

    template<int STREAK_DIM, typename REGION>
    inline void operator()(Streak<STREAK_DIM> *streak, IndexVectorType::const_iterator *iterators, const REGION& region, const Coord<STREAK_DIM>& offset) const
    {
        iterators[DIM] = region.indicesBegin(DIM);
        StreakIteratorInitBegin<DIM - 1>()(streak, iterators, region, offset);
    }
};

/**
 * internal helper class
 */
template<>
class StreakIteratorInitBegin<0>
{
public:
    typedef std::pair<int, int> IntPair;
    typedef std::vector<IntPair> IndexVectorType;

    template<int STREAK_DIM, typename REGION>
    inline void operator()(Streak<STREAK_DIM> *streak, IndexVectorType::const_iterator *iterators, const REGION& region, const Coord<STREAK_DIM>& offset) const
    {
        iterators[0] = region.indicesBegin(0);

        if (region.indicesSize(0) > 0) {
            ConstructStreakFromIterators<STREAK_DIM - 1>()(streak, iterators, offset);
        }
    }
};

/**
 * internal helper class
 */
template<int DIM>
class StreakIteratorInitEnd
{
public:
    typedef std::pair<int, int> IntPair;
    typedef std::vector<IntPair> IndexVectorType;

    template<int STREAK_DIM, typename REGION>
    inline void operator()(Streak<STREAK_DIM> *streak, IndexVectorType::const_iterator *iterators, const REGION& region, const Coord<STREAK_DIM>& offset) const
    {
        StreakIteratorInitEnd<DIM - 1>()(streak, iterators, region, offset);
        iterators[DIM] = region.indicesEnd(DIM);
    }
};

/**
 * internal helper class
 */
template<>
class StreakIteratorInitEnd<0>
{
public:
    typedef std::pair<int, int> IntPair;
    typedef std::vector<IntPair> IndexVectorType;

    template<int STREAK_DIM, typename REGION>
    inline void operator()(Streak<STREAK_DIM> *streak, IndexVectorType::const_iterator *iterators, const REGION& region, const Coord<STREAK_DIM>& offset) const
    {
        iterators[0] = region.indicesEnd(0);
    }
};

/**
 * internal helper class
 */
template<int DIM>
class RegionIntersectHelper
{
public:
    template<int MY_DIM>
    static inline bool intersects(const Streak<MY_DIM>& s1, const Streak<MY_DIM>& s2)
    {
        if (s1.origin[DIM] != s2.origin[DIM]) {
            return false;
        }

        return RegionIntersectHelper<DIM - 1>::intersects(s1, s2);
    }

    template<int MY_DIM>
    static inline bool lessThan(const Streak<MY_DIM>& s1, const Streak<MY_DIM>& s2)
    {
        if (s1.origin[DIM] < s2.origin[DIM]) {
            return true;
        }
        if (s1.origin[DIM] > s2.origin[DIM]) {
            return false;
        }

        return RegionIntersectHelper<DIM - 1>::lessThan(s1, s2);
    }
};

/**
 * internal helper class
 */
template<>
class RegionIntersectHelper<0>
{
public:
    template<int MY_DIM>
    static inline bool intersects(const Streak<MY_DIM>& s1, const Streak<MY_DIM>& s2)
    {
        return ((s1.origin.x() <= s2.origin.x() && s2.origin.x() < s1.endX) ||
                (s2.origin.x() <= s1.origin.x() && s1.origin.x() < s2.endX));

    }

    template<int MY_DIM>
    static inline bool lessThan(const Streak<MY_DIM>& s1, const Streak<MY_DIM>& s2)
    {
        return s1.endX < s2.endX;
    }
};

/**
 * internal helper class
 */
template<int DIM>
class RegionLookupHelper;

/**
 * internal helper class
 */
template<int DIM>
class RegionInsertHelper;

/**
 * internal helper class
 */
template<int DIM>
class RegionRemoveHelper;

}

/**
 * Region stores a set of coordinates. It performs a run-length
 * coding. Instead of storing complete Streak objects, these objects
 * get split up and are stored implicitly in the hierarchical indices
 * vectors.
 */
template<int DIM>
class Region
{
public:
    friend class Serialization;

    template<int MY_DIM> friend void swap(Region<MY_DIM>&, Region<MY_DIM>&);
    template<int MY_DIM> friend class RegionHelpers::RegionLookupHelper;
    template<int MY_DIM> friend class RegionHelpers::RegionInsertHelper;
    template<int MY_DIM> friend class RegionHelpers::RegionRemoveHelper;
    friend class LibGeoDecomp::RegionTest;

    typedef std::pair<int, int> IntPair;
    typedef std::vector<IntPair> IndexVectorType;
    typedef RegionStreakIterator<DIM, Region<DIM> > StreakIterator;
    typedef Coord<DIM> vector_type;
    typedef CoordBox<DIM> cube_type;

    class Iterator : public std::iterator<std::forward_iterator_tag,
                                          const Coord<DIM> >
    {
    public:
        inline explicit Iterator(const StreakIterator& streakIterator) :
            streakIterator(streakIterator),
            cursor(streakIterator->origin)
        {}

        inline void operator++()
        {
            cursor.x()++;
            if (cursor.x() >= streakIterator->endX) {
                ++streakIterator;
                cursor = streakIterator->origin;
            }
        }

        inline bool operator==(const Iterator& other) const
        {
            return streakIterator == other.streakIterator &&
                (streakIterator.endReached() || cursor == other.cursor);
        }

        inline bool operator!=(const Iterator& other) const
        {
            return !(*this == other);
        }

        inline const Coord<DIM>& operator*() const
        {
            return cursor;
        }

        inline const Coord<DIM> *operator->() const
        {
            return &cursor;
        }

    private:
        StreakIterator streakIterator;
        Coord<DIM> cursor;
    };

    inline Region() :
        mySize(0),
        geometryCacheTainted(false)
    {}

#ifdef LIBGEODECOMP_WITH_CPP14
    inline Region(const Region<DIM>& other) = default;

    inline Region(Region<DIM>&& other) :
        myBoundingBox(other.myBoundingBox),
        mySize(other.mySize),
        geometryCacheTainted(other.geometryCacheTainted)
    {
        for (int i = 0; i < DIM; ++i) {
            indices[i] = std::move(other.indices[i]);
        }
    }

    inline Region& operator=(const Region<DIM>& other) = default;

    inline Region& operator=(Region<DIM>&& other)
    {
        using std::swap;
        swap(*this, other);
        return *this;
    }
#endif

    template<class ITERATOR1, class ITERATOR2>
    inline Region(const ITERATOR1& start, const ITERATOR2& end) :
        mySize(0),
        geometryCacheTainted(false)
    {
        load(start, end);
    }

    template<class ITERATOR1, class ITERATOR2>
    inline void load(const ITERATOR1& start, const ITERATOR2& end)
    {
        for (ITERATOR1 i = start; i != end; ++i) {
            *this << *i;
        }
    }

    inline void clear()
    {
        for (int i = 0; i < DIM; ++i) {
            indices[i].clear();
        }
        mySize = 0;
        myBoundingBox = CoordBox<DIM>();
        geometryCacheTainted = false;
    }

    inline const CoordBox<DIM>& boundingBox() const
    {
        if (geometryCacheTainted) {
            resetGeometryCache();
        }
        return myBoundingBox;
    }

    inline std::size_t size() const
    {
        if (geometryCacheTainted) {
            resetGeometryCache();
        }
        return mySize;
    }

    inline const Coord<DIM>& dimension() const
    {
        return boundingBox().dimensions;
    }

    inline std::size_t numStreaks() const
    {
        return indices[0].size();
    }

    inline Region expand(const unsigned& width=1) const
    {
        return expand(Coord<DIM>::diagonal(width));
    }

    /**
     * Expands the region in each dimension d by radii[d] cells.
     */
    inline Region expand(const Coord<DIM>& radii) const
    {
        using std::swap;
        Region accumulator;
        Region buffer;

        // expansion in X dimension is a simple 1-pass operation:
        for (StreakIterator i = beginStreak(); i != endStreak(); ++i) {
            Streak<DIM> streak = *i;
            streak.origin[0] -= radii[0];
            streak.endX += radii[0];
            accumulator << streak;
        }

        // expand into other dimensions, one after another
        for (int d = 1; d < DIM; ++d) {
            expandInOneDimension(d, radii[d], accumulator, buffer);
        }

        return accumulator;
    }

    /**
     * does the same as expand, but will wrap overlap at edges
     * correctly. The instance of the TOPOLOGY is actually unused, but
     * without it g++ would complain...
     */
    template<typename TOPOLOGY>
    inline Region expandWithTopology(
        const unsigned& width,
        const Coord<DIM>& globalDimensions,
        TOPOLOGY /* unused */) const
    {
        Coord<DIM> dia = Coord<DIM>::diagonal(width);
        Region buffer = expand(dia);
        Region ret;

        for (StreakIterator i = buffer.beginStreak(); i != buffer.endStreak(); ++i) {
            Streak<DIM> streak = *i;
            if (TOPOLOGY::template WrapsAxis<0>::VALUE) {
                splitStreak<TOPOLOGY>(streak, &ret, globalDimensions);
            } else {
                normalizeStreak<TOPOLOGY>(
                    trimStreak(streak, globalDimensions), &ret, globalDimensions);
            }
        }

        return ret;
    }

#ifdef LIBGEODECOMP_WITH_CPP14
    template<typename TOPOLOGY>
    inline Region expandWithTopology(
        const unsigned& width,
        const Coord<DIM>& globalDimensions,
        TOPOLOGY topology,
        const Adjacency& adjacency) const
    {
        return expandWithTopology(width, globalDimensions, topology);
    }

    inline Region expandWithTopology(
        const unsigned& width,
        const Coord<DIM>& /* unused: globalDimensions */,
        Topologies::Unstructured /* used just for overload */,
        const Adjacency& adjacency) const
    {
        return expandWithAdjacency(width, adjacency);
    }

    /**
     * does the same as expand, but reads adjacent indices out of
     * an adjacency list
     */
    inline Region expandWithAdjacency(
        const unsigned& width,
        const Adjacency& adjacency) const
    {
        static_assert(DIM == 1, "expanding with adjacency only works on unstructured, i.e. 1-dimensional grids.");

        Region ret = *this;
        Region newCoords = *this;

        for (unsigned pass = 0; pass < width; ++pass) {
            Region add;

            // walk over all indices and remember adjacent neighbors
            // this is done in a separate pass to ensure that
            // containers are changed while iterating them.
            for (const Coord<1> index : newCoords) {
                auto it = adjacency.find(index.x());
                if (it != adjacency.end()) {
                    for (auto&& i: it->second) {
                        add << Coord<DIM>(i);
                    }
                }
            }

            ret += add;
            using std::swap;
            swap(add, newCoords);
        }

        return ret;
    }
#endif // LIBGEODECOMP_WITH_CPP14

    inline bool operator==(const Region<DIM>& other) const
    {
        for (int i = 0; i < DIM; ++i) {
            if (indices[i] != other.indices[i]) {
                return false;
            }
        }

        return true;
    }

    /**
     * Checks whether the Region includes the given Streak.
     */
    bool count(const Streak<DIM>& s) const
    {
        return RegionHelpers::RegionLookupHelper<DIM - 1>()(*this, s);
    }

    /**
     * Is the Coord contained withing the Region?
     */
    bool count(const Coord<DIM>& c) const
    {
        return RegionHelpers::RegionLookupHelper<DIM - 1>()(*this, Streak<DIM>(c, c[0] + 1));
    }

    /**
     * Alias for operator<<
     */
    template<typename ADDEND>
    inline void insert(const ADDEND& a)
    {
        *this << a;
    }

    /**
     * Add all coordinates of the Streak to this Region
     */
    inline Region& operator<<(const Streak<DIM>& s)
    {
        //ignore 0 length streaks
        if (s.endX <= s.origin.x()) {
            return *this;
        }

        if (count(s) == 0) {
            return *this;
        }

        geometryCacheTainted = true;
        RegionHelpers::RegionInsertHelper<DIM - 1>()(this, s);
        return *this;
    }

    inline Region& operator<<(const Coord<DIM>& c)
    {
        *this << Streak<DIM>(c, c.x() + 1);
        return *this;
    }

    inline Region& operator<<(const CoordBox<DIM>& box)
    {
        for (typename CoordBox<DIM>::StreakIterator i = box.beginStreak(); i != box.endStreak(); ++i) {
            *this << *i;
        }

        return *this;
    }

    /**
     * Remove the given Streak (or all of its coordinates) from the
     * Region. Takes into account that not all (or none) of the
     * coordinates may have been contained in the Region before.
     */
    inline Region& operator>>(const Streak<DIM>& s)
    {
        //ignore 0 length streaks and empty selves
        if (s.endX <= s.origin.x() || empty()) {
            return *this;
        }

        geometryCacheTainted = true;
        RegionHelpers::RegionRemoveHelper<DIM - 1>()(this, s);
        return *this;
    }

    inline Region& operator>>(const Coord<DIM>& c)
    {
        *this >> Streak<DIM>(c, c.x() + 1);
        return *this;
    }

    inline Region& operator>>(const CoordBox<DIM>& box)
    {
        for (typename CoordBox<DIM>::StreakIterator i = box.beginStreak(); i != box.endStreak(); ++i) {
            *this >> *i;
        }

        return *this;
    }

    inline void operator-=(const Region& other)
    {
        Region newValue = *this - other;
        *this = newValue;
    }

    /**
     * Equvalent to (A and (not B)) in sets, where other corresponds
     * to B and *this corresponds to A.
     */
    inline Region operator-(const Region& other) const
    {
        using std::max;
        using std::min;
        Region ret;
        // these conditionals are less a shortcut but more a guarantee
        // that the derefernce below will succeed:
        if (this->empty()) {
            return ret;
        }
        if (other.empty()) {
            return *this;
        }

        StreakIterator myIter = beginStreak();
        StreakIterator otherIter = other.beginStreak();

        StreakIterator myEnd = endStreak();
        StreakIterator otherEnd = other.endStreak();

        Streak<DIM> cursor = *myIter;

        for (;;) {
            if (RegionHelpers::RegionIntersectHelper<DIM - 1>::intersects(cursor, *otherIter)) {
                int intersectionOriginX = max(cursor.origin.x(), otherIter->origin.x());
                int intersectionEndX = min(cursor.endX, otherIter->endX);

                ret << Streak<DIM>(cursor.origin, intersectionOriginX);
                cursor.origin.x() = intersectionEndX;
            }

            if (RegionHelpers::RegionIntersectHelper<DIM - 1>::lessThan(cursor, *otherIter)) {
                ret << cursor;
                ++myIter;

                if (myIter == myEnd) {
                    break;
                } else {
                    cursor = *myIter;
                }
            } else {
                ++otherIter;
                if (otherIter == otherEnd) {
                    break;
                }
            }
        }

        // don't loose the remainder
        ret << cursor;
        if (myIter != myEnd) {
            ++myIter;
            for (; myIter != myEnd; ++myIter) {
                ret << *myIter;
            }
        }

        return ret;
    }

    inline void operator&=(const Region& other)
    {
        Region intersection = other & *this;
        *this = intersection;
    }

    /**
     * Computes the intersection of both regions.
     */
    inline Region operator&(const Region& other) const
    {
        using std::max;
        using std::min;
        Region ret;
        StreakIterator myIter = beginStreak();
        StreakIterator otherIter = other.beginStreak();

        StreakIterator myEnd = endStreak();
        StreakIterator otherEnd = other.endStreak();

        for (;;) {
            if ((myIter == myEnd) ||
                (otherIter == otherEnd)) {
                break;
            }

            if (RegionHelpers::RegionIntersectHelper<DIM - 1>::intersects(*myIter, *otherIter)) {
                Streak<DIM> intersection = *myIter;
                intersection.origin.x() = max(myIter->origin.x(), otherIter->origin.x());
                intersection.endX = min(myIter->endX, otherIter->endX);
                ret << intersection;
            }

            if (RegionHelpers::RegionIntersectHelper<DIM - 1>::lessThan(*myIter, *otherIter)) {
                ++myIter;
            } else {
                ++otherIter;
            }
        }

        return ret;
    }

    inline void operator+=(const Region& other)
    {
        Region newValue = *this + other;
        *this = newValue;
    }

#define LIBGEODECOMP_REGION_ADVANCE_ITERATOR(ITERATOR, END)            \
            if (*ITERATOR != lastInsert) {         \
                ret << *ITERATOR;                  \
                lastInsert = *ITERATOR;            \
            }                                      \
            ++ITERATOR;                            \
            if (ITERATOR == END) {                 \
                break;                             \
            }

    inline static void merge2way(
        Region& ret,
        const StreakIterator& beginA, const StreakIterator& endA,
        const StreakIterator& beginB, const StreakIterator& endB)
    {
        if (beginA == endA) {
            for (StreakIterator i = beginB; i != endB; ++i) {
                ret << *i;
            }
            return;
        }
        if (beginB == endB) {
            for (StreakIterator i = beginA; i != endA; ++i) {
                ret << *i;
            }
            return;
        }

        StreakIterator iterA = beginA;
        StreakIterator iterB = beginB;
        Streak<DIM> lastInsert;

        for (;;) {
            if (RegionHelpers::RegionIntersectHelper<DIM - 1>::lessThan(*iterA, *iterB)) {
                LIBGEODECOMP_REGION_ADVANCE_ITERATOR(iterA, endA);
            } else {
                LIBGEODECOMP_REGION_ADVANCE_ITERATOR(iterB, endB);
            }
        }

        for (; iterA != endA; ++iterA) {
            ret << *iterA;
        }
        for (; iterB != endB; ++iterB) {
            ret << *iterB;
        }
    }

    inline static void merge3way(
        Region& ret,
        const StreakIterator& beginA, const StreakIterator& endA,
        const StreakIterator& beginB, const StreakIterator& endB,
        const StreakIterator& beginC, const StreakIterator& endC)
    {
        StreakIterator iterA = beginA;
        StreakIterator iterB = beginB;
        StreakIterator iterC = beginC;

        if (iterA == endA) {
            merge2way(
                ret,
                iterB, endB,
                iterC, endC);
            return;
        }

        if (iterB == endB) {
            merge2way(
                ret,
                iterA, endA,
                iterC, endC);
            return;
        }

        if (iterC == endC) {
            merge2way(
                ret,
                iterA, endA,
                iterB, endB);
            return;
        }

        Streak<DIM> lastInsert;

        for (;;) {
            if (RegionHelpers::RegionIntersectHelper<DIM - 1>::lessThan(*iterA, *iterB)) {
                if (RegionHelpers::RegionIntersectHelper<DIM - 1>::lessThan(*iterA, *iterC)) {
                    LIBGEODECOMP_REGION_ADVANCE_ITERATOR(iterA, endA);
                } else {
                    LIBGEODECOMP_REGION_ADVANCE_ITERATOR(iterC, endC);
                }
            } else {
                if (RegionHelpers::RegionIntersectHelper<DIM - 1>::lessThan(*iterB, *iterC)) {
                    LIBGEODECOMP_REGION_ADVANCE_ITERATOR(iterB, endB);
                } else {
                    LIBGEODECOMP_REGION_ADVANCE_ITERATOR(iterC, endC);
                }
            }
        }

        if (iterA == endA) {
            merge2way(
                ret,
                iterB, endB,
                iterC, endC);
            return;
        }

        if (iterB == endB) {
            merge2way(
                ret,
                iterA, endA,
                iterC, endC);
            return;
        }

        if (iterC == endC) {
            merge2way(
                ret,
                iterA, endA,
                iterB, endB);
            return;
        }
    }

#undef LIBGEODECOMP_REGION_ADVANCE_ITERATOR

    inline Region operator+(const Region& other) const
    {
        Region ret;

        merge2way(
            ret,
            this->beginStreak(), this->endStreak(),
            other.beginStreak(), other.endStreak());

        return ret;
    }

    inline std::vector<Streak<DIM> > toVector() const
    {
        std::vector<Streak<DIM> > ret(numStreaks());
        std::copy(beginStreak(), endStreak(), ret.begin());
        return ret;
    }

    inline std::string toString() const
    {
        std::ostringstream buf;
        buf << "Region<" << DIM << ">(\n";
        for (int dim = 0; dim < DIM; ++dim) {
            buf << "  indices[" << dim << "] = "
                << indices[dim] << "\n";
        }
        buf << ")\n";

        return buf.str();

    }

    inline std::string prettyPrint() const
    {
        std::ostringstream buf;
        buf << "Region<" << DIM << ">(\n";

        for (StreakIterator i = beginStreak(); i != endStreak(); ++i) {
            buf << "  " << *i << "\n";
        }

        buf << ")\n";

        return buf.str();
    }

    inline bool empty() const
    {
        return (indices[0].size() == 0);
    }

    inline StreakIterator beginStreak(const Coord<DIM>& offset = Coord<DIM>()) const
    {
        return StreakIterator(this, RegionHelpers::StreakIteratorInitBegin<DIM - 1>(), offset);
    }

    inline StreakIterator endStreak(const Coord<DIM>& offset = Coord<DIM>()) const
    {
        return StreakIterator(this, RegionHelpers::StreakIteratorInitEnd<DIM - 1>(), offset);
    }

    /**
     * Returns an iterator whose internal iterators are set to the
     * given offsets from the corresponding array starts. Runs in O(1)
     * time.
     */
    inline StreakIterator operator[](const Coord<DIM>& offsets) const
    {
        return StreakIterator(this, RegionHelpers::StreakIteratorInitOffsets<DIM - 1, DIM>(offsets));
    }

    /**
     * Yields an iterator to the offset'th Streak in the Region. Runs
     * in O(log n) time.
     */
    inline StreakIterator operator[](std::size_t offset) const
    {
        if (offset == 0) {
            return beginStreak();
        }
        if (offset >= numStreaks()) {
            return endStreak();
        }
        return StreakIterator(this, RegionHelpers::StreakIteratorInitSingleOffsetWrapper<DIM - 1>(offset));
    }

    inline Iterator begin() const
    {
        return Iterator(beginStreak());
    }

    inline Iterator end() const
    {
        return Iterator(endStreak());
    }

    inline std::size_t indicesSize(std::size_t dim) const
    {
        return indices[dim].size();
    }

    inline IndexVectorType::const_iterator indicesAt(std::size_t dim, std::size_t offset) const
    {
        return indices[dim].begin() + offset;
    }

    inline IndexVectorType::const_iterator indicesBegin(std::size_t dim) const
    {
        return indices[dim].begin();
    }

    inline IndexVectorType::const_iterator indicesEnd(std::size_t dim) const
    {
        return indices[dim].end();
    }

private:
    IndexVectorType indices[DIM];
    mutable CoordBox<DIM> myBoundingBox;
    mutable std::size_t mySize;
    mutable bool geometryCacheTainted;

    inline void determineGeometry() const
    {
        if (empty()) {
            mySize = 0;
            myBoundingBox = CoordBox<DIM>();
        } else {
            Streak<DIM> someStreak = *beginStreak();
            Coord<DIM> minCoord = someStreak.origin;
            Coord<DIM> maxCoord = someStreak.origin;

            mySize = 0;
            for (StreakIterator i = beginStreak();
                 i != endStreak(); ++i) {
                Coord<DIM> left = i->origin;
                Coord<DIM> right = i->origin;
                right.x() = i->endX - 1;

                minCoord = (minCoord.min)(left);
                maxCoord = (maxCoord.max)(right);
                mySize += i->endX - i->origin.x();
            }

            myBoundingBox =
                CoordBox<DIM>(minCoord, maxCoord - minCoord + Coord<DIM>::diagonal(1));
        }
    }

    inline void resetGeometryCache() const
    {
        determineGeometry();
        geometryCacheTainted = false;
    }

    inline Streak<DIM> trimStreak(
        const Streak<DIM>& s,
        const Coord<DIM>& dimensions) const
    {
        using std::max;
        using std::min;
        int width = dimensions.x();
        Streak<DIM> buf = s;
        buf.origin.x() = max(buf.origin.x(), 0);
        buf.endX = min(width, buf.endX);
        return buf;
    }

    template<typename TOPOLOGY>
    void splitStreak(
        const Streak<DIM>& streak,
        Region *target,
        const Coord<DIM>& dimensions) const
    {
        using std::min;
        int width = dimensions.x();

        int currentX = streak.origin.x();
        if (currentX < 0) {
            Streak<DIM> section = streak;
            section.endX = min(streak.endX, 0);
            currentX = section.endX;

            // normalize left overhang
            section.origin.x() += width;
            section.endX += width;
            normalizeStreak<TOPOLOGY>(section, target, dimensions);
        }

        if (currentX < streak.endX) {
            Streak<DIM> section = streak;
            section.origin.x() = currentX;
            section.endX = min(streak.endX, width);
            currentX = section.endX;

            normalizeStreak<TOPOLOGY>(section, target, dimensions);
        }

        if (currentX < streak.endX) {
            Streak<DIM> section = streak;
            section.origin.x() = currentX;

            // normalize right overhang
            section.origin.x() -= width;
            section.endX -= width;
            normalizeStreak<TOPOLOGY>(section, target, dimensions);
        }
    }

    template<typename TOPOLOGY>
    void normalizeStreak(
        const Streak<DIM>& streak,
        Region *target,
        const Coord<DIM>& dimensions) const
    {
        Streak<DIM> ret;
        ret.origin = TOPOLOGY::normalize(streak.origin, dimensions);
        ret.endX = ret.origin.x() + streak.length();

        // it's bad to use a magic value to check for out of bounds
        // accesses, but throwing exceptions would be slower
        if (ret.origin != Coord<DIM>::diagonal(-1)) {
            (*target) << ret;
        }
    }

    static inline void expandInOneDimension(
        int dim, int radius, Region<DIM>& accumulator, Region<DIM>& buffer)
    {
        using std::swap;
        int targetWidth = 2 * radius + 1;
        int width = 1;

        for (; width < ((targetWidth + 2) / 3); width *= 3) {
            Coord<DIM> offset;
            offset[dim] = width;
            buffer.clear();

            merge3way(
                buffer,
                accumulator.beginStreak(-offset),
                accumulator.endStreak(-offset),
                accumulator.beginStreak(),
                accumulator.endStreak(),
                accumulator.beginStreak(offset),
                accumulator.endStreak(offset));
            swap(accumulator, buffer);
        }

        if (width < targetWidth) {
            Coord<DIM> finalOffset;
            finalOffset[dim] = radius - width / 2;
            buffer.clear();

            if ((width * 2) < targetWidth) {
                merge3way(
                    buffer,
                    accumulator.beginStreak(-finalOffset),
                    accumulator.endStreak(-finalOffset),
                    accumulator.beginStreak(),
                    accumulator.endStreak(),
                    accumulator.beginStreak(finalOffset),
                    accumulator.endStreak(finalOffset));
            } else {
                merge2way(
                    buffer,
                    accumulator.beginStreak(-finalOffset),
                    accumulator.endStreak(-finalOffset),
                    accumulator.beginStreak(finalOffset),
                    accumulator.endStreak(finalOffset));
            }
            swap(buffer, accumulator);
        }
    }
};

namespace RegionHelpers {

/**
 * internal helper class
 */
template<int DIM>
class RegionLookupHelper : public RegionCommonHelper
{
public:
    typedef Region<1>::IntPair IntPair;
    typedef Region<1>::IndexVectorType IndexVectorType;

    template<int MY_DIM>
    inline bool operator()(const Region<MY_DIM>& region, const Streak<MY_DIM>& s)
    {
        const IndexVectorType& indices = region.indices[DIM];
        return (*this)(region, s, 0, indices.size());
    }

    template<int MY_DIM>
    inline bool operator()(const Region<MY_DIM>& region, const Streak<MY_DIM>& s, const int& start, const int& end)
    {
        int c = s.origin[DIM];
        const IndexVectorType& indices = region.indices[DIM];

        IndexVectorType::const_iterator i =
            std::upper_bound(
                indices.begin() + start,
                indices.begin() + end,
                IntPair(c, 0),
                RegionCommonHelper::pairCompareFirst);

        int nextLevelStart = 0;
        int nextLevelEnd = 0;

        if (i != (indices.begin() + start)) {
            IndexVectorType::const_iterator entry = i;
            --entry;

            // recurse if found
            if (entry->first == c) {
                nextLevelStart = entry->second;
                nextLevelEnd = region.indices[DIM - 1].size();
                if (i != indices.end()) {
                    nextLevelEnd = i->second;
                }

                return RegionLookupHelper<DIM-1>()(
                    region,
                    s,
                    nextLevelStart,
                    nextLevelEnd);
            }
        }

        return false;
    }
};

/**
 * internal helper class
 */
template<>
class RegionLookupHelper<0> : public RegionCommonHelper
{
public:
    typedef Region<1>::IntPair IntPair;
    typedef Region<1>::IndexVectorType IndexVectorType;

    template<int MY_DIM>
    inline bool operator()(const Region<MY_DIM>& region, const Streak<MY_DIM>& s)
    {
        const IndexVectorType& indices = region.indices[0];
        return (*this)(region, s, 0, indices.size());
    }

    template<int MY_DIM>
    inline bool operator()(const Region<MY_DIM>& region, const Streak<MY_DIM>& s, const int& start, int end)
    {
        IntPair curStreak(s.origin.x(), s.endX);
        const IndexVectorType& indices = region.indices[0];
        if (indices.empty()) {
            return false;
        }

        IndexVectorType::const_iterator cursor =
            std::upper_bound(indices.begin() + start, indices.begin() + end,
                             curStreak, RegionCommonHelper::pairCompareFirst);
        // This will yield the streak AFTER the current origin
        // c. We can't really use lower_bound() as this doesn't
        // replace the < operator by >= but rather by <=, which is
        // IMO really sick...
        if (cursor != (indices.begin() + start)) {
            // ...so we revert to landing one past the streak we're
            // searching and moving back afterwards:
            cursor--;
        }

        return (cursor->first <= s.origin[0]) && (cursor->second >= s.endX);
    }

};

/**
 * internal helper class
 */
template<int DIM>
class RegionInsertHelper : public RegionCommonHelper
{
public:
    typedef Region<1>::IntPair IntPair;
    typedef Region<1>::IndexVectorType IndexVectorType;

    template<int MY_DIM>
    inline void operator()(Region<MY_DIM> *region, const Streak<MY_DIM>& s)
    {
        IndexVectorType& indices = region->indices[DIM];
        (*this)(region, s, 0, indices.size());
    }

    template<int MY_DIM>
    int operator()(Region<MY_DIM> *region, const Streak<MY_DIM>& s, const int& start, const int& end)
    {
        int c = s.origin[DIM];
        IndexVectorType& indices = region->indices[DIM];

        IndexVectorType::iterator i =
            std::upper_bound(
                indices.begin() + start,
                indices.begin() + end,
                IntPair(c, 0),
                RegionCommonHelper::pairCompareFirst);

        int nextLevelStart = 0;
        int nextLevelEnd = 0;

        if (i != (indices.begin() + start)) {
            IndexVectorType::iterator entry = i;
            --entry;

            // short-cut: no need to insert if index already present
            if (entry->first == c) {
                nextLevelStart = entry->second;
                nextLevelEnd = region->indices[DIM - 1].size();
                if (i != indices.end()) {
                    nextLevelEnd = i->second;
                }

                int inserts = RegionInsertHelper<DIM - 1>()(
                    region,
                    s,
                    nextLevelStart,
                    nextLevelEnd);
                incRemainder(i, indices.end(), inserts);
                return 0;
            }
        }

        if (i != indices.end()) {
            nextLevelStart = i->second;
        } else {
            nextLevelStart = region->indices[DIM - 1].size();
        }

        nextLevelEnd = nextLevelStart;

        IndexVectorType::iterator followingEntries;

        if (i == indices.end()) {
            indices << IntPair(c, nextLevelStart);
            followingEntries = indices.end();
        } else {
            followingEntries = indices.insert(i, IntPair(c, nextLevelStart));
            ++followingEntries;
        }

        int inserts = RegionInsertHelper<DIM - 1>()(region, s, nextLevelStart, nextLevelEnd);
        incRemainder(followingEntries, indices.end(), inserts);

        return 1;
    }
};

/**
 * internal helper class
 */
template<>
class RegionInsertHelper<0>
{
public:
    friend class LibGeoDecomp::RegionTest;
    typedef Region<1>::IntPair IntPair;
    typedef Region<1>::IndexVectorType IndexVectorType;

    template<int MY_DIM>
    inline void operator()(Region<MY_DIM> *region, const Streak<MY_DIM>& s)
    {
        IndexVectorType& indices = region->indices[0];
        (*this)(region, s, 0, indices.size());
    }

    template<int MY_DIM>
    inline int operator()(Region<MY_DIM> *region, const Streak<MY_DIM>& s, int start, int end)
    {
        IntPair curStreak(s.origin.x(), s.endX);
        IndexVectorType& indices = region->indices[0];

        IndexVectorType::iterator cursor =
            std::upper_bound(indices.begin() + start, indices.begin() + end,
                             curStreak, RegionCommonHelper::pairCompareFirst);
        // This will yield the streak AFTER the current origin
        // c. We can't really use lower_bound() as this doesn't
        // replace the < operator by >= but rather by <=, which is
        // IMO really sick...
        if (cursor != (indices.begin() + start)) {
            // ...so we revert to landing one past the streak we're
            // searching and moving back afterwards:
            cursor--;
        }

        int inserts = 1;

        while ((cursor != (indices.begin() + end)) &&
               (curStreak.second >= cursor->first)) {
            if (intersectOrTouch(*cursor, curStreak)) {
                curStreak = fuse(*cursor, curStreak);
                cursor = indices.erase(cursor);
                --end;
                --inserts;
            } else {
                cursor++;
            }

            if ((cursor == (indices.begin() + end)) ||
                (!intersectOrTouch(*cursor, curStreak))) {
                break;
            }
        }

        indices.insert(cursor, curStreak);
        return inserts;
    }

private:
    inline bool intersectOrTouch(const IntPair& a, const IntPair& b) const
    {
        return
            ((a.first <= b.first && b.first <= a.second) ||
             (b.first <= a.first && a.first <= b.second));
    }

    inline IntPair fuse(const IntPair& a, const IntPair& b) const
    {
        using std::min;
        using std::max;
        return IntPair(min(a.first,  b.first),
                       max(a.second, b.second));
    }
};

/**
 * internal helper class
 */
template<int DIM>
class RegionRemoveHelper : public RegionCommonHelper
{
public:
    typedef Region<1>::IntPair IntPair;
    typedef Region<1>::IndexVectorType IndexVectorType;

    template<int MY_DIM>
    inline void operator()(Region<MY_DIM> *region, const Streak<MY_DIM>& s)
    {
        IndexVectorType& indices = region->indices[DIM];
        (*this)(region, s, 0, indices.size());
    }

    /**
     * tries to remove a streak from the set. Returns the number of
     * inserted streaks (may be negative).
     */
    template<int MY_DIM>
    int operator()(Region<MY_DIM> *region, const Streak<MY_DIM>& s, const int& start, const int& end)
    {
        int c = s.origin[DIM];
        IndexVectorType& indices = region->indices[DIM];

        IndexVectorType::iterator i =
            std::upper_bound(
                indices.begin() + start,
                indices.begin() + end,
                IntPair(c, 0),
                RegionCommonHelper::pairCompareFirst);

        // key is not present, so no need to remove it
        if (i == (indices.begin() + start)) {
            return 0;
        }

        IndexVectorType::iterator entry = i;
        --entry;

        // ditto
        if (entry->first != c) {
            return 0;
        }

        int nextLevelStart = entry->second;
        int nextLevelEnd = region->indices[DIM - 1].size();
        if (i != indices.end()) {
            nextLevelEnd = i->second;
        }

        int inserts = RegionRemoveHelper<DIM - 1>()(
            region,
            s,
            nextLevelStart,
            nextLevelEnd);

        int myInserts = 0;

        // current entry needs to be removed if no childs are left
        if ((nextLevelStart - nextLevelEnd) == inserts) {
            entry = indices.erase(entry);
            myInserts = -1;
        } else {
            ++entry;
        }

        incRemainder(entry, indices.end(), inserts);
        return myInserts;
    }
};

/**
 * internal helper class
 */
template<>
class RegionRemoveHelper<0>
{
public:
    friend class LibGeoDecomp::RegionTest;
    typedef Region<1>::IntPair IntPair;
    typedef Region<1>::IndexVectorType IndexVectorType;

    template<int MY_DIM>
    inline void operator()(Region<MY_DIM> *region, const Streak<MY_DIM>& s)
    {
        IndexVectorType& indices = region->indices[0];
        (*this)(region, s, 0, indices.size());
    }

    template<int MY_DIM>
    int operator()(Region<MY_DIM> *region, const Streak<MY_DIM>& s, const int& start, int end)
    {
        int c = s.origin[0];
        IndexVectorType& indices = region->indices[0];
        int inserts = 0;

        // This will yield the streak AFTER the current origin
        // c. We can't really use lower_bound() as this doesn't
        // replace the < operator by >= but rather by <=, which is
        // IMO really sick...
        IndexVectorType::iterator cursor =
            std::upper_bound(
                indices.begin() + start,
                indices.begin() + end,
                IntPair(c, 0),
                RegionCommonHelper::pairCompareFirst);
        if (cursor != (indices.begin() + start)) {
            // ...so we resort to landing one past the streak we're
            // searching and moving back afterwards:
            --cursor;
        }

        IntPair curStreak(s.origin.x(), s.endX);

        while (cursor != (indices.begin() + end)) {
            if (intersect(curStreak, *cursor)) {
                IndexVectorType newStreaks(substract(*cursor, curStreak));
                cursor = indices.erase(cursor);
                int delta = newStreaks.size() - 1;
                end += delta;
                inserts += delta;

                for (IndexVectorType::iterator i = newStreaks.begin(); i != newStreaks.end(); ++i) {
                    cursor = indices.insert(cursor, *i);
                    ++cursor;
                }
            } else {
                ++cursor;
            }

            if (cursor == (indices.begin() + end) || !intersect(*cursor, curStreak)) {
                break;
            }
        }

        return inserts;
    }

private:
    inline bool intersect(const IntPair& a, const IntPair& b) const
    {
        return
            ((a.first <= b.first && b.first < a.second) ||
             (b.first <= a.first && a.first < b.second));
    }

    inline IndexVectorType substract(const IntPair& base, const IntPair& minuend) const
    {
        if (!intersect(base, minuend)) {
            return std::vector<IntPair>(1, base);
        }

        std::vector<IntPair> ret;
        IntPair s1(base.first, minuend.first);
        IntPair s2(minuend.second, base.second);

        if (s1.second > s1.first) {
            ret.push_back(s1);
        }
        if (s2.second > s2.first) {
            ret.push_back(s2);
        }
        return ret;
    }
};

}

template<int DIM>
inline void swap(Region<DIM>& regionA, Region<DIM>& regionB)
{
    using std::swap;
    swap(regionA.indices,              regionB.indices);
    swap(regionA.myBoundingBox,        regionB.myBoundingBox);
    swap(regionA.mySize,               regionB.mySize);
    swap(regionA.geometryCacheTainted, regionB.geometryCacheTainted);

}

template<typename _CharT, typename _Traits, int _Dim>
std::basic_ostream<_CharT, _Traits>&
operator<<(std::basic_ostream<_CharT, _Traits>& __os,
           const LibGeoDecomp::Region<_Dim>& region)
{
    __os << region.toString();
    return __os;
}

}

#endif
