#ifndef LIBGEODECOMP_GEOMETRY_REGION_H
#define LIBGEODECOMP_GEOMETRY_REGION_H

#include <libgeodecomp/geometry/coordbox.h>
#include <libgeodecomp/geometry/regionstreakiterator.h>
#include <libgeodecomp/geometry/streak.h>
#include <libgeodecomp/misc/stdcontaineroverloads.h>

namespace LibGeoDecomp {

class RegionTest;

namespace RegionHelpers {

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
    typedef std::vector<IntPair> VecType;

    inline void incRemainder(const VecType::iterator& start, const VecType::iterator& end, const int& inserts)
    {
        if (inserts == 0) {
            return;
        }

        for (VecType::iterator incrementer = start;
             incrementer != end; ++incrementer) {
            incrementer->second += inserts;
        }
    }
};

template<int DIM>
class ConstructStreakFromIterators
{
public:
    typedef std::pair<int, int> IntPair;
    typedef std::vector<IntPair> VecType;

    template<int STREAK_DIM>
    inline void operator()(Streak<STREAK_DIM> *streak, VecType::const_iterator *iterators)
    {
        ConstructStreakFromIterators<DIM - 1>()(streak, iterators);
        streak->origin[DIM] = iterators[DIM]->first;
    }
};

template<>
class ConstructStreakFromIterators<0>
{
public:
    typedef std::pair<int, int> IntPair;
    typedef std::vector<IntPair> VecType;

    template<int STREAK_DIM>
    inline void operator()(Streak<STREAK_DIM> *streak, VecType::const_iterator *iterators)
    {
        streak->origin[0] = iterators[0]->first;
        streak->endX      = iterators[0]->second;
    }
};

template<int DIM>
class StreakIteratorInitSingleOffset
{
public:
    typedef std::pair<int, int> IntPair;
    typedef std::vector<IntPair> VecType;

    explicit StreakIteratorInitSingleOffset(const std::size_t& offset) :
        offset(offset)
    {}

    template<int STREAK_DIM, typename REGION>
    inline std::size_t operator()(Streak<STREAK_DIM> *streak, VecType::const_iterator *iterators, const REGION& region) const
    {
        StreakIteratorInitSingleOffset<DIM - 1> delegate(offset);
        std::size_t newOffset = delegate(streak, iterators, region);

        VecType::const_iterator upperBound =
            std::upper_bound(region.indicesBegin(DIM),
                             region.indicesEnd(DIM),
                             IntPair(0, newOffset),
                             RegionHelpers::RegionCommonHelper::pairCompareSecond);
        iterators[DIM] = upperBound - 1;
        newOffset =  iterators[DIM] - region.indicesBegin(DIM);

        return newOffset;
    }

private:
    const std::size_t& offset;
};

template<>
class StreakIteratorInitSingleOffset<0>
{
public:
    typedef std::pair<int, int> IntPair;
    typedef std::vector<IntPair> VecType;

    explicit StreakIteratorInitSingleOffset(const std::size_t& offset) :
        offset(offset)
    {}

    template<int STREAK_DIM, typename REGION>
    inline std::size_t operator()(Streak<STREAK_DIM> *streak, VecType::const_iterator *iterators, const REGION& region) const
    {
        iterators[0] = region.indicesBegin(0) + offset;
        return offset;
    }

private:
    const std::size_t& offset;
};

template<int DIM>
class StreakIteratorInitSingleOffsetWrapper
{
public:
    typedef std::pair<int, int> IntPair;
    typedef std::vector<IntPair> VecType;

    explicit StreakIteratorInitSingleOffsetWrapper(const std::size_t& offset) :
        offset(offset)
    {}

    template<int STREAK_DIM, typename REGION>
    inline void operator()(Streak<STREAK_DIM> *streak, VecType::const_iterator *iterators, const REGION& region) const
    {
        StreakIteratorInitSingleOffset<DIM> delegate(offset);
        delegate(streak, iterators, region);
        ConstructStreakFromIterators<DIM>()(streak, iterators);
    }

private:
    const std::size_t& offset;
};

template<int DIM, int COORD_DIM>
class StreakIteratorInitOffsets
{
public:
    typedef std::pair<int, int> IntPair;
    typedef std::vector<IntPair> VecType;

    explicit StreakIteratorInitOffsets(const Coord<COORD_DIM>& offsets) :
        offsets(offsets)
    {}

    template<int STREAK_DIM, typename REGION>
    inline void operator()(Streak<STREAK_DIM> *streak, VecType::const_iterator *iterators, const REGION& region) const
    {
        iterators[DIM] = region.indicesBegin(DIM) + offsets[DIM];

        StreakIteratorInitOffsets<DIM - 1, COORD_DIM> delegate(offsets);
        delegate(streak, iterators, region);
    }

private:
    const Coord<COORD_DIM>& offsets;
};

template<int COORD_DIM>
class StreakIteratorInitOffsets<0, COORD_DIM>
{
public:
    typedef std::pair<int, int> IntPair;
    typedef std::vector<IntPair> VecType;

    explicit StreakIteratorInitOffsets(const Coord<COORD_DIM>& offsets) :
        offsets(offsets)
    {}

    template<int STREAK_DIM, typename REGION>
    inline void operator()(Streak<STREAK_DIM> *streak, VecType::const_iterator *iterators, const REGION& region) const
    {
        iterators[0] = region.indicesBegin(0) + offsets[0];

        if (int(region.indicesSize(0)) > offsets[0]) {
            ConstructStreakFromIterators<STREAK_DIM - 1>()(streak, iterators);
        }
    }

private:
    const Coord<COORD_DIM>& offsets;
};

template<int DIM>
class StreakIteratorInitBegin
{
public:
    typedef std::pair<int, int> IntPair;
    typedef std::vector<IntPair> VecType;

    template<int STREAK_DIM, typename REGION>
    inline void operator()(Streak<STREAK_DIM> *streak, VecType::const_iterator *iterators, const REGION& region) const
    {
        iterators[DIM] = region.indicesBegin(DIM);
        StreakIteratorInitBegin<DIM - 1>()(streak, iterators, region);
    }
};

template<>
class StreakIteratorInitBegin<0>
{
public:
    typedef std::pair<int, int> IntPair;
    typedef std::vector<IntPair> VecType;

    template<int STREAK_DIM, typename REGION>
    inline void operator()(Streak<STREAK_DIM> *streak, VecType::const_iterator *iterators, const REGION& region) const
    {
        iterators[0] = region.indicesBegin(0);

        if (region.indicesSize(0) > 0) {
            ConstructStreakFromIterators<STREAK_DIM - 1>()(streak, iterators);
        }
    }
};

template<int DIM>
class StreakIteratorInitEnd
{
public:
    typedef std::pair<int, int> IntPair;
    typedef std::vector<IntPair> VecType;

    template<int STREAK_DIM, typename REGION>
    inline void operator()(Streak<STREAK_DIM> *streak, VecType::const_iterator *iterators, const REGION& region) const
    {
        StreakIteratorInitEnd<DIM - 1>()(streak, iterators, region);
        iterators[DIM] = region.indicesEnd(DIM);
    }
};

template<>
class StreakIteratorInitEnd<0>
{
public:
    typedef std::pair<int, int> IntPair;
    typedef std::vector<IntPair> VecType;

    template<int STREAK_DIM, typename REGION>
    inline void operator()(Streak<STREAK_DIM> *streak, VecType::const_iterator *iterators, const REGION& region) const
    {
        iterators[0] = region.indicesEnd(0);
    }
};

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

template<int DIM>
class RegionLookupHelper;

template<int DIM>
class RegionInsertHelper;

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

    template<int MY_DIM> friend class RegionHelpers::RegionLookupHelper;
    template<int MY_DIM> friend class RegionHelpers::RegionInsertHelper;
    template<int MY_DIM> friend class RegionHelpers::RegionRemoveHelper;
    friend class LibGeoDecomp::RegionTest;

    typedef std::pair<int, int> IntPair;
    typedef std::vector<IntPair> VecType;
    typedef RegionStreakIterator<DIM, Region<DIM> > StreakIterator;

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

    inline const std::size_t& size() const
    {
        if (geometryCacheTainted) {
            resetGeometryCache();
        }
        return mySize;
    }

    inline std::size_t numStreaks() const
    {
        return indices[0].size();
    }

    inline Region expand(const unsigned& width=1) const
    {
        Region ret;
        Coord<DIM> dia = Coord<DIM>::diagonal(width);

        for (StreakIterator i = beginStreak(); i != endStreak(); ++i) {
            Streak<DIM> streak = *i;

            Coord<DIM> boxOrigin = streak.origin - dia;
            Coord<DIM> boxDim = Coord<DIM>::diagonal(2 * width + 1);
            boxDim.x() = 1;
            int endX = streak.endX + width;
            CoordBox<DIM> box(boxOrigin, boxDim);

            for (typename CoordBox<DIM>::Iterator i = box.begin(); i != box.end(); ++i) {
                ret << Streak<DIM>(*i, endX);
            }
        }

        return ret;
    }

    /**
     * does the same as expand, but will wrap overlap at edges
     * correctly. The instance of the TOPOLOGY is actually unused, but
     * without it g++ would complain...
     */
    template<typename TOPOLOGY>
    inline Region expandWithTopology(
        const unsigned& width,
        const Coord<DIM>& dimensions,
        TOPOLOGY /* unused */) const
    {
        Region ret;
        Coord<DIM> dia = Coord<DIM>::diagonal(width);

        for (StreakIterator i = beginStreak(); i != endStreak(); ++i) {
            Streak<DIM> streak = *i;

            Coord<DIM> boxOrigin = streak.origin - dia;
            Coord<DIM> boxDim = Coord<DIM>::diagonal(2 * width + 1);
            boxDim.x() = 1;
            int endX = streak.endX + width;
            CoordBox<DIM> box(boxOrigin, boxDim);

            for (typename CoordBox<DIM>::Iterator i = box.begin(); i != box.end(); ++i) {
                Streak<DIM> newStreak(*i, endX);
                if (TOPOLOGY::template WrapsAxis<0>::VALUE) {
                    splitStreak<TOPOLOGY>(newStreak, &ret, dimensions);
                } else {
                    normalizeStreak<TOPOLOGY>(
                        trimStreak(newStreak, dimensions), &ret, dimensions);
                }
            }
        }

        return ret;
    }

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

        geometryCacheTainted = true;
        RegionHelpers::RegionInsertHelper<DIM - 1>()(this, s);
        return *this;
    }

    inline Region& operator<<(const Coord<DIM>& c)
    {
        *this << Streak<DIM>(c, c.x() + 1);
        return *this;
    }

    inline Region& operator<<(CoordBox<DIM> box)
    {
        int width = box.dimensions.x();
        box.dimensions.x() = 1;

        for (typename CoordBox<DIM>::Iterator i = box.begin(); i != box.end(); ++i) {
            *this << Streak<DIM>(*i, i->x() + width);
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

    inline void operator-=(const Region& other)
    {
        if(other.empty()) return;
        for (StreakIterator i = other.beginStreak(); i != other.endStreak(); ++i) {
            *this >> *i;
        }
    }

    /**
     * Equvalent to (A and (not B)) in sets, where other corresponds
     * to B and *this corresponds to A.
     */
    inline Region operator-(const Region& other) const
    {
        Region ret(*this);
        ret -= other;
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
                intersection.origin.x() = (std::max)(myIter->origin.x(), otherIter->origin.x());
                intersection.endX = (std::min)(myIter->endX, otherIter->endX);
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
        if(other.empty()) return;
        for (StreakIterator i = other.beginStreak(); i != other.endStreak(); ++i) {
            *this << *i;
        }
    }

    inline Region operator+(const Region& other) const
    {
        Region ret(*this);
        ret += other;
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

    inline StreakIterator beginStreak() const
    {
        return StreakIterator(this, RegionHelpers::StreakIteratorInitBegin<DIM - 1>());
    }

    inline StreakIterator endStreak() const
    {
        return StreakIterator(this, RegionHelpers::StreakIteratorInitEnd<DIM - 1>());
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

    inline VecType::const_iterator indicesAt(std::size_t dim, std::size_t offset) const
    {
        return indices[dim].begin() + offset;
    }

    inline VecType::const_iterator indicesBegin(std::size_t dim) const
    {
        return indices[dim].begin();
    }

    inline VecType::const_iterator indicesEnd(std::size_t dim) const
    {
        return indices[dim].end();
    }

private:
    VecType indices[DIM];
    mutable CoordBox<DIM> myBoundingBox;
    mutable std::size_t mySize;
    mutable bool geometryCacheTainted;

    inline void determineGeometry() const
    {
        if (empty()) {
            myBoundingBox = CoordBox<DIM>();
        } else {
            Streak<DIM> someStreak = *beginStreak();
            Coord<DIM> min = someStreak.origin;
            Coord<DIM> max = someStreak.origin;

            mySize = 0;
            for (StreakIterator i = beginStreak();
                 i != endStreak(); ++i) {
                Coord<DIM> left = i->origin;
                Coord<DIM> right = i->origin;
                right.x() = i->endX - 1;

                min = (min.min)(left);
                max = (max.max)(right);
                mySize += i->endX - i->origin.x();
            }

            myBoundingBox =
                CoordBox<DIM>(min, max - min + Coord<DIM>::diagonal(1));
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
        int width = dimensions.x();
        Streak<DIM> buf = s;
        buf.origin.x() = (std::max)(buf.origin.x(), 0);
        buf.endX = (std::min)(width, buf.endX);
        return buf;
    }

    template<typename TOPOLOGY>
    void splitStreak(
        const Streak<DIM>& streak,
        Region *target,
        const Coord<DIM>& dimensions) const
    {
        int width = dimensions.x();

        int currentX = streak.origin.x();
        if (currentX < 0) {
            Streak<DIM> section = streak;
            section.endX = (std::min)(streak.endX, 0);
            currentX = section.endX;

            // normalize left overhang
            section.origin.x() += width;
            section.endX += width;
            normalizeStreak<TOPOLOGY>(section, target, dimensions);
        }

        if (currentX < streak.endX) {
            Streak<DIM> section = streak;
            section.origin.x() = currentX;
            section.endX = (std::min)(streak.endX, width);
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
};

namespace RegionHelpers {

template<int DIM>
class RegionLookupHelper : public RegionCommonHelper
{
public:
    typedef Region<1>::IntPair IntPair;
    typedef Region<1>::VecType VecType;

    template<int MY_DIM>
    inline bool operator()(const Region<MY_DIM>& region, const Streak<MY_DIM>& s)
    {
        const VecType& indices = region.indices[DIM];
        return (*this)(region, s, 0, indices.size());
    }

    template<int MY_DIM>
    inline bool operator()(const Region<MY_DIM>& region, const Streak<MY_DIM>& s, const int& start, const int& end)
    {
        int c = s.origin[DIM];
        const VecType& indices = region.indices[DIM];

        VecType::const_iterator i =
            std::upper_bound(
                indices.begin() + start,
                indices.begin() + end,
                IntPair(c, 0),
                RegionCommonHelper::pairCompareFirst);

        int nextLevelStart = 0;
        int nextLevelEnd = 0;

        if (i != (indices.begin() + start)) {
            VecType::const_iterator entry = i;
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

template<>
class RegionLookupHelper<0> : public RegionCommonHelper
{
public:
    typedef Region<1>::IntPair IntPair;
    typedef Region<1>::VecType VecType;

    template<int MY_DIM>
    inline bool operator()(const Region<MY_DIM>& region, const Streak<MY_DIM>& s, const int& start, int end)
    {
        IntPair curStreak(s.origin.x(), s.endX);
        const VecType& indices = region.indices[0];

        VecType::const_iterator cursor =
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

template<int DIM>
class RegionInsertHelper : public RegionCommonHelper
{
public:
    typedef Region<1>::IntPair IntPair;
    typedef Region<1>::VecType VecType;

    template<int MY_DIM>
    inline void operator()(Region<MY_DIM> *region, const Streak<MY_DIM>& s)
    {
        VecType& indices = region->indices[DIM];
        (*this)(region, s, 0, indices.size());
    }

    template<int MY_DIM>
    int operator()(Region<MY_DIM> *region, const Streak<MY_DIM>& s, const int& start, const int& end)
    {
        int c = s.origin[DIM];
        VecType& indices = region->indices[DIM];

        VecType::iterator i =
            std::upper_bound(
                indices.begin() + start,
                indices.begin() + end,
                IntPair(c, 0),
                RegionCommonHelper::pairCompareFirst);

        int nextLevelStart = 0;
        int nextLevelEnd = 0;

        if (i != (indices.begin() + start)) {
            VecType::iterator entry = i;
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

        VecType::iterator followingEntries;

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

template<>
class RegionInsertHelper<0>
{
public:
    friend class LibGeoDecomp::RegionTest;
    typedef Region<1>::IntPair IntPair;
    typedef Region<1>::VecType VecType;

    template<int MY_DIM>
    inline int operator()(Region<MY_DIM> *region, const Streak<MY_DIM>& s, int start, int end)
    {
        IntPair curStreak(s.origin.x(), s.endX);
        VecType& indices = region->indices[0];

        VecType::iterator cursor =
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
        return IntPair((std::min)(a.first, b.first),
                       (std::max)(a.second, b.second));
    }
};

template<int DIM>
class RegionRemoveHelper : public RegionCommonHelper
{
public:
    typedef Region<1>::IntPair IntPair;
    typedef Region<1>::VecType VecType;

    template<int MY_DIM>
    inline void operator()(Region<MY_DIM> *region, const Streak<MY_DIM>& s)
    {
        VecType& indices = region->indices[DIM];
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
        VecType& indices = region->indices[DIM];

        VecType::iterator i =
            std::upper_bound(
                indices.begin() + start,
                indices.begin() + end,
                IntPair(c, 0),
                RegionCommonHelper::pairCompareFirst);

        // key is not present, so no need to remove it
        if (i == (indices.begin() + start)) {
            return 0;
        }

        VecType::iterator entry = i;
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

template<>
class RegionRemoveHelper<0>
{
public:
    friend class LibGeoDecomp::RegionTest;
    typedef Region<1>::IntPair IntPair;
    typedef Region<1>::VecType VecType;

    template<int MY_DIM>
    int operator()(Region<MY_DIM> *region, const Streak<MY_DIM>& s, const int& start, int end)
    {
        int c = s.origin[0];
        VecType& indices = region->indices[0];
        int inserts = 0;

        // This will yield the streak AFTER the current origin
        // c. We can't really use lower_bound() as this doesn't
        // replace the < operator by >= but rather by <=, which is
        // IMO really sick...
        VecType::iterator cursor =
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
                VecType newStreaks(substract(*cursor, curStreak));
                cursor = indices.erase(cursor);
                int delta = newStreaks.size() - 1;
                end += delta;
                inserts += delta;

                for (VecType::iterator i = newStreaks.begin(); i != newStreaks.end(); ++i) {
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

    inline VecType substract(const IntPair& base, const IntPair& minuend) const
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
