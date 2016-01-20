#ifndef LIBGEODECOMP_GEOMETRY_REGIONSTREAKITERATOR_H
#define LIBGEODECOMP_GEOMETRY_REGIONSTREAKITERATOR_H

#include <algorithm>
#include <libgeodecomp/geometry/streak.h>
#include <libgeodecomp/misc/stdcontaineroverloads.h>

namespace LibGeoDecomp {

namespace RegionStreakIteratorHelpers {

/**
 * Recursively compares two iterators' components; useful for deciding
 * which iterator preceeds the other.
 */
template<int DIM>
class Compare
{
public:
    typedef std::pair<int, int> IntPair;
    typedef std::vector<IntPair> IndexVectorType;

    inline bool operator()(
        const IndexVectorType::const_iterator *a,
        const IndexVectorType::const_iterator *b)
    {
        if (a[DIM] != b[DIM]) {
            return false;
        }

        return Compare<DIM - 1>()(a, b);
    }
};

/**
 * See above.
 */
template<>
class Compare<0>
{
public:
    typedef std::pair<int, int> IntPair;
    typedef std::vector<IntPair> IndexVectorType;

    inline bool operator()(
        const IndexVectorType::const_iterator *a,
        const IndexVectorType::const_iterator *b)
    {
        return a[0] == b[0];
    }
};

}

/**
 * Iterates Streak-wise through a Region. This is often preferable to
 * a Coordinate-wise iteration as this (combined with an additional
 * inner loop for iteration through the Streak) reduces the effective
 * overhead. It also preserves the original runlenght coding within
 * the Region.
 */
template<int DIM, typename REGION>
class RegionStreakIterator : public std::iterator<std::forward_iterator_tag, const Streak<DIM> >
{
public:
    template<typename REGION_TYPE>
    friend Coord<DIM> operator-(const RegionStreakIterator<DIM, REGION_TYPE>& a,
                                const RegionStreakIterator<DIM, REGION_TYPE>& b);

    template<int> friend class InitIterators;
    template<int> friend class Region;
    friend class RegionStreakIteratorTest;

    typedef std::pair<int, int> IntPair;
    typedef std::vector<IntPair> IndexVectorType;

    template<typename INIT_HELPER>
    inline RegionStreakIterator(
        const REGION *region,
        INIT_HELPER initHelper,
        const Coord<DIM>& offset = Coord<DIM>()) :
        offset(offset),
        region(region)
    {
        initHelper(&streak, iterators, *region, offset);
    }

    inline void operator++()
    {
        ++iterators[0];
        if (iterators[0] == region->indicesEnd(0)) {
            for (int i = 1; i < DIM; ++i) {
                iterators[i] = region->indicesEnd(i);
            }
            return;
        } else {
            streak.origin[0] = iterators[0]->first + offset[0];
            streak.endX = iterators[0]->second + offset[0];
        }

        for (int i = 1; i < DIM; ++i) {
            // we don't need to (and can't without performing
            // illegal reads) advance upper-level iterators if
            // they're already pointing to the second-to-last
            // field:
            if ((iterators[i] + 1) == region->indicesEnd(i)) {
                return;
            }

            IndexVectorType::const_iterator nextEnd =
                region->indicesBegin(i - 1) + (iterators[i] + 1)->second;

            if (iterators[i - 1] != nextEnd) {
                return;
            }

            ++iterators[i];
            streak.origin[i] = iterators[i]->first + offset[i];
        }
    }

    inline bool operator==(const RegionStreakIterator& other) const
    {
        return RegionStreakIteratorHelpers::Compare<DIM - 1>()(iterators, other.iterators);
    }

    inline bool operator!=(const RegionStreakIterator& other) const
    {
        return !(*this == other);
    }

    inline const Streak<DIM>& operator*() const
    {
        return streak;
    }

    inline const Streak<DIM> *operator->() const
    {
        return &streak;
    }

    inline bool endReached() const
    {
        return iterators[0] == region->indicesEnd(0);
    }

private:
    IndexVectorType::const_iterator iterators[DIM];
    Streak<DIM> streak;
    Coord<DIM> offset;
    const REGION *region;
};

template<typename REGION>
inline Coord<1> operator-(
    const RegionStreakIterator<1, REGION>& a,
    const RegionStreakIterator<1, REGION>& b)
{
    return Coord<1>(a.iterators[0] - b.iterators[0]);
}

template<typename REGION>
inline Coord<2> operator-(
    const RegionStreakIterator<2, REGION>& a,
    const RegionStreakIterator<2, REGION>& b)
{
    return Coord<2>(a.iterators[0] - b.iterators[0],
                    a.iterators[1] - b.iterators[1]);
}

template<typename REGION>
inline Coord<3> operator-(
    const RegionStreakIterator<3, REGION>& a,
    const RegionStreakIterator<3, REGION>& b)
{
    return Coord<3>(a.iterators[0] - b.iterators[0],
                    a.iterators[1] - b.iterators[1],
                    a.iterators[2] - b.iterators[2]);
}

}

#endif
