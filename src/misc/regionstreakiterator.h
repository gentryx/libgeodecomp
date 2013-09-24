#ifndef LIBGEODECOMP_MISC_REGIONSTREAKITERATOR_H
#define LIBGEODECOMP_MISC_REGIONSTREAKITERATOR_H

#include <algorithm>
#include <libgeodecomp/misc/streak.h>
#include <libgeodecomp/misc/supervector.h>

namespace LibGeoDecomp {

// fixme: rename
namespace RegionHelpers {

template<int DIM>
class StreakIteratorCompareIterators
{
public:
    typedef std::pair<int, int> IntPair;
    typedef SuperVector<IntPair> VecType;

    inline bool operator()(const VecType::const_iterator *a, const VecType::const_iterator *b)
    {
        if (a[DIM] != b[DIM]) {
            return false;
        }

        return StreakIteratorCompareIterators<DIM - 1>()(a, b);
    }
};

template<>
class StreakIteratorCompareIterators<0>
{
public:
    typedef std::pair<int, int> IntPair;
    typedef SuperVector<IntPair> VecType;

    inline bool operator()(const VecType::const_iterator *a, const VecType::const_iterator *b)
    {
        return a[0] == b[0];
    }
};

}

template<int DIM, typename REGION>
class RegionStreakIterator : public std::iterator<std::forward_iterator_tag, const Streak<DIM> >
{
public:
    template<int> friend class InitIterators;
    template<int> friend class Region;
    typedef std::pair<int, int> IntPair;
    typedef SuperVector<IntPair> VecType;

    template<template<int D> class INIT_HELPER>
    inline RegionStreakIterator(const REGION *region, INIT_HELPER<DIM> /*unused*/) :
        region(region)
    {
        INIT_HELPER<DIM - 1>()(&streak, iterators, *region);
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
            streak.origin[0] = iterators[0]->first;
            streak.endX = iterators[0]->second;
        }

        for (int i = 1; i < DIM; ++i) {
            // we don't need to (and can't without performing
            // illegal reads) advance upper-level iterators if
            // they're already pointing to the second-to-last
            // field:
            if ((iterators[i] + 1) == region->indicesEnd(i)) {
                return;
            }

            VecType::const_iterator nextEnd =
                region->indicesBegin(i - 1) + (iterators[i] + 1)->second;

            if (iterators[i - 1] != nextEnd) {
                return;
            }

            ++iterators[i];
            streak.origin[i] = iterators[i]->first;
        }
    }

    inline bool operator==(const RegionStreakIterator& other) const
    {
        return RegionHelpers::StreakIteratorCompareIterators<DIM - 1>()(iterators, other.iterators);
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
    VecType::const_iterator iterators[DIM];
    Streak<DIM> streak;
    const REGION *region;
};

}

#endif
