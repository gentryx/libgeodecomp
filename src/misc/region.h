#ifndef _libgeodecomp_misc_region_h_
#define _libgeodecomp_misc_region_h_

#include <libgeodecomp/misc/coordbox.h>
#include <libgeodecomp/misc/streak.h>
#include <libgeodecomp/misc/supermap.h>
#include <libgeodecomp/misc/supervector.h>

namespace LibGeoDecomp {

template<int DIM>
class StreakMapDefine;

template<>
class StreakMapDefine<2> 
{ 
public:
    typedef SuperMap<int, Streak<2> > Line;
    typedef SuperMap<int, Line> StreakMap;
};

template<>
class StreakMapDefine<3> 
{ 
public:
    typedef SuperMap<int, Streak<3> > Line;
    typedef SuperMap<int, Line> Slice;
    typedef SuperMap<int, Slice> StreakMap;
};

template<int DIM>
class StreakIterator;

template<>
class StreakIterator<2> : public std::iterator<std::forward_iterator_tag, const Streak<2> >
{
public:
    typedef StreakMapDefine<2>::Line Line;
    typedef StreakMapDefine<2>::StreakMap StreakMap;

    inline StreakIterator(
        const StreakMap::const_iterator& startLine, 
        const StreakMap *_streaks) :
        curLine(startLine),
        streaks(_streaks)
    {
        if (!endReached()) 
            curStreak = curLine->second.begin();
    }

    inline bool operator==(const StreakIterator& other) const
    {         
        return (curLine == other.curLine) && 
            ((endReached() || other.endReached()) || curStreak == other.curStreak);
    }

    inline bool operator!=(const StreakIterator& other)
    {
        return !(*this == other);
    }

    inline void operator++()
    {
        if (curStreak != curLine->second.end())
            curStreak++;
        if (curStreak == curLine->second.end()) {
            curLine++;
            if (!endReached()) 
                curStreak = curLine->second.begin();
        }
    }

    inline const Streak<2>& operator*() const
    {
        return curStreak->second;
    }

    inline const Streak<2> *operator->() const
    {
        return &curStreak->second;
    }

    inline bool endReached() const
    {
        return curLine == streaks->end();
    }

private:
    StreakMap::const_iterator curLine;
    Line::const_iterator curStreak;
    const StreakMap *streaks;
};

template<>
class StreakIterator<3> : public std::iterator<std::forward_iterator_tag, const Streak<3> >
{
public:
    typedef StreakMapDefine<3>::Line Line;
    typedef StreakMapDefine<3>::Slice Slice;
    typedef StreakMapDefine<3>::StreakMap StreakMap;

    inline StreakIterator(
        const StreakMap::const_iterator& startSlice, 
        const StreakMap *_streaks) :
        curSlice(startSlice),
        streaks(_streaks)
    {
        if (!endReached()) {
            curLine = curSlice->second.begin();
            curStreak = curLine->second.begin();
        }
    }

    inline bool operator==(const StreakIterator& other) const
    {         
        return (curSlice == other.curSlice) && 
            ((endReached() || other.endReached()) || 
             ((curLine   == other.curLine) && 
              (curStreak == other.curStreak)));
    }

    inline bool operator!=(const StreakIterator& other)
    {
        return !(*this == other);
    }

    inline void operator++()
    {
        if (curStreak != curLine->second.end())
            curStreak++;
        if (curStreak == curLine->second.end()) {
            curLine++;
            if (curLine == curSlice->second.end()) {
                curSlice++;
                if (!endReached()) 
                    curLine = curSlice->second.begin();
            }
            
            if (!endReached()) 
                curStreak = curLine->second.begin();
        }
    }

    inline const Streak<3>& operator*() const
    {
        return curStreak->second;
    }

    inline const Streak<3> *operator->() const
    {
        return &curStreak->second;
    }

    inline bool endReached() const
    {
        return curSlice == streaks->end();
    }

private:
    StreakMap::const_iterator curSlice;
    Slice::const_iterator curLine;
    Line::const_iterator curStreak;
    const StreakMap *streaks;
};

template<int DIM>
class LookupLine;

template<>
class LookupLine<2>
{
public:
    typedef StreakMapDefine<2>::Line Line;
    typedef StreakMapDefine<2>::StreakMap StreakMap;

    Line& operator()(StreakMap& streaks, const Coord<2> c)
    {
        return streaks[c.y()];
    }
};

template<>
class LookupLine<3>
{
public:
    typedef StreakMapDefine<3>::Line Line;
    typedef StreakMapDefine<3>::StreakMap StreakMap;

    Line& operator()(StreakMap& streaks, const Coord<3> c)
    {
        return streaks[c.z()][c.y()];
    }
};

template<int DIM>
class EraseEmptyStreakEntries;

template<>
class EraseEmptyStreakEntries<2>
{
public:
    void operator()(StreakMapDefine<2>::StreakMap *streaks, const Streak<2>& streak)
    {
        streaks->erase(streak.origin.y());
    }
};

template<>
class EraseEmptyStreakEntries<3>
{
public:
    void operator()(StreakMapDefine<3>::StreakMap *streaks, const Streak<3>& streak)
    {
        (*streaks)[streak.origin.z()].erase(streak.origin.y());
        if ((*streaks)[streak.origin.z()].empty())
            streaks->erase(streak.origin.z());
    }
};

template<int DIM>
class Region
{
    friend class RegionTest;
    friend class Iterator;
    friend class StreakIterator<2>;
    friend class StreakIterator<3>;
public:
    typedef typename StreakMapDefine<DIM>::Line Line;
    typedef typename StreakMapDefine<DIM>::StreakMap StreakMap;

    class Iterator : public std::iterator<std::forward_iterator_tag, 
                                          const Coord<DIM> >
    {
    public:
        inline Iterator(const StreakIterator<DIM>& _streakIterator) :
            streakIterator(_streakIterator)
        {
            if (!streakIterator.endReached()) 
                cursor = _streakIterator->origin;
        }
        
        inline void operator++()
        {
            cursor.x()++;
            if (cursor.x() >= streakIterator->endX) {
                ++streakIterator;
                if (!streakIterator.endReached())
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
        StreakIterator<DIM> streakIterator;
        Coord<DIM> cursor;
    };

    inline Region() :
        myNumStreaks(0),
        geometryCacheTainted(false)
    {}

    template<class ITERATOR1, class ITERATOR2>
    inline Region(const ITERATOR1& start, const ITERATOR2& end)
    {
        load(start, end);
    }

    template<class ITERATOR1, class ITERATOR2>
    inline void load(const ITERATOR1& start, const ITERATOR2& end)
    {
        for (ITERATOR1 i = start; i != end; ++i)
            *this << *i;
    }

    inline void clear()
    {
        streaks.clear();
        geometryCacheTainted = true;
    }

    inline Region& operator<<(const Coord<DIM>& coord)
    {
        return *this << Streak<DIM>(coord, coord.x() + 1);
    }

    inline Region& operator<<(CoordBox<DIM> box)
    {
        int width = box.dimensions.x();
        box.dimensions.x() = 1;

        CoordBoxSequence<DIM> s = box.sequence();
        while (s.hasNext()) 
            *this << Streak<DIM>(s.next(), width);
        
        return *this;
    }
   
    inline Region& operator<<(const Streak<DIM>& streak)
    {
        geometryCacheTainted = true;
        Line& line(lookupLine(streak.origin));
        Streak<DIM> curStreak(streak);
        typename Line::iterator cursor(line.upper_bound(curStreak.origin.x()));
        // This will yield the streak AFTER the current origin
        // c. We can't really use lower_bound() as this doesn't
        // replace the < operator by >= but rather by <=, which is
        // IMO really sick...
        if (cursor != line.begin()) {
            // ...so we revert to landing one past the streak we're
            // searching and moving back afterwards:
            cursor--;
        }
   
        while (cursor != line.end()) {
            if (intersectOrTouch(cursor->second, curStreak)) {
                curStreak = fuse(cursor->second, curStreak);
                line.erase(cursor);
                cursor = line.upper_bound(curStreak.origin.x());
            } else {
                cursor++;
            }
                
            if (cursor == line.end() || !intersectOrTouch(cursor->second, curStreak))
                break;
        }
        
        line[curStreak.origin.x()] = curStreak;
        return *this;
    }
        
    inline void operator>>(const Coord<DIM>& coord)
    {
        *this >> Streak<DIM>(coord, coord.x() + 1);
    }
    
    inline Region& operator>>(const Streak<DIM>& streak)
    {
        //ignore 0 length streaks and empty selves
        if (streak.endX <= streak.origin.x() || empty())
            return *this;

        geometryCacheTainted = true;
        Line& line(lookupLine(streak.origin));
        typename Line::iterator cursor(line.upper_bound(streak.origin.x()));
        // This will yield the streak AFTER the current origin
        // c. We can't really use lower_bound() as this doesn't
        // replace the < operator by >= but rather by <=, which is
        // IMO really sick...
        if (cursor != line.begin()) {
            // ...so we revert to landing one past the streak we're
            // searching and moving back afterwards:
            cursor--;
        }
   
        while (cursor != line.end()) {
            if (intersect(cursor->second, streak)) {
                SuperVector<Streak<DIM> > newStreaks(substract(cursor->second, streak));
                line.erase(cursor);
                cursor = line.upper_bound(streak.origin.x());
                for (typename SuperVector<Streak<DIM> >::iterator i = newStreaks.begin(); 
                     i != newStreaks.end(); 
                     ++i)
                    line[i->origin.x()] = *i;
            } else {
                cursor++;
            }
                
            if (cursor == line.end() || !intersect(cursor->second, streak))
                break;
        }

        if (line.empty())
            EraseEmptyStreakEntries<DIM>()(&streaks, streak);
        
        return *this;
    }

    inline bool operator==(const Region& other) const
    {    
        return streaks == other.streaks;
    }

    inline bool operator!=(const Region& other) const
    {    
        return !(*this == other);
    }

    inline StreakIterator<DIM> beginStreak() const
    {
        return StreakIterator<DIM>(streaks.begin(), &streaks);
    }

    inline StreakIterator<DIM> endStreak() const
    {
        return StreakIterator<DIM>(streaks.end(), &streaks);
    }

    inline Iterator begin() const
    {
        return Iterator(beginStreak());
    }
    
    inline Iterator end() const
    {
        return Iterator(endStreak());
    }

    inline bool empty() const
    {
        return streaks.empty();
    }

    inline const CoordBox<DIM>& boundingBox() const
    {
        if (geometryCacheTainted)
            resetGeometryCache();
        return myBoundingBox;
    }

    inline Region expand(const unsigned& width=1) const
    {
        Region ret;
        Coord<DIM> dia = CoordDiagonal<DIM>()(width);

        for (StreakIterator<DIM> i = beginStreak(); i != endStreak(); ++i) {
            Streak<DIM> streak = *i;

            Coord<DIM> boxOrigin = streak.origin - dia;
            Coord<DIM> boxDim = CoordDiagonal<DIM>()(2 * width + 1);
            boxDim.x() = 1;
            CoordBox<DIM> box(boxOrigin, boxDim);
            CoordBoxSequence<DIM> s = box.sequence();
            int endX = streak.endX + width;

            while (s.hasNext()) 
                ret << Streak<DIM>(s.next(), endX);
        }
        return ret;
    }
    
    inline const unsigned& numStreaks() const
    {
        if (geometryCacheTainted)
            resetGeometryCache();
        return myNumStreaks;
    }

    inline SuperVector<Streak<DIM> > toVector() const
    {
        SuperVector<Streak<DIM> > ret(numStreaks());
        std::copy(this->beginStreak(), this->endStreak(), ret.begin());
        return ret;
    }
    
    inline void operator-=(const Region& other) 
    {
        for (StreakIterator<DIM> i = other.beginStreak(); i != other.endStreak(); ++i) 
            *this >> *i;
    }
    
    inline Region operator-(const Region& other) const
    {
        Region ret(*this);
        ret -= other;
        return ret;
    }
    
    inline void operator&=(const Region& other) 
    {
        Region excess(*this);
        excess -= other;
        *this -= excess;
    }
    
    inline Region operator&(const Region& other) const
    {
        Region ret(*this);
        ret &= other;
        return ret;
    }

    inline void operator+=(const Region& other)
    {
        for (StreakIterator<DIM> i = other.beginStreak(); i != other.endStreak(); ++i) 
        *this << *i;
    }

    inline Region operator+(const Region& other) const
    {
        Region ret(*this);
        ret += other;
        return ret;
    }

    inline std::string toString() const
    {
        std::ostringstream buf;
        buf << "Region(\n"
            << "  streaks: " << streaks << "\n"
            << "  geometryCacheTainted: " << geometryCacheTainted << "\n"
            << "  myBoundingBox: " << myBoundingBox << ")\n";
        return buf.str();
    }

private:
    StreakMap streaks;
    mutable CoordBox<DIM> myBoundingBox;
    mutable unsigned myNumStreaks;
    mutable bool geometryCacheTainted;

    Line& lookupLine(const Coord<DIM> c)
    {
        return LookupLine<DIM>()(streaks, c);
    }
    
    inline bool intersectOrTouch(const Streak<DIM>& a, const Streak<DIM>& b) const
    {
        return 
            ((a.origin.x() <= b.origin.x() && b.origin.x() <= a.endX) || 
             (b.origin.x() <= a.origin.x() && a.origin.x() <= b.endX));
    }
    
    inline bool intersect(const Streak<DIM>& a, const Streak<DIM>& b) const
    {
        return 
            ((a.origin.x() <= b.origin.x() && b.origin.x() < a.endX) || 
             (b.origin.x() <= a.origin.x() && a.origin.x() < b.endX));
    }
    
    inline Streak<DIM> fuse(const Streak<DIM>& a, const Streak<DIM>& b) const
    {
        Streak<DIM> ret;
        ret.origin = a.origin;
        ret.origin.x() = std::min(a.origin.x(), b.origin.x());
        ret.endX = std::max(a.endX, b.endX);
        return ret;
    }

    inline SuperVector<Streak<DIM> > substract(
        const Streak<DIM>& base, const Streak<DIM>& minuend) const
    {
        if (!intersect(base, minuend))
            return SuperVector<Streak<DIM> >(1, base);
        SuperVector<Streak<DIM> > ret;
        Streak<DIM> s1(base.origin, minuend.origin.x());
        Streak<DIM> s2(base);
        s2.origin.x() = minuend.endX;

        if (s1.endX > s1.origin.x())
            ret.push_back(s1);
        if (s2.endX > s2.origin.x())
            ret.push_back(s2);
        return ret;
    }

    inline void determineGeometry() const
    {
        if (empty()) {
            myBoundingBox = CoordBox<DIM>();
            myNumStreaks = 0;
        } else {
            const Streak<DIM>& someStreak = *beginStreak();
            Coord<DIM> min = someStreak.origin;
            Coord<DIM> max = someStreak.origin;

            myNumStreaks = 0;
            for (StreakIterator<DIM> i = beginStreak(); i != endStreak(); ++i) {
                Coord<DIM> left = i->origin;
                Coord<DIM> right = i->origin;
                right.x() = i->endX - 1;

                min = min.min(left);
                max = max.max(right);
                myNumStreaks++;
            }
            myBoundingBox = CoordBox<DIM>(min, max - min + CoordDiagonal<DIM>()(1));
        }
    }

    inline void resetGeometryCache() const
    {
        determineGeometry();       
        geometryCacheTainted = false;
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

#endif
