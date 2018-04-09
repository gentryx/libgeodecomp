#ifndef LIBGEODECOMP_GEOMETRY_COORDBOX_H
#define LIBGEODECOMP_GEOMETRY_COORDBOX_H

#include <libgeodecomp/geometry/coord.h>
#include <libgeodecomp/geometry/streak.h>
#include <libgeodecomp/geometry/topologies.h>

// Kill warning 4514 in system headers
#ifdef _MSC_BUILD
#pragma warning( push )
#pragma warning( disable : 4514 )
#endif

#include <iostream>
#include <stdexcept>

#ifdef _MSC_BUILD
#pragma warning( pop )
#endif

namespace LibGeoDecomp {

/**
 * CoordBox describes a rectangular, N-dimensional set of coordinates.
 * If can be used for iteration and supports bounds checking. This
 * makes it useful for writing code which is independed of the
 * dimensionality of its data.
 */
template<int DIM>
class CoordBox
{
public:
    friend class BoostSerialization;
    friend class HPXSerialization;
    friend class Typemaps;

    class Iterator;
    class StreakIterator;
    typedef Coord<DIM> CoordType;

    Coord<DIM> origin;
    Coord<DIM> dimensions;

    __host__ __device__
    explicit CoordBox(
        const Coord<DIM>& origin = Coord<DIM>(),
        const Coord<DIM>& dimensions = Coord<DIM>()) :
        origin(origin),
        dimensions(dimensions)
    {}

    __host__ __device__
    inline bool operator==(const CoordBox& other) const
    {
        return (origin == other.origin &&
                dimensions == other.dimensions);
    }


    __host__ __device__
    inline bool operator!=(const CoordBox& other) const
    {
        return !(*this == other);
    }

    inline Iterator begin() const
    {
        return Iterator(origin, origin, dimensions);
    }

    inline StreakIterator beginStreak() const
    {
        return StreakIterator(origin, dimensions);
    }

    inline Iterator end() const
    {
        Coord<DIM> pos = origin;
        pos[DIM - 1] += dimensions[DIM - 1];
        return Iterator(origin, pos, dimensions);
    }

    inline StreakIterator endStreak() const
    {
        Coord<DIM> pos = endPos(origin, dimensions);
        return StreakIterator(pos, dimensions);
    }

    /**
     * checks whether the box' volume includes/containes coord.
     * Returns true if coord is within the limits of the CoordBox.
     */
    __host__ __device__
    inline bool inBounds(const Coord<DIM>& coord) const
    {
        Coord<DIM> relativeCoord = coord - origin;
        return !Topologies::Cube<DIM>::Topology::isOutOfBounds(relativeCoord, dimensions);
    }

    std::string toString() const
    {
        std::ostringstream temp;
        temp << "CoordBox<" << DIM << ">(origin: " << origin << ", "
             << "dimensions: " << dimensions << ")";
        return temp.str();
    }

    __host__ __device__
    bool intersects(const CoordBox& other) const
    {
        Coord<DIM> maxOrigin = (origin.max)(other.origin);
        return inBounds(maxOrigin) && other.inBounds(maxOrigin);
    }

    __host__ __device__
    inline std::size_t size() const
    {
        return static_cast<std::size_t>(dimensions.prod());
    }

    class Iterator
    {
    public:
        friend class StreakIterator;

        inline Iterator(
            const Coord<DIM>& origin,
            const Coord<DIM>& start,
            const Coord<DIM>& dimensions) :
            cursor(start),
            origin(origin),
            end(origin + dimensions)
        {}

        inline bool operator==(const Iterator& other) const
        {
            return cursor == other.cursor;
        }

        inline bool operator!=(const Iterator& other) const
        {
            return !(*this == other);
        }

        inline const Coord<DIM>& operator*() const
        {
            return cursor;
        }

        inline const Coord<DIM>* operator->() const
        {
            return &cursor;
        }

        inline Iterator& operator++()
        {
            int i;

            for (i = 0; i < DIM - 1; ++i) {
                if (++cursor[i] == end[i]) {
                    cursor[i] = origin[i];
                } else {
                    break;
                }
            }
            if (i == DIM - 1) {
                ++cursor[DIM - 1];
            }
            return *this;
        }

        inline std::string toString() const
        {
            std::ostringstream buffer;
            buffer << "StripingPartition::Iterator(" << cursor << ", " << end << ")";
            return buffer.str();
        }

    private:
        Coord<DIM> cursor;
        Coord<DIM> origin;
        Coord<DIM> end;
    };

    class StreakIterator
    {
    public:
        inline StreakIterator(
            const Coord<DIM>& origin,
            const Coord<DIM>& dimensions) :
            streak(origin, origin.x() + dimensions.x()),
            iter(origin, origin, dimensions)
        {
            iter.end.x() = origin.x() + 1;
        }

        inline bool operator==(const StreakIterator& other) const
        {
            return iter == other.iter;
        }

        inline bool operator!=(const StreakIterator& other) const
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

        inline StreakIterator& operator++()
        {
            ++iter;
            streak.origin = *iter;
            return *this;
        }

    private:
        Streak<DIM> streak;
        Iterator iter;
    };

private:
    static Coord<1> endPos(const Coord<1>& origin, const Coord<1>& dimensions)
    {
        Coord<1> pos = origin;
        if (dimensions.x() > 0) {
            pos[0] += 1;
        }

        return pos;
    }

    template<int DIM2>
    static Coord<DIM2> endPos(const Coord<DIM2>& origin, const Coord<DIM2>& dimensions)
    {
        Coord<DIM2> pos = origin;
        pos[DIM2 - 1] += dimensions[DIM2 - 1];

        return pos;
    }
};

/**
 * The MPI typemap generator need to find out for which template
 * parameter values it should generate typemaps. It does so by
 * scanning all class members. Therefore this dummy class forces the
 * typemap generator to create MPI datatypes for CoordBox with the
 * dimensions as specified below.
 */
class CoordBoxMPIDatatypeHelper
{
    friend class Typemaps;

    CoordBox<1> a;
    CoordBox<2> b;
    CoordBox<3> c;
};

template<typename _CharT, typename _Traits, int _Dimensions>
std::basic_ostream<_CharT, _Traits>&
operator<<(std::basic_ostream<_CharT, _Traits>& __os,
           const LibGeoDecomp::CoordBox<_Dimensions>& rect)
{
    __os << rect.toString();
    return __os;
}

}


#endif
