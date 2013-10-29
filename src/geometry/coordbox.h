#ifndef LIBGEODECOMP_GEOMETRY_COORDBOX_H
#define LIBGEODECOMP_GEOMETRY_COORDBOX_H

#include <iostream>
#include <stdexcept>
#include <libgeodecomp/geometry/coord.h>
#include <libgeodecomp/geometry/topologies.h>

namespace LibGeoDecomp {

template<int DIM>
class CoordBoxSequence;

template<int DIM>
class CoordBox
{
public:
    friend class Serialization;
    friend class Typemaps;

    class Iterator;

    Coord<DIM> origin;
    Coord<DIM> dimensions;

    explicit CoordBox(const Coord<DIM>& origin = Coord<DIM>(),
                      const Coord<DIM>& dimensions = Coord<DIM>()) :
        origin(origin),
        dimensions(dimensions)
    {}

    inline bool operator==(const CoordBox& other) const
    {
        return (origin == other.origin &&
                dimensions == other.dimensions);
    }


    inline bool operator!=(const CoordBox& other) const
    {
        return ! (*this == other);
    }

    inline Iterator begin() const
    {
        return Iterator(origin, origin, dimensions);
    }

    inline Iterator end() const
    {
        Coord<DIM> pos = origin;
        pos[DIM - 1] += dimensions[DIM - 1];
        return Iterator(origin, pos, dimensions);
    }

    /**
     * checks whether the box' volume includes coord.
     */
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

    bool intersects(const CoordBox& other) const
    {
        Coord<DIM> maxOrigin = (origin.max)(other.origin);
        return inBounds(maxOrigin) && other.inBounds(maxOrigin);
    }

    inline unsigned size() const
    {
        return dimensions.prod();
    }

    class Iterator
    {
    public:
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
