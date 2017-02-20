#ifndef LIBGEODECOMP_GEOMETRY_STREAK_H
#define LIBGEODECOMP_GEOMETRY_STREAK_H

#include <libgeodecomp/config.h>
#include <libgeodecomp/geometry/coord.h>

namespace LibGeoDecomp {

/**
 * A single run-lenght coded fragment of the StreakCollection. In the
 * 2D case, it begins at origin (x, y) and runs until (endX, y). endX
 * points just one past the last contained coordinate. It can be
 * tagged as follows:
 */
template<int DIM>
class Streak
{
public:
    friend class BoostSerialization;
    friend class HPXSerialization;
    friend class Typemaps;

    inline explicit Streak(
        const Coord<DIM>& origin = Coord<DIM>(),
        int endX = 0) :
        origin(origin),
        endX(endX)
    {}

    inline Coord<DIM> end() const
    {
        Coord<DIM> ret = origin;
        ret.x() = endX;
        return ret;
    }

    inline bool operator==(const Streak& other) const
    {
        return
            (origin == other.origin) &&
            (endX == other.endX);
    }

    inline Streak<DIM> operator+(const Coord<DIM>& displacement) const
    {
        return Streak<DIM>(
            origin + displacement,
            endX + displacement.x());
    }

    inline Streak<DIM> operator-(const Coord<DIM>& displacement) const
    {
        return Streak<DIM>(
            origin - displacement,
            endX - displacement.x());
    }

    inline Streak<DIM>&  operator+=(const Coord<DIM>& displacement)
    {
        origin += displacement;
        endX += displacement.x();
        return *this;
    }

    inline Streak<DIM>& operator-=(const Coord<DIM>& displacement)
    {
        origin -= displacement;
        endX -= displacement.x();
        return *this;
    }

    inline bool operator!=(const Streak& other) const
    {
        return !(*this == other);
    }

    inline int length() const
    {
        return endX - origin.x();
    }

    Coord<DIM> origin;
    int endX;
};

/**
 * The MPI typemap generator need to find out for which template
 * parameter values it should generate typemaps. It does so by
 * scanning all class members. Therefore this dummy class forces the
 * typemap generator to create MPI datatypes for Streaks with the
 * dimensions as specified below.
 */
class StreakMPIDatatypeHelper
{
    friend class Typemaps;
    Streak<1> a;
    Streak<2> b;
    Streak<3> c;
};

template<typename _CharT, typename _Traits, int _Dim>
std::basic_ostream<_CharT, _Traits>&
operator<<(std::basic_ostream<_CharT, _Traits>& __os,
           const Streak<_Dim>& streak)
{
    __os << "(origin: " << streak.origin << ", endX: " << streak.endX << ")";
    return __os;
}

}

#endif
