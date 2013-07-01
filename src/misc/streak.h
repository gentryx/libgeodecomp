#ifndef LIBGEODECOMP_MISC_STREAK_H
#define LIBGEODECOMP_MISC_STREAK_H

#include <libgeodecomp/misc/coord.h>

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
    friend class Typemaps;
public:
    inline explicit Streak(
        const Coord<DIM>& origin=Coord<DIM>(),
        const int& endX=0) :
        origin(origin),
        endX(endX)
    {}

    std::string toString() const
    {
        std::ostringstream buffer;
        buffer << "(origin: " << origin << ", endX: " << endX << ")";
        return buffer.str();
    }

    bool operator==(const Streak& other) const
    {
        return
            origin == other.origin &&
            endX == other.endX;
    }

    int length() const
    {
        return endX - origin.x();
    }

    template <typename Archive>
    void serialize(Archive & ar, unsigned)
    {
        ar & origin;
        ar & endX;
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

}

template<typename _CharT, typename _Traits, int _Dim>
std::basic_ostream<_CharT, _Traits>&
operator<<(std::basic_ostream<_CharT, _Traits>& __os,
           const LibGeoDecomp::Streak<_Dim>& streak)
{
    __os << streak.toString();
    return __os;
}

#endif
