#ifndef LIBGEODECOMP_MISC_FLOATCOORD_H
#define LIBGEODECOMP_MISC_FLOATCOORD_H

#include <cmath>
#include <sstream>
#include <libgeodecomp/geometry/coord.h>
#include <libgeodecomp/geometry/floatcoordbase.h>

namespace LibGeoDecomp {

/**
 * A real valued coordinate class. Can also be seen as a short,
 * fixed-size vector.
 */
template<int DIM>
class FloatCoord;

template<>
class FloatCoord<1> : public FloatCoordBase<1>
{
public:
    friend class Typemaps;

    explicit
    inline
    FloatCoord(const double x = 0)
    {
        c[0] = x;
    }

    inline
    FloatCoord(const Coord<1>& p)
    {
        c[0] = p[0];
    }

    inline
    double length() const
    {
        return fabs(c[0]);
    }

    inline
    const double& sum() const
    {
        return c[0];
    }

    inline
    FloatCoord<1> operator+(const FloatCoord<1>& a) const
    {
        return FloatCoord<1>(c[0] + a.c[0]);
    }

    inline
    FloatCoord<1> operator-(const FloatCoord<1>& a) const
    {
        return FloatCoord<1>(c[0] - a.c[0]);
    }

    inline
    FloatCoord<1>& operator+=(const FloatCoord<1>& a)
    {
        c[0] += a.c[0];
        return *this;
    }

    inline
    FloatCoord<1>& operator-=(const FloatCoord<1>& a)
    {
        c[0] -= a.c[0];
        return *this;
    }

    inline
    FloatCoord<1> operator*(const double& s) const
    {
        return FloatCoord<1>(c[0] * s);
    }

    inline
    FloatCoord<1>& operator*=(const double s)
    {
        c[0] *= s;
        return *this;
    }

    inline
    bool operator==(const FloatCoord<1>& a) const
    {
        return (c[0] == a.c[0]);
    }

    inline
    bool operator!=(const FloatCoord<1>& a) const
    {
        return !(*this == a);
    }

    template<typename ARCHIVE>
    void serialize(ARCHIVE& ar, unsigned)
    {
        ar & c[0];
    }
};

template<>
class FloatCoord<2> : public FloatCoordBase<2>
{
    friend class Typemaps;
public:
    explicit
    inline
    FloatCoord(
        const double x = 0,
        const double y = 0)
    {
        c[0] = x;
        c[1] = y;
    }

    inline
    FloatCoord(const Coord<2>& p)
    {
        c[0] = p[0];
        c[1] = p[1];
    }

    inline
    double length() const
    {
        return sqrt(c[0] * c[0] +
                    c[1] * c[1]);
    }

    inline
    double sum() const
    {
        return c[0] + c[1];
    }

    inline
    FloatCoord<2> operator+(const FloatCoord<2>& a) const
    {
        return FloatCoord<2>(c[0] + a.c[0],
                             c[1] + a.c[1]);
    }

    inline
    FloatCoord<2> operator-(const FloatCoord<2>& a) const
    {
        return FloatCoord<2>(c[0] - a.c[0],
                             c[1] - a.c[1]);
    }

    inline
    FloatCoord<2>& operator+=(const FloatCoord<2>& a)
    {
        c[0] += a.c[0];
        c[1] += a.c[1];
        return *this;
    }

    inline
    FloatCoord<2>& operator-=(const FloatCoord<2>& a)
    {
        c[0] -= a.c[0];
        c[1] -= a.c[1];
        return *this;
    }

    inline
    FloatCoord<2> operator*(const double s) const
    {
        return FloatCoord<2>(c[0] * s, c[1] * s);
    }

    inline
    FloatCoord<2>& operator*=(const double s)
    {
        c[0] *= s;
        c[1] *= s;
        return *this;
    }

    inline
    bool operator==(const FloatCoord<2>& a) const
    {
        return (c[0] == a.c[0]) && (c[1] == a.c[1]);
    }

    inline
    bool operator!=(const FloatCoord<2>& a) const
    {
        return !(*this == a);
    }

    template<typename ARCHIVE>
    void serialize(ARCHIVE& ar, unsigned)
    {
        ar & c[0];
        ar & c[1];
    }
};

template<>
class FloatCoord<3> : public FloatCoordBase<3>
{
    friend class Typemaps;
public:
    explicit
    inline
    FloatCoord(
        const double x = 0,
        const double y = 0,
        const double z = 0)
    {
        c[0] = x;
        c[1] = y;
        c[2] = z;
    }

    inline
    FloatCoord(const Coord<3>& p)
    {
        c[0] = p[0];
        c[1] = p[1];
        c[2] = p[2];
    }

    inline
    double length() const
    {
        return sqrt(c[0] * c[0] +
                    c[1] * c[1] +
                    c[2] * c[2]);
    }

    inline
    double sum() const
    {
        return c[0] + c[1] + c[2];
    }

    inline
    FloatCoord<3> operator+(const FloatCoord<3>& a) const
    {
        return FloatCoord<3>(c[0] + a.c[0],
                             c[1] + a.c[1],
                             c[2] + a.c[2]);
    }

    inline
    FloatCoord<3> operator-(const FloatCoord<3>& a) const
    {
        return FloatCoord<3>(c[0] - a.c[0],
                             c[1] - a.c[1],
                             c[2] - a.c[2]);
    }

    inline
    FloatCoord<3>& operator+=(const FloatCoord<3>& a)
    {
        c[0] += a.c[0];
        c[1] += a.c[1];
        c[2] += a.c[2];
        return *this;
    }

    inline
    FloatCoord<3>& operator-=(const FloatCoord<3>& a)
    {
        c[0] -= a.c[0];
        c[1] -= a.c[1];
        c[2] -= a.c[2];
        return *this;
    }

    inline
    FloatCoord<3> operator*(const double s) const
    {
        return FloatCoord<3>(c[0] * s, c[1] * s, c[2] * s);
    }

    inline
    FloatCoord<3>& operator*=(const double s)
    {
        c[0] *= s;
        c[1] *= s;
        c[2] *= s;
        return *this;
    }

    inline
    bool operator==(const FloatCoord<3>& a) const
    {
        return (c[0] == a.c[0]) && (c[1] == a.c[1]) && (c[2] == a.c[2]);
    }

    inline
    bool operator!=(const FloatCoord<3>& a) const
    {
        return !(*this == a);
    }

    template<typename ARCHIVE>
    void serialize(ARCHIVE& ar, unsigned)
    {
        ar & c[0];
        ar & c[1];
        ar & c[2];
    }
};

/**
 * The MPI typemap generator need to find out for which template
 * parameter values it should generate typemaps. It does so by
 * scanning all class members. Therefore this dummy class forces the
 * typemap generator to create MPI datatypes for FloatCoord with the
 * dimensions as specified below.
 */
class FloatCoordMPIDatatypeHelper
{
    friend class Typemaps;
    FloatCoord<1> a;
    FloatCoord<2> b;
    FloatCoord<3> c;
};

}

template<typename _CharT, typename _Traits, int DIMENSIONS>
std::basic_ostream<_CharT, _Traits>&
operator<<(std::basic_ostream<_CharT, _Traits>& __os,
           const LibGeoDecomp::FloatCoord<DIMENSIONS>& coord)
{
    __os << coord.toString();
    return __os;
}

#endif
