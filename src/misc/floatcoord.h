#ifndef _libgeodecomp_misc_floatcoord_h_
#define _libgeodecomp_misc_floatcoord_h_

#include <cmath>
#include <sstream>
#include <boost/array.hpp>

#include <libgeodecomp/misc/coord.h>

namespace LibGeoDecomp {

/**
 * A real valued coordinate class. Can also be seen as a short,
 * fixed-size vector.
 */
template<int DIM>
class FloatCoordBase 
{
public:
    inline
    std::string toString() const
    {
        std::stringstream s;
        s << "(";
        for (int i = 0; i < DIM - 1; ++i)
            s << c[i] << ", ";
        s << c[DIM - 1] << ")";
        return s.str();
    }

    boost::array<double, DIM> c;
};

template<int DIM>
class FloatCoord;

template<>
class FloatCoord<1> : public FloatCoordBase<1>
{
public:
    explicit
    inline
    FloatCoord(const double& x = 0) 
    {
        c[0] = x;
    }

    inline
    FloatCoord(const Coord<1>& p) 
    {
        c[0] = p.c[0];
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
    FloatCoord<1>& operator*=(const double& s)
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
};

template<>
class FloatCoord<2> : public FloatCoordBase<2>
{
public:

    explicit
    inline
    FloatCoord(
        const double& x = 0, 
        const double& y = 0) 
    {
        c[0] = x;
        c[1] = y;
    }

    inline
    FloatCoord(const Coord<2>& p) 
    {
        c[0] = p.c[0];
        c[1] = p.c[1];
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
    FloatCoord<2> operator*(const double& s) const
    {
        return FloatCoord<2>(c[0] * s, c[1] * s);
    }

    inline 
    FloatCoord<2>& operator*=(const double& s)
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
};

template<>
class FloatCoord<3> : public FloatCoordBase<3>
{
public:

    explicit
    inline
    FloatCoord(
        const double& x = 0, 
        const double& y = 0, 
        const double& z = 0)
    {
        c[0] = x;
        c[1] = y;
        c[2] = z;
    }

    inline
    FloatCoord(const Coord<3>& p) 
    {
        c[0] = p.c[0];
        c[1] = p.c[1];
        c[2] = p.c[2];
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
    FloatCoord<3> operator*(const double& s) const
    {
        return FloatCoord<3>(c[0] * s, c[1] * s, c[2] * s);
    }

    inline 
    FloatCoord<3>& operator*=(const double& s)
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
