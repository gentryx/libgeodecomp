#ifndef LIBGEODECOMP_GEOMETRY_FLOATCOORD_H
#define LIBGEODECOMP_GEOMETRY_FLOATCOORD_H

#include <libgeodecomp/misc/math.h>

// fixme: experiment 4996: unknown here
#ifdef _MSC_BUILD
#pragma warning( disable : 4996 )
#endif

// Kill warning 4514 in system headers
#ifdef _MSC_BUILD
#pragma warning( push )
#pragma warning( disable : 4514 )
#endif

#include <algorithm>
#include <sstream>

#ifdef _MSC_BUILD
#pragma warning( pop )
#endif

namespace LibGeoDecomp {

// Hardwire this warning to off as MSVC would otherwise complain about
// inline functions not being included in object files:
#ifdef _MSC_BUILD
#pragma warning( push )
#pragma warning( disable : 4514 )
#endif

/**
 * A real valued coordinate class. Can also be seen as a short,
 * fixed-size vector.
 *
 * Conversion operators and arithmetic operators from Coord to
 * FloatCoord are available to ease interoperability. Vice versa is
 * left out intentionally to avoid inadvertent conversions to Coord
 * (and thus losses of accuracy).
 */
template<int DIM>
class FloatCoord;

/**
 * see above
 */
template<>
class FloatCoord<1>
{
public:
    friend class BoostSerialization;
    friend class HPXSerialization;
    friend class Typemaps;
    typedef double ValueType;
    static const int DIM = 1;

    static inline FloatCoord<1> diagonal(double x)
    {
        return FloatCoord<1>(x);
    }

    explicit
    inline
    FloatCoord(const double x = 0)
    {
        c[0] = x;
    }

#ifdef __ICC
// disabling this warning as implicit type conversion is exactly our goal here:
#pragma warning push
#pragma warning (disable: 2304)
#endif
    template<template<int> class OTHER_COORD>
    inline
    FloatCoord(const OTHER_COORD<1>& p)
    {
        c[0] = p[0];
    }
#ifdef __ICC
#pragma warning pop
#endif

    inline
    double length() const
    {
        return fabs(c[0]);
    }

    inline
    FloatCoord<1> abs() const
    {
        return FloatCoord<1>(
            fabs(c[0]));
    }

    template<template<int> class OTHER_COORD>
    inline bool
    dominates(const OTHER_COORD<1>& other) const
    {
        return c[0] <= other[0];
    }

    template<template<int> class OTHER_COORD>
    inline bool
    strictlyDominates(const OTHER_COORD<1>& other) const
    {
        return c[0] < other[0];
    }

    template<template<int> class OTHER_COORD>
    inline
    FloatCoord<1> operator+(const OTHER_COORD<1>& a) const
    {
        return FloatCoord<1>(c[0] + a[0]);
    }

    template<template<int> class OTHER_COORD>
    inline
    FloatCoord<1> operator-(const OTHER_COORD<1>& a) const
    {
        return FloatCoord<1>(c[0] - a[0]);
    }

    inline FloatCoord<1> operator-() const
    {
        return FloatCoord<1>(-c[0]);
    }

    template<template<int> class OTHER_COORD>
    inline
    FloatCoord<1>& operator+=(const OTHER_COORD<1>& a)
    {
        c[0] += a[0];
        return *this;
    }

    template<template<int> class OTHER_COORD>
    inline
    FloatCoord<1>& operator-=(const OTHER_COORD<1>& a)
    {
        c[0] -= a[0];
        return *this;
    }

    inline
    FloatCoord<1>& operator/=(const double s)
    {
        c[0] /= s;
        return *this;
    }

    inline
    FloatCoord<1> operator/(double s) const
    {
        return FloatCoord<1>(c[0] / s);
    }

    template<template<int> class OTHER_COORD>
    inline
    FloatCoord<1> operator/(const OTHER_COORD<1>& a) const
    {
        return FloatCoord<1>(
            c[0] / a[0]);
    }

    template<template<int> class OTHER_COORD>
    inline
    double operator*(const OTHER_COORD<1>& a) const
    {
        return c[0] * a[0];
    }

    inline
    FloatCoord<1> operator*(double s) const
    {
        return FloatCoord<1>(c[0] * s);
    }

    inline
    FloatCoord<1>& operator*=(const double s)
    {
        c[0] *= s;
        return *this;
    }

    template<template<int> class OTHER_COORD>
    inline
    bool operator==(const OTHER_COORD<1>& a) const
    {
        return (c[0] == a[0]);
    }

    template<template<int> class OTHER_COORD>
    inline
    bool operator!=(const OTHER_COORD<1>& a) const
    {
        return !(*this == a);
    }

    template<template<int> class OTHER_COORD>
    inline
    bool operator<(const OTHER_COORD<1>& comp) const
    {
        return (c[0] < comp[0]);
    }

    inline double& operator[](const int i)
    {
        return c[i];
    }

    inline double operator[](const int i) const
    {
        return c[i];
    }

    template<template<int> class OTHER_COORD>
    inline
    FloatCoord scale(const OTHER_COORD<1>& other)
    {
        return FloatCoord(c[0] * other[0]);
    }

    inline double prod() const
    {
        return c[0];
    }

    inline double sum() const
    {
        return c[0];
    }

    inline FloatCoord<1> (max)(const FloatCoord<1>& other) const
    {
        return FloatCoord<1>(
            (std::max)(c[0], other[0]));
    }

    inline FloatCoord<1> (min)(const FloatCoord<1>& other) const
    {
        return FloatCoord<1>(
            (std::min)(c[0], other[0]));
    }

    inline double minElement() const
    {
        return c[0];
    }

    inline double maxElement() const
    {
        return c[0];
    }

    inline
    std::string toString() const
    {
        std::stringstream s;
        s << "(";
        s << c[0] << ")";
        return s.str();
    }

public:
    double c[1];
};

/**
 * see above
 */
template<>
class FloatCoord<2>
{
public:
    friend class BoostSerialization;
    friend class HPXSerialization;
    friend class Typemaps;
    typedef double ValueType;
    static const int DIM = 2;

    static inline FloatCoord<2> diagonal(double x)
    {
        return FloatCoord<2>(x, x);
    }

    explicit
    inline
    FloatCoord(
        const double x = 0,
        const double y = 0)
    {
        c[0] = x;
        c[1] = y;
    }

#ifdef __ICC
// disabling this warning as implicit type conversion is exactly our goal here:
#pragma warning push
#pragma warning (disable: 2304)
#endif
    template<template<int> class OTHER_COORD>
    inline
    FloatCoord(const OTHER_COORD<2>& p)
    {
        c[0] = p[0];
        c[1] = p[1];
    }
#ifdef __ICC
#pragma warning pop
#endif

    inline
    double length() const
    {
        return sqrt(c[0] * c[0] +
                    c[1] * c[1]);
    }

    inline
    FloatCoord<2> abs() const
    {
        return FloatCoord<2>(
            fabs(c[0]),
            fabs(c[1]));
    }

    template<template<int> class OTHER_COORD>
    inline bool
    dominates(const OTHER_COORD<2>& other) const
    {
        return
            (c[0] <= other[0]) &&
            (c[1] <= other[1]);
    }

    template<template<int> class OTHER_COORD>
    inline bool
    strictlyDominates(const OTHER_COORD<2>& other) const
    {
        return
            (c[0] < other[0]) &&
            (c[1] < other[1]);
    }

    template<template<int> class OTHER_COORD>
    inline
    FloatCoord<2> operator+(const OTHER_COORD<2>& a) const
    {
        return FloatCoord<2>(c[0] + a[0],
                             c[1] + a[1]);
    }

    template<template<int> class OTHER_COORD>
    inline
    FloatCoord<2> operator-(const OTHER_COORD<2>& a) const
    {
        return FloatCoord<2>(c[0] - a[0],
                             c[1] - a[1]);
    }

    inline FloatCoord<2> operator-() const
    {
        return FloatCoord<2>(-c[0], -c[1]);
    }

    template<template<int> class OTHER_COORD>
    inline
    FloatCoord<2>& operator+=(const OTHER_COORD<2>& a)
    {
        c[0] += a[0];
        c[1] += a[1];
        return *this;
    }

    template<template<int> class OTHER_COORD>
    inline
    FloatCoord<2>& operator-=(const OTHER_COORD<2>& a)
    {
        c[0] -= a[0];
        c[1] -= a[1];
        return *this;
    }

    inline
    FloatCoord<2>& operator/=(const double s)
    {
        c[0] /= s;
        c[1] /= s;
        return *this;
    }

    inline
    FloatCoord<2> operator/(double s) const
    {
        return FloatCoord<2>(
            c[0] / s,
            c[1] / s);
    }

    template<template<int> class OTHER_COORD>
    inline
    FloatCoord<2> operator/(const OTHER_COORD<2>& a) const
    {
        return FloatCoord<2>(
            c[0] / a[0],
            c[1] / a[1]);
    }

    template<template<int> class OTHER_COORD>
    inline
    double operator*(const OTHER_COORD<2>& a) const
    {
        return c[0] * a[0] + c[1] * a[1];
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

    template<template<int> class OTHER_COORD>
    inline
    bool operator==(const OTHER_COORD<2>& a) const
    {
        return (c[0] == a[0]) && (c[1] == a[1]);
    }

    template<template<int> class OTHER_COORD>
    inline
    bool operator!=(const OTHER_COORD<2>& a) const
    {
        return !(*this == a);
    }

    template<template<int> class OTHER_COORD>
    inline
    bool operator<(const OTHER_COORD<2>& comp) const
    {
        return
            (c[0] <  comp[0]) ||
            ((c[0] == comp[0]) && (c[1] <  comp[1]));
    }

    inline double& operator[](const int i)
    {
        return c[i];
    }

    inline double operator[](const int i) const
    {
        return c[i];
    }

    template<template<int> class OTHER_COORD>
    inline
    FloatCoord scale(const OTHER_COORD<2>& other)
    {
        return FloatCoord(c[0] * other[0],
                          c[1] * other[1]);
    }

    inline double prod() const
    {
        return c[0] * c[1];
    }

    inline double sum() const
    {
        return c[0] + c[1];
    }

    inline FloatCoord<2> (max)(const FloatCoord<2>& other) const
    {
        return FloatCoord<2>(
            (std::max)(c[0], other[0]),
            (std::max)(c[1], other[1]));
    }

    inline FloatCoord<2> (min)(const FloatCoord<2>& other) const
    {
        return FloatCoord<2>(
            (std::min)(c[0], other[0]),
            (std::min)(c[1], other[1]));
    }

    inline double minElement() const
    {
        return c[0] < c[1] ? c[0] : c[1];
    }

    inline double maxElement() const
    {
        return c[0] > c[1] ? c[0] : c[1];
    }

    inline
    std::string toString() const
    {
        std::stringstream s;
        s << "(";
        for (int i = 0; i < 1; ++i)
            s << c[i] << ", ";
        s << c[1] << ")";
        return s.str();
    }

public:
    double c[2];
};

/**
 * see above
 */
template<>
class FloatCoord<3>
{
public:
    friend class BoostSerialization;
    friend class HPXSerialization;
    friend class Typemaps;
    typedef double ValueType;
    static const int DIM = 3;

    static inline FloatCoord<3> diagonal(double x)
    {
        return FloatCoord<3>(x, x, x);
    }

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

#ifdef __ICC
// disabling this warning as implicit type conversion is exactly our goal here:
#pragma warning push
#pragma warning (disable: 2304)
#endif
    template<template<int> class OTHER_COORD>
    inline
    FloatCoord(const OTHER_COORD<3>& p)
    {
        c[0] = p[0];
        c[1] = p[1];
        c[2] = p[2];
    }
#ifdef __ICC
#pragma warning pop
#endif

    inline
    double length() const
    {
        return sqrt(c[0] * c[0] +
                    c[1] * c[1] +
                    c[2] * c[2]);
    }

    inline
    FloatCoord<3> abs() const
    {
        return FloatCoord<3>(
            fabs(c[0]),
            fabs(c[1]),
            fabs(c[2]));
    }

    template<template<int> class OTHER_COORD>
    inline bool
    dominates(const OTHER_COORD<3>& other) const
    {
        return
            (c[0] <= other[0]) &&
            (c[1] <= other[1]) &&
            (c[2] <= other[2]);
    }

    template<template<int> class OTHER_COORD>
    inline bool
    strictlyDominates(const OTHER_COORD<3>& other) const
    {
        return
            (c[0] < other[0]) &&
            (c[1] < other[1]) &&
            (c[2] < other[2]);
    }

    template<template<int> class OTHER_COORD>
    inline
    FloatCoord<3> operator+(const OTHER_COORD<3>& a) const
    {
        return FloatCoord<3>(c[0] + a[0],
                             c[1] + a[1],
                             c[2] + a[2]);
    }

    template<template<int> class OTHER_COORD>
    inline
    FloatCoord<3> operator-(const OTHER_COORD<3>& a) const
    {
        return FloatCoord<3>(c[0] - a[0],
                             c[1] - a[1],
                             c[2] - a[2]);
    }

    inline FloatCoord<3> operator-() const
    {
        return FloatCoord<3>(-c[0], -c[1], -c[2]);
    }

    template<template<int> class OTHER_COORD>
    inline
    FloatCoord<3>& operator+=(const OTHER_COORD<3>& a)
    {
        c[0] += a[0];
        c[1] += a[1];
        c[2] += a[2];
        return *this;
    }

    template<template<int> class OTHER_COORD>
    inline
    FloatCoord<3>& operator-=(const OTHER_COORD<3>& a)
    {
        c[0] -= a[0];
        c[1] -= a[1];
        c[2] -= a[2];
        return *this;
    }

    inline
    FloatCoord<3>& operator/=(const double s)
    {
        c[0] /= s;
        c[1] /= s;
        c[2] /= s;
        return *this;
    }

    inline
    FloatCoord<3> operator/(double s) const
    {
        return FloatCoord<3>(
            c[0] / s,
            c[1] / s,
            c[2] / s);
    }

    template<template<int> class OTHER_COORD>
    inline
    FloatCoord<3> operator/(const OTHER_COORD<3>& a) const
    {
        return FloatCoord<3>(
            c[0] / a[0],
            c[1] / a[1],
            c[2] / a[2]);
    }

    template<template<int> class OTHER_COORD>
    inline
    double operator*(const OTHER_COORD<3>& a) const
    {
        return c[0] * a[0] + c[1] * a[1] + c[2] * a[2];
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

    template<template<int> class OTHER_COORD>
    inline
    bool operator==(const OTHER_COORD<3>& a) const
    {
        return (c[0] == a[0]) && (c[1] == a[1]) && (c[2] == a[2]);
    }

    template<template<int> class OTHER_COORD>
    inline
    bool operator!=(const OTHER_COORD<3>& a) const
    {
        return !(*this == a);
    }

    template<template<int> class OTHER_COORD>
    inline
    bool operator<(const OTHER_COORD<3>& comp) const
    {
        return
            (c[0] <  comp[0]) ||
            ((c[0] == comp[0]) && (c[1] <  comp[1])) ||
            ((c[0] == comp[0]) && (c[1] == comp[1]) && (c[2] <  comp[2]));
    }

    inline double& operator[](const int i)
    {
        return c[i];
    }

    inline double operator[](const int i) const
    {
        return c[i];
    }

    template<template<int> class OTHER_COORD>
    inline
    FloatCoord scale(const OTHER_COORD<3>& other)
    {
        return FloatCoord(c[0] * other[0],
                          c[1] * other[1],
                          c[2] * other[2]);
    }

    inline double prod() const
    {
        return c[0] * c[1] * c[2];
    }

    inline double sum() const
    {
        return c[0] + c[1] + c[2];
    }

    inline FloatCoord<3> (max)(const FloatCoord<3>& other) const
    {
        return FloatCoord<3>(
            (std::max)(c[0], other[0]),
            (std::max)(c[1], other[1]),
            (std::max)(c[2], other[2]));
    }

    inline FloatCoord<3> (min)(const FloatCoord<3>& other) const
    {
        return FloatCoord<3>(
            (std::min)(c[0], other[0]),
            (std::min)(c[1], other[1]),
            (std::min)(c[2], other[2]));
    }

    inline double minElement() const
    {
        return c[0] < c[1] ?
                  (c[0] < c[2] ? c[0] : c[2]) : (c[1] < c[2] ? c[1] : c[2]);
    }

    inline double maxElement() const
    {
        return c[0] > c[1] ?
                  (c[0] > c[2] ? c[0] : c[2]) : (c[1] > c[2] ? c[1] : c[2]);
    }

    inline FloatCoord<3> crossProduct(const FloatCoord<3>& other) const
    {
        return FloatCoord<3>(
            c[1] * other[2] - c[2] * other[1],
            c[2] * other[0] - c[0] * other[2],
            c[0] * other[1] - c[1] * other[0]);
    }

    inline
    std::string toString() const
    {
        std::stringstream s;
        s << "(";
        for (int i = 0; i < 2; ++i)
            s << c[i] << ", ";
        s << c[2] << ")";
        return s.str();
    }

public:
    double c[3];
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

template<typename _CharT, typename _Traits, int DIMENSIONS>
std::basic_ostream<_CharT, _Traits>&
operator<<(std::basic_ostream<_CharT, _Traits>& __os,
           const FloatCoord<DIMENSIONS>& coord)
{
    __os << coord.toString();
    return __os;
}

#ifdef _MSC_BUILD
#pragma warning( pop )
#endif

}

#endif
