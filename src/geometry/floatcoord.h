#ifndef LIBGEODECOMP_GEOMETRY_FLOATCOORD_H
#define LIBGEODECOMP_GEOMETRY_FLOATCOORD_H

#include <libgeodecomp/geometry/coord.h>

#include <cmath>
#include <sstream>

namespace LibGeoDecomp {

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

template<>
class FloatCoord<1>
{
public:
    friend class Serialization;
    friend class Typemaps;

    explicit
    inline
    FloatCoord(const double x = 0)
    {
        c[0] = x;
    }

    template<template<int> class OTHER_COORD>
    inline
    FloatCoord(const OTHER_COORD<1>& p)
    {
        c[0] = p[0];
    }

    inline
    double length() const
    {
        return fabs(c[0]);
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
    FloatCoord<1> operator/(const double& s) const
    {
        return FloatCoord<1>(c[0] / s);
    }

    template<template<int> class OTHER_COORD>
    inline
    double operator*(const OTHER_COORD<1>& a) const
    {
        return c[0] * a[0];
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

    inline double& operator[](const int i)
    {
        return c[i];
    }

    inline const double& operator[](const int i) const
    {
        return c[i];
    }

    template<template<int> class OTHER_COORD>
    inline
    FloatCoord scale(const OTHER_COORD<1> other)
    {
        return FloatCoord(c[0] * other[0]);
    }

    inline const double& prod() const
    {
        return c[0];
    }

    inline const double& sum() const
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

template<>
class FloatCoord<2>
{
public:
    friend class Serialization;
    friend class Typemaps;

    explicit
    inline
    FloatCoord(
        const double x = 0,
        const double y = 0)
    {
        c[0] = x;
        c[1] = y;
    }

    template<template<int> class OTHER_COORD>
    inline
    FloatCoord(const OTHER_COORD<2>& p)
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
    FloatCoord<2> operator/(const double& s) const
    {
        return FloatCoord<2>(
            c[0] / s,
            c[1] / s);
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

    inline double& operator[](const int i)
    {
        return c[i];
    }

    inline const double& operator[](const int i) const
    {
        return c[i];
    }

    template<template<int> class OTHER_COORD>
    inline
    FloatCoord scale(const OTHER_COORD<2> other)
    {
        return FloatCoord(c[0] * other[0],
                          c[1] * other[1]);
    }

    inline const double prod() const
    {
        return c[0] * c[1];
    }

    inline const double sum() const
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

template<>
class FloatCoord<3>
{
public:
    friend class Serialization;
    friend class Typemaps;

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

    template<template<int> class OTHER_COORD>
    inline
    FloatCoord(const OTHER_COORD<3>& p)
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
    FloatCoord<3> operator/(const double& s) const
    {
        return FloatCoord<3>(
            c[0] / s,
            c[1] / s,
            c[2] / s);
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

    inline double& operator[](const int i)
    {
        return c[i];
    }

    inline const double& operator[](const int i) const
    {
        return c[i];
    }

    template<template<int> class OTHER_COORD>
    inline
    FloatCoord scale(const OTHER_COORD<3> other)
    {
        return FloatCoord(c[0] * other[0],
                          c[1] * other[1],
                          c[2] * other[2]);
    }

    inline const double prod() const
    {
        return c[0] * c[1] * c[2];
    }

    inline const double sum() const
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
