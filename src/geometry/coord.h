#ifndef LIBGEODECOMP_MISC_COORD_H
#define LIBGEODECOMP_MISC_COORD_H

// CodeGear's C++ compiler isn't compatible with boost::multi_array
// (at least the version that ships with C++ Builder 2009)
#ifndef __CODEGEARC__
#include <boost/multi_array.hpp>
#endif

#include <string>
#include <stdlib.h>
#include <libgeodecomp/config.h>
#include <libgeodecomp/geometry/fixedcoord.h>

#ifdef LIBGEODECOMP_FEATURE_BOOST_SERIALIZATION
#include <boost/serialization/is_bitwise_serializable.hpp>
#endif

#ifdef LIBGEODECOMP_FEATURE_QT
#include <QtCore/QSize>
#endif

namespace LibGeoDecomp {

/**
 * represents an integer coordinate.
 */
template<int DIMENSIONS>
class Coord;

template<>
class Coord<1>
{
public:
    friend class Typemaps;

    static Coord<1> diagonal(const int& nx)
    {
        return Coord<1>(nx);
    }

    inline explicit Coord(const int& nx=0)
    {
        c[0] = nx;
    }

    template<int X, int Y, int Z>
    inline explicit Coord(FixedCoord<X, Y, Z> /*unused*/)
    {
        c[0] = X;
    }

    /**
     * converts a linear index to a coordinate in a cuboid of size given by the Coord.
     */
    inline Coord<1> indexToCoord(int index) const
    {
        return Coord<1>(index);
    }

    /**
     * converts the coord to a linear index. This is good for addressing a linear array via Coords.
     */
    inline std::size_t toIndex(const Coord<1>& dim) const
    {
        return x();
    }

    int& x()
    {
        return c[0];
    }

    const int& x() const
    {
        return c[0];
    }

    inline int& operator[](const int& i)
    {
        return c[i];
    }

    inline const int& operator[](const int& i) const
    {
        return c[i];
    }

    inline bool operator==(const Coord& comp) const
    {
        return (x() == comp.x());
    }

    inline bool operator!=(const Coord& comp) const
    {
        return (x() != comp.x());
    }

    inline bool operator<(const Coord& comp) const
    {
        return (x() < comp.x());
    }

    inline Coord operator+(const Coord& addend) const
    {
        return Coord(x() + addend.x());
    }

    inline void operator+=(const Coord& addend)
    {
        x() += addend.x();
    }

    inline void operator-=(const Coord& minuend)
    {
        x() -= minuend.x();
    }

    inline Coord operator-(const Coord& minuend) const
    {
        return Coord(x() - minuend.x());
    }

    inline Coord operator-() const
    {
        return Coord(-x());
    }

    inline Coord operator*(const int& scale) const
    {
        return Coord(scale * x());
    }

    inline int operator*(const Coord& multiplier) const
    {
        return x() * multiplier.x();
    }

    inline Coord operator/(const int& divisor) const
    {
        return Coord(x()/ divisor);
    }

    inline const int& prod() const
    {
        return x();
    }

    inline const int& sum() const
    {
        return x();
    }

    inline Coord<1> (max)(const Coord<1>& other) const
    {
        return Coord<1>((std::max)(x(), other.x()));
    }

    inline Coord<1> (min)(const Coord<1>& other) const
    {
        return Coord<1>((std::min)(x(), other.x()));
    }

    boost::detail::multi_array::extent_gen<1ul> toExtents() const
    {
        return boost::extents[x()];
    }

    std::string toString() const
    {
        std::stringstream s;
        s << "(" << x() << ")";
        return s.str();
    }

private:
    int c[1];
};

template<>
class Coord<2>
{
public:
    friend class Typemaps;

    static Coord<2> diagonal(const int& nx)
    {
        return Coord<2>(nx, nx);
    }

    inline explicit Coord(const int& nx=0, const int& ny=0)
    {
        c[0] = nx;
        c[1] = ny;
    }

#ifdef LIBGEODECOMP_FEATURE_QT
    inline Coord(const QSize& size)
    {
        c[0] = size.width();
        c[1] = size.height();
    }
#endif

    template<int X, int Y, int Z>
    inline explicit Coord(FixedCoord<X, Y, Z> /*unused*/)
    {
        c[0] = X;
        c[1] = Y;
    }

    /**
     * converts a linear index to a coordinate in a cuboid of size given by the Coord
     */
    inline Coord<2> indexToCoord(int index) const
    {
        return Coord<2>(index % x(), index / x());
    }

    /**
     * converts the coord to a linear index. This is good for addressing a linear array via Coords.
     */
    inline std::size_t toIndex(const Coord<2>& dim) const
    {
        return std::size_t(y()) * dim.x() + x();
    }

    int& x()
    {
        return c[0];
    }

    const int& x() const
    {
        return c[0];
    }

    int& y()
    {
        return c[1];
    }

    const int& y() const
    {
        return c[1];
    }

    inline int& operator[](const int& i)
    {
        return c[i];
    }

    inline const int& operator[](const int& i) const
    {
        return c[i];
    }

    inline bool operator==(const Coord& comp) const
    {
        return (x() == comp.x()) && (y() == comp.y());
    }

    inline bool operator!=(const Coord& comp) const
    {
        return (x() != comp.x()) || (y() != comp.y());
    }

    inline bool operator<(const Coord& comp) const
    {
        return (x() < comp.x()) || ((x() == comp.x()) && (y() < comp.y()));
    }

    inline Coord operator+(const Coord& addend) const
    {
        return Coord(x() + addend.x(), y() + addend.y());
    }

    inline void operator+=(const Coord& addend)
    {
        x() += addend.x();
        y() += addend.y();
    }

    inline void operator-=(const Coord& minuend)
    {
        x() -= minuend.x();
        y() -= minuend.y();
    }

    inline Coord operator-(const Coord& minuend) const
    {
        return Coord(x() - minuend.x(), y() - minuend.y());
    }

    inline Coord operator-() const
    {
        return Coord(-x(), -y());
    }

    inline Coord operator*(const int& scale) const
    {
        return Coord(scale * x(), scale * y());
    }

    inline int operator*(const Coord& multiplier) const
    {
        return x() * multiplier.x() + y() * multiplier.y();
    }

    inline Coord operator/(const int& divisor) const
    {
        return Coord(x()/ divisor, y() / divisor);
    }

    inline int prod() const
    {
        return x() * y();
    }

    inline int sum() const
    {
        return x() + y();
    }

    inline Coord<2> (max)(const Coord<2>& other) const
    {
        return Coord<2>(
            (std::max)(x(), other.x()),
            (std::max)(y(), other.y()));
    }

    inline Coord<2> (min)(const Coord<2>& other) const
    {
        return Coord<2>(
            (std::min)(x(), other.x()),
            (std::min)(y(), other.y()));
    }

    boost::detail::multi_array::extent_gen<2ul> toExtents() const
    {
        return boost::extents[y()][x()];
    }

    std::string toString() const
    {
        std::stringstream s;
        s << "(" << x() << ", " << y() << ")";
        return s.str();
    }

private:
    int c[2];
};

template<>
class Coord<3>
{
public:
    friend class Typemaps;

    static Coord<3> diagonal(const int& nx)
    {
        return Coord<3>(nx, nx, nx);
    }

    inline explicit Coord(const int& nx=0, const int& ny=0, const int& nz=0)
    {
        c[0] = nx;
        c[1] = ny;
        c[2] = nz;
    }

    template<int X, int Y, int Z>
    inline explicit Coord(FixedCoord<X, Y, Z> /*unused*/)
    {
        c[0] = X;
        c[1] = Y;
        c[2] = Z;
    }

    /**
     * converts a linear index to a coordinate in a cuboid of size given by the Coord
     */
    inline Coord<3> indexToCoord(int index) const
    {
        int coordX = index % x();
        int remainder = index / x();
        int coordY = remainder % y();
        int coordZ = remainder / y();
        return Coord<3>(coordX, coordY, coordZ);
    }

    /**
     * converts the coord to a linear index. This is good for addressing a linear array via Coords.
     */
    inline std::size_t toIndex(const Coord<3>& dim) const
    {
        return
            std::size_t(z()) * dim.x() * dim.y() +
            y() * dim.x() +
            x();
    }

    int& x()
    {
        return c[0];
    }

    const int& x() const
    {
        return c[0];
    }

    int& y()
    {
        return c[1];
    }

    const int& y() const
    {
        return c[1];
    }

    int& z()
    {
        return c[2];
    }

    const int& z() const
    {
        return c[2];
    }

    inline int& operator[](const int& i)
    {
        return c[i];
    }

    inline const int& operator[](const int& i) const
    {
        return c[i];
    }

    inline bool operator==(const Coord& comp) const
    {
        return (x() == comp.x()) && (y() == comp.y()) && (z() == comp.z());
    }

    inline bool operator!=(const Coord& comp) const
    {
        return (x() != comp.x()) || (y() != comp.y()) || (z() != comp.z());
    }

    inline bool operator<(const Coord& comp) const
    {
        return
            (x() < comp.x()) ||
            ((x() == comp.x()) && (y() < comp.y())) ||
            ((x() == comp.x()) && (y() == comp.y()) && (z() < comp.z()));
    }

    inline Coord operator+(const Coord& addend) const
    {
        return Coord(x() + addend.x(), y() + addend.y(), z() + addend.z());
    }

    inline void operator+=(const Coord& addend)
    {
        x() += addend.x();
        y() += addend.y();
        z() += addend.z();
    }

    inline void operator-=(const Coord& minuend)
    {
        x() -= minuend.x();
        y() -= minuend.y();
        z() -= minuend.z();
    }

    inline Coord operator-(const Coord& minuend) const
    {
        return Coord(x() - minuend.x(), y() - minuend.y(), z() - minuend.z());
    }

    inline Coord operator-() const
    {
        return Coord(-x(), -y(), -z());
    }

    inline Coord operator*(const int& scale) const
    {
        return Coord(scale * x(), scale * y(), scale * z());
    }

    inline int operator*(const Coord& multiplier) const
    {
        return x() * multiplier.x() + y() * multiplier.y() + z() * multiplier.z();
    }

    inline Coord operator/(const int& divisor) const
    {
        return Coord(x()/ divisor, y() / divisor, z() / divisor);
    }

    inline int prod() const
    {
        return x() * y() * z();
    }

    inline int sum() const
    {
        return x() + y() + z();
    }

    inline Coord<3> (max)(const Coord<3>& other) const
    {
        return Coord<3>(
            (std::max)(x(), other.x()),
            (std::max)(y(), other.y()),
            (std::max)(z(), other.z()));
    }

    inline Coord<3> (min)(const Coord<3>& other) const
    {
        return Coord<3>(
            (std::min)(x(), other.x()),
            (std::min)(y(), other.y()),
            (std::min)(z(), other.z()));
    }

    boost::detail::multi_array::extent_gen<3ul> toExtents() const
    {
        return boost::extents[z()][y()][x()];
    }

    std::string toString() const
    {
        std::stringstream s;
        s << "(" << x() << ", " << y() << ", " << z() << ")";
        return s.str();
    }

private:
    int c[3];
};

template<typename _CharT, typename _Traits, int DIMENSIONS>
std::basic_ostream<_CharT, _Traits>&
operator<<(std::basic_ostream<_CharT, _Traits>& os,
           const LibGeoDecomp::Coord<DIMENSIONS>& coord)
{
    os << coord.toString();
    return os;
}

}

#ifdef LIBGEODECOMP_FEATURE_BOOST_SERIALIZATION
BOOST_IS_BITWISE_SERIALIZABLE(LibGeoDecomp::Coord<1>)
BOOST_IS_BITWISE_SERIALIZABLE(LibGeoDecomp::Coord<2>)
BOOST_IS_BITWISE_SERIALIZABLE(LibGeoDecomp::Coord<3>)
#endif

#endif
