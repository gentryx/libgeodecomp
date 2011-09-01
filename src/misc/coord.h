#ifndef _libgeodecomp_misc_coord_h_
#define _libgeodecomp_misc_coord_h_

// CodeGear's C++ compiler isn't compatible with boost::multi_array
// (at least the version that ships with C++ Builder 2009)
#ifndef __CODEGEARC__
#include <boost/multi_array.hpp>
#endif

#include <string>
#include <stdlib.h>
#include <libgeodecomp/misc/supervector.h>

namespace LibGeoDecomp {

/**
 * represents an integer coordinate.
 */
template<int DIMENSIONS>
class Coord;

template<>
class Coord<1>
{
    friend class Typemaps;

public:
    typedef SuperVector<Coord> Vector;

    int c[1];

    inline explicit Coord(int nx=0) 
    {
        c[0] = nx;
    }

    int& x() 
    { 
        return c[0]; 
    }

    const int& x() const
    { 
        return c[0]; 
    }

    inline bool operator==(const Coord& comp) const
    {
        return (x() == comp.x());
    }

    inline const int& prod() const
    {
        return x();
    }

    inline const int& sum() const
    {
        return x();
    }

    inline Coord<1> max(const Coord<1>& other) const 
    {
        return Coord<1>(std::max(x(), other.x()));
    }

    inline Coord<1> min(const Coord<1>& other) const 
    {
        return Coord<1>(std::min(x(), other.x()));
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
};

template<>
class Coord<2>
{
    friend class Typemaps;

public:
    typedef SuperVector<Coord> Vector;

    int c[2];

    inline explicit Coord(int nx=0, int ny=0) 
    {
        c[0] = nx;
        c[1] = ny;
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

    inline int operator*(const Coord& multiplyer) const
    {
        return x() * multiplyer.x() + y() * multiplyer.y();
    }

    inline Coord operator/(const int& divisor) const
    {
        return Coord(x()/ divisor, y() / divisor);
    }

    inline int prod() const
    {
        return x() * y();
    }

    inline const int sum() const
    {
        return x() + y();
    }

    inline Coord<2> max(const Coord<2>& other) const 
    {
        return Coord<2>(
            std::max(x(), other.x()),
            std::max(y(), other.y()));
    }

    inline Coord<2> min(const Coord<2>& other) const 
    {
        return Coord<2>(
            std::min(x(), other.x()),
            std::min(y(), other.y()));
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
};

template<>
class Coord<3>
{
    friend class Typemaps;

public:
    typedef SuperVector<Coord> Vector;

    int c[3];

    inline explicit Coord(int nx=0, int ny=0, int nz=0) 
    {
        c[0] = nx;
        c[1] = ny;
        c[2] = nz;
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

    inline int operator*(const Coord& multiplyer) const
    {
        return x() * multiplyer.x() + y() * multiplyer.y() + z() * multiplyer.z();
    }

    inline Coord operator/(const int& divisor) const
    {
        return Coord(x()/ divisor, y() / divisor, z() / divisor);
    }

    inline int prod() const
    {
        return x() * y() * z();
    }

    inline const int sum() const
    {
        return x() + y() + z();
    }

    inline Coord<3> max(const Coord<3>& other) const 
    {
        return Coord<3>(
            std::max(x(), other.x()),
            std::max(y(), other.y()),
            std::max(z(), other.z()));
    }

    inline Coord<3> min(const Coord<3>& other) const 
    {
        return Coord<3>(
            std::min(x(), other.x()),
            std::min(y(), other.y()),
            std::min(z(), other.z()));
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
};

template<int DIM>
class CoordDiagonal;

template<>
class CoordDiagonal<1>
{
public:
    Coord<1> operator()(const int& d)
    {
        return Coord<1>(d);
    }
};

template<>
class CoordDiagonal<2>
{
public:
    Coord<2> operator()(const int& d)
    {
        return Coord<2>(d, d);
    }
};

template<>
class CoordDiagonal<3>
{
public:
    Coord<3> operator()(const int& d)
    {
        return Coord<3>(d, d, d);
    }
};

/**
 * converts a linear index to a coordinate in a cuboid of size dim
 */
template<int DIM>
class IndexToCoord
{};

template<>
class IndexToCoord<1>
{
public:
    Coord<1> operator()(const int& index, const Coord<1>& dim)
    {
        return Coord<1>(index);
    }
};

template<>
class IndexToCoord<2>
{
public:
    Coord<2> operator()(const int& index, const Coord<2>& dim)
    {
        return Coord<2>(index % dim.x(), index / dim.x());
    }
};

template<>
class IndexToCoord<3>
{
public:
    Coord<3> operator()(const int& index, const Coord<3>& dim)
    {
        int x = index % dim.x();
        int remainder = index / dim.x();
        int y = remainder % dim.y();
        int z = remainder / dim.y();
        return Coord<3>(x, y, z);
    }
};

template<int DIM>
class CoordToIndex
{};

template<>
class CoordToIndex<1>
{
public:
    long long operator()(const Coord<1>& c, const Coord<1>& dim)
    {
        return c.x();
    }
};

template<>
class CoordToIndex<2>
{
public:
    long long operator()(const Coord<2>& c, const Coord<2>& dim)
    {
        return ((long long)c.y()) * dim.x() + c.x();
    }
};

template<>
class CoordToIndex<3>
{
public:
    long long operator()(const Coord<3>& c, const Coord<3>& dim)
    {
        return 
            ((long long)c.z()) * dim.x() * dim.y() + 
            ((long long)c.y()) * dim.x() + 
            c.x();
    }
};


}

template<typename _CharT, typename _Traits, int DIMENSIONS>
std::basic_ostream<_CharT, _Traits>&
operator<<(std::basic_ostream<_CharT, _Traits>& __os,
           const LibGeoDecomp::Coord<DIMENSIONS>& coord)
{
    __os << coord.toString();
    return __os;
}

#endif
