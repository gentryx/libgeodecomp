#ifndef LIBGEODECOMP_GEOMETRY_COORD_H
#define LIBGEODECOMP_GEOMETRY_COORD_H

// coord.h is typically one of the first headers to be pulled into any
// code using LibGeoDecomp. This makes it a good spot to resolve some
// include order troubles:

// HPX' config needs to be included before Boost's config:
#include <libgeodecomp/config.h>
#ifdef LIBGEODECOMP_WITH_HPX
#include <libgeodecomp/misc/cudaboostworkaround.h>
#include <hpx/config.hpp>
#include <hpx/serialization/array.hpp>
#include <hpx/serialization/serialize.hpp>
#endif

// For Intel MPI we need to source mpi.h before stdio.h:
#ifdef LIBGEODECOMP_WITH_MPI
#include <mpi.h>
#endif

#include <algorithm>
#include <cstdlib>
#include <string>
#include <sstream>

#include <libgeodecomp/geometry/fixedcoord.h>
#include <libgeodecomp/geometry/floatcoord.h>

#ifdef LIBGEODECOMP_WITH_BOOST_SERIALIZATION
#include <libgeodecomp/misc/cudaboostworkaround.h>
#include <boost/mpl/bool.hpp>
#include <boost/serialization/is_bitwise_serializable.hpp>
#endif

#ifdef LIBGEODECOMP_WITH_QT5

#ifdef __ICC
// disabling this warning as implicit type conversion is exactly our goal here:
#pragma warning push
#pragma warning (disable: 2304)
#endif

#ifdef __GNUC__
#ifdef __CUDACC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wattributes"
#endif
#endif

#include <QtCore/QSize>

#ifdef __GNUC__
#ifdef __CUDACC__
#pragma GCC diagnostic pop
#endif
#endif

#ifdef __ICC
#pragma warning pop
#endif

#endif

namespace LibGeoDecomp {

#ifdef __ICC
// disabling this warning as implicit type conversion is exactly our goal here:
#pragma warning push
#pragma warning (disable: 2304)
#endif

/**
 * represents an integer coordinate.
 */
template<int DIMENSIONS>
class Coord;

/**
 * see above
 */
template<>
class Coord<1>
{
public:
    friend class BoostSerialization;
    friend class HPXSerialization;
    friend class MemberFilterCudaTest;
    friend class MemberFilterTest;
    friend class Typemaps;

    typedef int ValueType;

    static const int DIM = 1;

    __host__ __device__
    static Coord<1> diagonal(int nx)
    {
        return Coord<1>(nx);
    }

    __host__ __device__
    inline explicit Coord(int nx=0)
    {
        c[0] = nx;
    }

    template<int X, int Y, int Z>
    __host__ __device__
    inline explicit Coord(FixedCoord<X, Y, Z> /*unused*/)
    {
        c[0] = X;
    }

    inline explicit Coord(const FloatCoord<1>& other)
    {
        c[0] = int(other[0]);
    }

#ifdef LIBGEODECOMP_WITH_CPP14
    inline Coord(std::initializer_list<int> list)
    {
        c[0] = list.size() ? *list.begin() : 0;
    }
#endif

#ifdef __CUDACC__
    inline Coord(const dim3& dim)
    {
        c[0] = dim.x;
    }

    inline operator dim3()
    {
        dim3 ret;

        ret.x = c[0];

        return ret;
    }
#endif

    inline Coord abs() const
    {
        return Coord(std::abs(x()));
    }

    /**
     * converts a linear index to a coordinate in a cuboid of size given by the Coord.
     */
    __host__ __device__
    inline Coord<1> indexToCoord(int index) const
    {
        return Coord<1>(index);
    }

    /**
     * converts the coord to a linear index. This is good for addressing a linear array via Coords.
     */
    __host__ __device__
    inline std::size_t toIndex(const Coord<1>& /* dim */) const
    {
        return std::size_t(x());
    }

    __host__ __device__
    int& x()
    {
        return c[0];
    }

    __host__ __device__
    int x() const
    {
        return c[0];
    }

    __host__ __device__
    inline int& operator[](int i)
    {
        return c[i];
    }

    __host__ __device__
    inline int operator[](int i) const
    {
        return c[i];
    }

    __host__ __device__
    inline bool operator==(const Coord& comp) const
    {
        return (x() == comp.x());
    }

    __host__ __device__
    inline bool operator!=(const Coord& comp) const
    {
        return (x() != comp.x());
    }

    __host__ __device__
    inline bool operator<(const Coord& comp) const
    {
        return (x() < comp.x());
    }

    __host__ __device__
    inline Coord operator+(const Coord& addend) const
    {
        return Coord(x() + addend.x());
    }

    __host__ __device__
    inline void operator+=(const Coord& addend)
    {
        x() += addend.x();
    }

    __host__ __device__
    inline void operator-=(const Coord& minuend)
    {
        x() -= minuend.x();
    }

    __host__ __device__
    inline Coord operator-(const Coord& minuend) const
    {
        return Coord(x() - minuend.x());
    }

    __host__ __device__
    inline Coord operator-() const
    {
        return Coord(-x());
    }

    __host__ __device__
    inline Coord operator*(int scale) const
    {
        return Coord(scale * x());
    }

    __host__ __device__
    inline Coord operator*(float scale) const
    {
        return Coord(int(scale * x()));
    }

    __host__ __device__
    inline Coord operator*(double scale) const
    {
        return Coord(int(scale * x()));
    }

    __host__ __device__
    inline int operator*(const Coord& multiplier) const
    {
        return x() * multiplier.x();
    }

    __host__ __device__
    inline Coord operator/(int divisor) const
    {
        return Coord(x() / divisor);
    }

    __host__ __device__
    inline Coord scale(const Coord<1>& scale) const
    {
        return Coord(scale.x() * x());
    }

    __host__ __device__
    inline int prod() const
    {
        return x();
    }

    __host__ __device__
    inline int sum() const
    {
        return x();
    }

    __host__ __device__
    inline Coord<1> (max)(const Coord<1>& other) const
    {
        return Coord<1>(
            x() > other.x() ? x() : other.x());
    }

    __host__ __device__
    inline Coord<1> (min)(const Coord<1>& other) const
    {
        return Coord<1>(
            x() < other.x() ? x() : other.x());
    }

    inline int minElement() const
    {
        return x();
    }

    inline int maxElement() const
    {
        return x();
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

/**
 * see above
 */
template<>
class Coord<2>
{
public:
    friend class BoostSerialization;
    friend class HPXSerialization;
    friend class MemberFilterCudaTest;
    friend class MemberFilterTest;
    friend class Typemaps;

    typedef int ValueType;

    static const int DIM = 2;

    __host__ __device__
    static Coord<2> diagonal(int nx)
    {
        return Coord<2>(nx, nx);
    }

    __host__ __device__
    inline explicit Coord(int nx=0, int ny=0)
    {
        c[0] = nx;
        c[1] = ny;
    }

#ifdef LIBGEODECOMP_WITH_QT5

    inline Coord(const QSize& size)
    {
        c[0] = size.width();
        c[1] = size.height();
    }
#endif

    template<int X, int Y, int Z>
    __host__ __device__
    inline explicit Coord(FixedCoord<X, Y, Z> /*unused*/)
    {
        c[0] = X;
        c[1] = Y;
    }

    inline explicit Coord(const FloatCoord<2>& other)
    {
        c[0] = int(other[0]);
        c[1] = int(other[1]);
    }

#ifdef LIBGEODECOMP_WITH_CPP14
    inline Coord(std::initializer_list<int> list)
    {
        int i = 0;
        for (int coord : list) {
            c[i] = coord;
            i++;
            if (i > 1) {
                break;
            }
        }
        for (; i < 2; i++) {
            c[i] = 0;
        }
    }
#endif

#ifdef __CUDACC__
    inline Coord(const dim3& dim)
    {
        c[0] = dim.x;
        c[1] = dim.y;
    }

    inline operator dim3()
    {
        dim3 ret;

        ret.x = c[0];
        ret.y = c[1];

        return ret;
    }
#endif

    inline Coord abs() const
    {
        return Coord(std::abs(x()), std::abs(y()));
    }

    /**
     * converts a linear index to a coordinate in a cuboid of size given by the Coord
     */
    __host__ __device__
    inline Coord<2> indexToCoord(int index) const
    {
        return Coord<2>(index % x(), index / x());
    }

    /**
     * converts the coord to a linear index. This is good for addressing a linear array via Coords.
     */
    __host__ __device__
    inline std::size_t toIndex(const Coord<2>& dim) const
    {
        return std::size_t(y()) * dim.x() + x();
    }

    __host__ __device__
    int& x()
    {
        return c[0];
    }

    __host__ __device__
    int x() const
    {
        return c[0];
    }

    __host__ __device__
    int& y()
    {
        return c[1];
    }

    __host__ __device__
    int y() const
    {
        return c[1];
    }

    __host__ __device__
    inline int& operator[](int i)
    {
        return c[i];
    }

    __host__ __device__
    inline int operator[](int i) const
    {
        return c[i];
    }

    __host__ __device__
    inline bool operator==(const Coord& comp) const
    {
        return (x() == comp.x()) && (y() == comp.y());
    }

    __host__ __device__
    inline bool operator!=(const Coord& comp) const
    {
        return (x() != comp.x()) || (y() != comp.y());
    }

    __host__ __device__
    inline bool operator<(const Coord& comp) const
    {
        return (x() < comp.x()) || ((x() == comp.x()) && (y() < comp.y()));
    }

    __host__ __device__
    inline Coord operator+(const Coord& addend) const
    {
        return Coord(x() + addend.x(), y() + addend.y());
    }

    __host__ __device__
    inline void operator+=(const Coord& addend)
    {
        x() += addend.x();
        y() += addend.y();
    }

    __host__ __device__
    inline void operator-=(const Coord& minuend)
    {
        x() -= minuend.x();
        y() -= minuend.y();
    }

    __host__ __device__
    inline Coord operator-(const Coord& minuend) const
    {
        return Coord(x() - minuend.x(), y() - minuend.y());
    }

    __host__ __device__
    inline Coord operator-() const
    {
        return Coord(-x(), -y());
    }

    __host__ __device__
    inline Coord operator*(int scale) const
    {
        return Coord(scale * x(), scale * y());
    }

    __host__ __device__
    inline Coord operator*(float scale) const
    {
        return Coord(int(scale * x()), int(scale * y()));
    }

    __host__ __device__
    inline Coord operator*(double scale) const
    {
        return Coord(int(scale * x()), int(scale * y()));
    }

    __host__ __device__
    inline int operator*(const Coord& multiplier) const
    {
        return x() * multiplier.x() + y() * multiplier.y();
    }

    __host__ __device__
    inline Coord operator/(int divisor) const
    {
        return Coord(x() / divisor, y() / divisor);
    }

    __host__ __device__
    inline Coord scale(const Coord<2>& scale) const
    {
        return Coord(scale.x() * x(),
                     scale.y() * y());
    }

    __host__ __device__
    inline int prod() const
    {
        return x() * y();
    }

    __host__ __device__
    inline int sum() const
    {
        return x() + y();
    }

    __host__ __device__
    inline Coord<2> (max)(const Coord<2>& other) const
    {
        return Coord<2>(
            x() > other.x() ? x() : other.x(),
            y() > other.y() ? y() : other.y());
    }

    __host__ __device__
    inline Coord<2> (min)(const Coord<2>& other) const
    {
        return Coord<2>(
            x() < other.x() ? x() : other.x(),
            y() < other.y() ? y() : other.y());
    }

    inline int minElement() const
    {
        return x() < y() ? x() : y();
    }

    inline int maxElement() const
    {
        return x() > y() ? x() : y();
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

/**
 * see above
 */
template<>
class Coord<3>
{
public:
    friend class BoostSerialization;
    friend class HPXSerialization;
    friend class MemberFilterCudaTest;
    friend class MemberFilterTest;
    friend class Typemaps;

    typedef int ValueType;

    static const int DIM = 3;

    __host__ __device__
    static Coord<3> diagonal(int nx)
    {
        return Coord<3>(nx, nx, nx);
    }

    __host__ __device__
    inline explicit Coord(int nx=0, int ny=0, int nz=0)
    {
        c[0] = nx;
        c[1] = ny;
        c[2] = nz;
    }

    template<int X, int Y, int Z>
    __host__ __device__
    inline explicit Coord(FixedCoord<X, Y, Z> /*unused*/)
    {
        c[0] = X;
        c[1] = Y;
        c[2] = Z;
    }

    inline explicit Coord(const FloatCoord<3>& other)
    {
        c[0] = int(other[0]);
        c[1] = int(other[1]);
        c[2] = int(other[2]);
    }

#ifdef LIBGEODECOMP_WITH_CPP14
    inline Coord(std::initializer_list<int> list)
    {
        int i = 0;
        for (int coord : list) {
            c[i] = coord;
            i++;
            if (i > 2) {
                break;
            }
        }
        for (; i < 3; i++) {
            c[i] = 0;
        }
    }
#endif

#ifdef __CUDACC__
    inline Coord(const dim3& dim)
    {
        c[0] = dim.x;
        c[1] = dim.y;
        c[2] = dim.z;
    }

    inline operator dim3()
    {
        dim3 ret;

        ret.x = c[0];
        ret.y = c[1];
        ret.z = c[2];

        return ret;
    }
#endif

    inline Coord abs()
    {
        return Coord(std::abs(x()), std::abs(y()), std::abs(z()));
    }

    /**
     * converts a linear index to a coordinate in a cuboid of size given by the Coord
     */
    __host__ __device__
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
    __host__ __device__
    inline std::size_t toIndex(const Coord<3>& dim) const
    {
        return
            std::size_t(z()) * dim.x() * dim.y() +
            y() * dim.x() +
            x();
    }

    __host__ __device__
    int& x()
    {
        return c[0];
    }

    __host__ __device__
    int x() const
    {
        return c[0];
    }

    __host__ __device__
    int& y()
    {
        return c[1];
    }

    __host__ __device__
    int y() const
    {
        return c[1];
    }

    __host__ __device__
    int& z()
    {
        return c[2];
    }

    __host__ __device__
    int z() const
    {
        return c[2];
    }

    __host__ __device__
    inline int& operator[](int i)
    {
        return c[i];
    }

    __host__ __device__
    inline int operator[](int i) const
    {
        return c[i];
    }

    __host__ __device__
    inline bool operator==(const Coord& comp) const
    {
        return (x() == comp.x()) && (y() == comp.y()) && (z() == comp.z());
    }

    __host__ __device__
    inline bool operator!=(const Coord& comp) const
    {
        return (x() != comp.x()) || (y() != comp.y()) || (z() != comp.z());
    }

    __host__ __device__
    inline bool operator<(const Coord& comp) const
    {
        return
            (x() < comp.x()) ||
            ((x() == comp.x()) && (y() < comp.y())) ||
            ((x() == comp.x()) && (y() == comp.y()) && (z() < comp.z()));
    }

    __host__ __device__
    inline Coord operator+(const Coord& addend) const
    {
        return Coord(x() + addend.x(), y() + addend.y(), z() + addend.z());
    }

    __host__ __device__
    inline void operator+=(const Coord& addend)
    {
        x() += addend.x();
        y() += addend.y();
        z() += addend.z();
    }

    __host__ __device__
    inline void operator-=(const Coord& minuend)
    {
        x() -= minuend.x();
        y() -= minuend.y();
        z() -= minuend.z();
    }

    __host__ __device__
    inline Coord operator-(const Coord& minuend) const
    {
        return Coord(x() - minuend.x(), y() - minuend.y(), z() - minuend.z());
    }

    __host__ __device__
    inline Coord operator-() const
    {
        return Coord(-x(), -y(), -z());
    }

    __host__ __device__
    inline Coord operator*(int scale) const
    {
        return Coord(scale * x(), scale * y(), scale * z());
    }

    __host__ __device__
    inline Coord operator*(float scale) const
    {
        return Coord(int(scale * x()), int(scale * y()), int(scale * z()));
    }


    __host__ __device__
    inline Coord operator*(double scale) const
    {
        return Coord(int(scale * x()), int(scale * y()), int(scale * z()));
    }

    __host__ __device__
    inline int operator*(const Coord& multiplier) const
    {
        return x() * multiplier.x() + y() * multiplier.y() + z() * multiplier.z();
    }

    __host__ __device__
    inline Coord operator/(int divisor) const
    {
        return Coord(x() / divisor,
                     y() / divisor,
                     z() / divisor);
    }

    __host__ __device__
    inline Coord scale(const Coord<3>& scale) const
    {
        return Coord(scale.x() * x(),
                     scale.y() * y(),
                     scale.z() * z());
    }

    __host__ __device__
    inline int prod() const
    {
        return x() * y() * z();
    }

    __host__ __device__
    inline int sum() const
    {
        return x() + y() + z();
    }

    __host__ __device__
    inline Coord<3> (max)(const Coord<3>& other) const
    {
        return Coord<3>(
            x() > other.x() ? x() : other.x(),
            y() > other.y() ? y() : other.y(),
            z() > other.z() ? z() : other.z());
    }

    __host__ __device__
    inline Coord<3> (min)(const Coord<3>& other) const
    {
        return Coord<3>(
            x() < other.x() ? x() : other.x(),
            y() < other.y() ? y() : other.y(),
            z() < other.z() ? z() : other.z());
    }

    inline int minElement() const
    {
        return x() < y() ?
                  (x() < z() ? x() : z()) : (y() < z() ? y() : z());
    }

    inline int maxElement() const
    {
        return x() > y() ?
                  (x() > z() ? x() : z()) : (y() > z() ? y() : z());
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

#ifdef __ICC
#pragma warning pop
#endif

}

#ifdef LIBGEODECOMP_WITH_BOOST_SERIALIZATION
BOOST_IS_BITWISE_SERIALIZABLE(LibGeoDecomp::Coord<1>)
BOOST_IS_BITWISE_SERIALIZABLE(LibGeoDecomp::Coord<2>)
BOOST_IS_BITWISE_SERIALIZABLE(LibGeoDecomp::Coord<3>)
#endif

#endif
