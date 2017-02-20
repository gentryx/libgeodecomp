#ifndef LIBGEODECOMP_GEOMETRY_TOPOLOGIES_H
#define LIBGEODECOMP_GEOMETRY_TOPOLOGIES_H

#include <stdexcept>
#include <iostream>
#include <libgeodecomp/geometry/coord.h>
#include <libgeodecomp/misc/likely.h>

namespace LibGeoDecomp {

namespace TopologiesHelpers {

/**
 * Helper class for checking whether a given topology has periodic
 * boundary conditions in a given dimension.
 */
template<int DIM, class TOPOLOGY>
class WrapsAxis;

/**
 * see above
 */
template<class TOPOLOGY>
class WrapsAxis<0, TOPOLOGY>
{
public:
    static const bool VALUE = TOPOLOGY::WRAP_AXIS0;
};

/**
 * see above
 */
template<class TOPOLOGY>
class WrapsAxis<1, TOPOLOGY>
{
public:
    static const bool VALUE = TOPOLOGY::WRAP_AXIS1;
};

/**
 * see above
 */
template<class TOPOLOGY>
class WrapsAxis<2, TOPOLOGY>
{
public:
    static const bool VALUE = TOPOLOGY::WRAP_AXIS2;
};

/**
 * Helper for mapping coords into a defined range (dimension)
 */
template<class TOPOLOGY>
class NormalizeEdges
{
public:
    __host__ __device__
    Coord<1> operator()(const Coord<1>& coord, const Coord<1>& dim)
    {
        return Coord<1>(
            wrap(coord[0], dim[0]));
    }

    __host__ __device__
    Coord<2> operator()(const Coord<2>& coord, const Coord<2>& dim)
    {
        return Coord<2>(
            wrap(coord[0], dim[0]),
            wrap(coord[1], dim[1]));
    }

    __host__ __device__
    Coord<3> operator()(const Coord<3>& coord, const Coord<3>& dim)
    {
        return Coord<3>(
            wrap(coord[0], dim[0]),
            wrap(coord[1], dim[1]),
            wrap(coord[2], dim[2]));
    }

private:
    __host__ __device__
    inline int wrap(int x, int dim)
    {
        if (LGD_UNLIKELY(x < 0)) {
            return x + dim;
        }
        if (LGD_UNLIKELY(x >= dim)) {
            return x - dim;
        }

        return x;
    }
};

/**
 * Helper for checking whether a given coord is outside of a defined range
 */
template<class TOPOLOGY>
class OutOfBounds
{
public:
    __host__ __device__
    bool operator()(const Coord<1> coord, const Coord<1> dim)
    {
        return
            ((!WrapsAxis<0, TOPOLOGY>::VALUE) && ((coord[0] < 0) || (coord[0] >= dim[0])));
    }

    __host__ __device__
    bool operator()(const Coord<2> coord, const Coord<2> dim)
    {
        return
            ((!WrapsAxis<0, TOPOLOGY>::VALUE) && ((coord[0] < 0) || (coord[0] >= dim[0]))) ||
            ((!WrapsAxis<1, TOPOLOGY>::VALUE) && ((coord[1] < 0) || (coord[1] >= dim[1])));
    }

#if !defined(__NVCC__) || (__CUDACC_VER_MAJOR__ >= 8)
#ifdef __GNUG__
#if __GNUC__ > 4 || (__GNUC__ == 4 && (__GNUC_MINOR__ >= 6) )
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wstrict-overflow"
#endif
#endif
#endif
    __host__ __device__
    bool operator()(const Coord<3> coord, const Coord<3> dim)
    {
        return
            ((!WrapsAxis<0, TOPOLOGY>::VALUE) && ((coord[0] < 0) || (coord[0] >= dim[0]))) ||
            ((!WrapsAxis<1, TOPOLOGY>::VALUE) && ((coord[1] < 0) || (coord[1] >= dim[1]))) ||
            ((!WrapsAxis<2, TOPOLOGY>::VALUE) && ((coord[2] < 0) || (coord[2] >= dim[2])));
    }
#if !defined(__NVCC__) || (__CUDACC_VER_MAJOR__ >= 8)
#ifdef __GNUG__
#if __GNUC__ > 4 || (__GNUC__ == 4 && (__GNUC_MINOR__ >= 6) )
#pragma GCC diagnostic pop
#endif
#endif
#endif
};

/**
 * Maps coord to range depending on topology
 */
template<class TOPOLOGY>
class NormalizeCoord
{
public:
    template<int DIM>
    __host__ __device__
    Coord<DIM> operator()(const Coord<DIM>& coord, const Coord<DIM>& dim)
    {
        if (OutOfBounds<TOPOLOGY>()(coord, dim)) {
            return Coord<DIM>::diagonal(-1);
        }

        return NormalizeEdges<TOPOLOGY>()(coord, dim);
    }
};

/**
 * Helper class for direct access to topology specs
 */
template<int DIMENSIONS, bool WRAP_DIM0, bool WRAP_DIM1, bool WRAP_DIM2>
class RawTopology
{
public:
    const static int DIM = DIMENSIONS;
    const static bool WRAP_AXIS0 = WRAP_DIM0;
    const static bool WRAP_AXIS1 = WRAP_DIM1;
    const static bool WRAP_AXIS2 = WRAP_DIM2;
};

/**
 * The actual topology class delived by factory class Topologies
 */
template<int DIMENSIONS, bool WRAP_DIM0=false, bool WRAP_DIM1=false, bool WRAP_DIM2=false>
class Topology
{
public:
    typedef RawTopology<DIMENSIONS, WRAP_DIM0, WRAP_DIM1, WRAP_DIM2> RawTopologyType;
    static const int DIM = DIMENSIONS;

    template<typename GRID>
    static inline const typename GRID::value_type& locate(
        const GRID& grid,
        const Coord<DIM>& coord,
        const Coord<DIM>& gridDim,
        const typename GRID::value_type& edgeCell)
    {
        if (OutOfBounds<RawTopologyType>()(coord, gridDim)) {
            return edgeCell;
        }

        const Coord<DIM> normalizedCoord = NormalizeEdges<RawTopologyType>()(coord, gridDim);
        const std::size_t index = normalizedCoord.toIndex(gridDim);
        return grid[index];
    }

    template<typename GRID>
    static inline typename GRID::value_type& locate(
        GRID& grid,
        const Coord<DIM>& coord,
        const Coord<DIM>& gridDim,
        typename GRID::value_type& edgeCell)
    {
        if (OutOfBounds<RawTopologyType>()(coord, gridDim)) {
            return edgeCell;
        }

        const Coord<DIM> normalizedCoord = NormalizeEdges<RawTopologyType>()(coord, gridDim);
        const std::size_t index = normalizedCoord.toIndex(gridDim);
        return grid[index];
    }

    template<int D>
    class WrapsAxis
    {
    public:
        static const bool VALUE = TopologiesHelpers::WrapsAxis<D, RawTopologyType>::VALUE;
    };

    __host__ __device__
    static Coord<DIM> normalize(const Coord<DIMENSIONS>& coord, const Coord<DIMENSIONS>& dimensions)
    {
        return NormalizeCoord<RawTopologyType>()(coord, dimensions);
    }

    __host__ __device__
    static bool isOutOfBounds(const Coord<DIM>& coord, const Coord<DIM>& dim)
    {
        return TopologiesHelpers::OutOfBounds<RawTopologyType>()(coord, dim);
    }

    /**
     * Checks whether the current topoloy uses periodic boundary
     * conditions on the edges of the given dimendion dim. Only use
     * this when you need to set dim at runtime. In every other case
     * WrapsAxis<DIM>::VALUE is the prefered way of checking.
     */
    static bool wrapsAxis(int dim)
    {
        if (dim == 0) {
            return WrapsAxis<0>::VALUE;
        }
        if (dim == 1) {
            return WrapsAxis<1>::VALUE;
        }

        return WrapsAxis<2>::VALUE;
    }
};

}

/**
 * This class is a type factory which can be used to describe the
 * boundary conditions of a simulation model. Per axis a user may
 * choose whether to wrap accesses (periodic boundary condition) or to
 * cut accesses off (constant boundary).
 */
class Topologies
{
public:
    template<int DIM>
    class Cube
    {
    public:
        typedef TopologiesHelpers::Topology<DIM, false, false, false> Topology;
    };

    template<int DIM>
    class Torus
    {
    public:
        typedef TopologiesHelpers::Topology<DIM, true, true, true> Topology;
    };

    class Unstructured
    {
    public:
        class Topology
        {
        public:
            static const int DIM = 1;

            template<int D>
            class WrapsAxis
            {
            public:
                static const bool VALUE = false;
            };

            template<typename GRID>
            static inline const typename GRID::value_type& locate(
                const GRID& grid,
                const Coord<1>& coord,
                const Coord<1>& gridDim,
                const typename GRID::value_type& edgeCell)
            {
                return grid[coord.x()];
            }

            template<typename GRID>
            static inline typename GRID::value_type& locate(
                GRID& grid,
                const Coord<1>& coord,
                const Coord<1>& gridDim,
                const typename GRID::value_type& edgeCell)
            {
                return grid[coord.x()];
            }

            static Coord<1> normalize(
                const Coord<1>& coord,
                const Coord<1>& /* gridDim */)
            {
                return coord;
            }
        };
    };

};

}

#endif
