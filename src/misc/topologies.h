#ifndef LIBGEODECOMP_MISC_TOPOLOGIES_H
#define LIBGEODECOMP_MISC_TOPOLOGIES_H

#include <stdexcept>
#include <iostream>
#include <libgeodecomp/misc/coord.h>

namespace LibGeoDecomp {

namespace TopologiesHelpers {

template<int DIM, class TOPOLOGY>
class WrapsAxis;

template<class TOPOLOGY>
class WrapsAxis<0, TOPOLOGY>
{
public:
    static const bool VALUE = TOPOLOGY::WRAP_AXIS0;
};

template<class TOPOLOGY>
class WrapsAxis<1, TOPOLOGY>
{
public:
    static const bool VALUE = TOPOLOGY::WRAP_AXIS1;
};

template<class TOPOLOGY>
class WrapsAxis<2, TOPOLOGY>
{
public:
    static const bool VALUE = TOPOLOGY::WRAP_AXIS2;
};

template<class TOPOLOGY>
class NormalizeEdges
{
public:
    Coord<1> operator()(const Coord<1>& coord, const Coord<1>& dim)
    {
        return Coord<1>(
            wrap(coord[0], dim[0]));
    }

    Coord<2> operator()(const Coord<2>& coord, const Coord<2>& dim)
    {
        return Coord<2>(
            wrap(coord[0], dim[0]),
            wrap(coord[1], dim[1]));
    }

    Coord<3> operator()(const Coord<3>& coord, const Coord<3>& dim)
    {
        return Coord<3>(
            wrap(coord[0], dim[0]),
            wrap(coord[1], dim[1]),
            wrap(coord[2], dim[2]));
    }

private:
    inline int wrap(int x, int dim)
    {
        if (x < 0) {
            return (x + dim) % dim;
        }
        if (x >= dim) {
            return x % dim;
        }

        return x;
    }
};

template<class TOPOLOGY>
class OutOfBounds
{
public:
    bool operator()(const Coord<1> coord, const Coord<1> dim)
    {
        return 
            ((!WrapsAxis<0, TOPOLOGY>::VALUE) && ((coord[0] < 0) || (coord[0] >= dim[0])));
    }

    bool operator()(const Coord<2> coord, const Coord<2> dim)
    {
        return 
            ((!WrapsAxis<0, TOPOLOGY>::VALUE) && ((coord[0] < 0) || (coord[0] >= dim[0]))) ||
            ((!WrapsAxis<1, TOPOLOGY>::VALUE) && ((coord[1] < 0) || (coord[1] >= dim[1])));
    }

    bool operator()(const Coord<3> coord, const Coord<3> dim)
    {
        return 
            ((!WrapsAxis<0, TOPOLOGY>::VALUE) && ((coord[0] < 0) || (coord[0] >= dim[0]))) ||
            ((!WrapsAxis<1, TOPOLOGY>::VALUE) && ((coord[1] < 0) || (coord[1] >= dim[1]))) ||
            ((!WrapsAxis<2, TOPOLOGY>::VALUE) && ((coord[2] < 0) || (coord[2] >= dim[2])));
    }
};

template<class TOPOLOGY>
class NormalizeCoord
{
public:
    template<int DIM>
    Coord<DIM> operator()(const Coord<DIM>& coord, const Coord<DIM>& dim)
    {
        if (OutOfBounds<TOPOLOGY>()(coord, dim)) {
            return Coord<DIM>::diagonal(-1);
        }

        return NormalizeEdges<TOPOLOGY>()(coord, dim);
    }
};

template<int DIM>
class Accessor;

template<>
class Accessor<1>
{
public:
    template<typename STORAGE, typename VALUE>
    void operator()(STORAGE& storage, VALUE **value, const Coord<1>& coord) const
    {
        *value = &storage[coord.x()];
    }

    template<typename STORAGE, typename VALUE>
    void operator()(const STORAGE& storage, const VALUE **value, const Coord<1>& coord) const
    {
        *value = &storage[coord.x()];
    }    
};

template<>
class Accessor<2>
{
public:
    template<typename STORAGE, typename VALUE>
    void operator()(STORAGE& storage, VALUE **value, const Coord<2>& coord) const
    {
        *value = &storage[coord.y()][coord.x()];
    }

    template<typename STORAGE, typename VALUE>
    void operator()(const STORAGE& storage, const VALUE **value, const Coord<2>& coord) const
    {
        *value = &storage[coord.y()][coord.x()];
    }    
};

template<>
class Accessor<3>
{
public:
    template<typename STORAGE, typename VALUE>
    void operator()(STORAGE& storage, VALUE **value, const Coord<3>& coord) const
    {
        *value = &storage[coord.z()][coord.y()][coord.x()];
    }

    template<typename STORAGE, typename VALUE>
    void operator()(const STORAGE& storage, const VALUE **value, const Coord<3>& coord) const
    {
        *value = &storage[coord.z()][coord.y()][coord.x()];
    }    
};

template<int DIMENSIONS, bool WRAP_DIM0, bool WRAP_DIM1, bool WRAP_DIM2>
class RawTopology
{
public:
    const static int DIM = DIMENSIONS;
    const static bool WRAP_AXIS0 = WRAP_DIM0;
    const static bool WRAP_AXIS1 = WRAP_DIM1;
    const static bool WRAP_AXIS2 = WRAP_DIM2;
};

template<int DIMENSIONS, bool WRAP_DIM0=false, bool WRAP_DIM1=false, bool WRAP_DIM2=false>
class Topology
{
public:
    typedef RawTopology<DIMENSIONS, WRAP_DIM0, WRAP_DIM1, WRAP_DIM2> MyRawTopology;
    static const int DIM = DIMENSIONS;

    template<typename GRID, int DIM>
    static inline const typename GRID::CellType& locate(
        const GRID& grid, 
        const Coord<DIM>& coord)
    {
        const Coord<DIM>& dim = grid.getDimensions();
        if (OutOfBounds<MyRawTopology>()(coord, dim)) {
            return grid.getEdgeCell();
        }

        typename GRID::CellType *ret;
        Accessor<DIM>()(grid, &ret, NormalizeEdges<MyRawTopology>()(coord, dim));
        return *ret;
    }

    template<typename GRID>
    static inline typename GRID::CellType& locate(
        GRID& grid, 
        const Coord<DIMENSIONS>& coord)
    {
        const Coord<DIMENSIONS>& dim = grid.getDimensions();
        if (OutOfBounds<MyRawTopology>()(coord, dim)) {
            return grid.getEdgeCell();
        }

        typename GRID::CellType *ret;
        Accessor<DIMENSIONS>()(grid, &ret, NormalizeEdges<MyRawTopology>()(coord, dim));
        return *ret;
    }

    template<int D>
    class WrapsAxis
    {
    public:
        static const bool VALUE = TopologiesHelpers::WrapsAxis<D, MyRawTopology>::VALUE;
    };

    static Coord<DIM> normalize(const Coord<DIMENSIONS>& coord, const Coord<DIMENSIONS>& dimensions)
    {
        return NormalizeCoord<MyRawTopology>()(coord, dimensions);
    }

    static bool isOutOfBounds(const Coord<DIM>& coord, const Coord<DIM>& dim) 
    {
        return TopologiesHelpers::OutOfBounds<MyRawTopology>()(coord, dim);
    }
    
    /**
     * Checks whether the current topoloy uses periodic boundary
     * conditions on the edges of the given dimendion dim. Only use
     * this when you need to set dim at runtime. In every other case
     * WrapsAxis<DIM>::VALUE is the prefered way of checking.
     */
    static bool wrapsAxis(const int& dim)
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
};

}

#endif
