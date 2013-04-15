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
    Coord<3> operator()(const Coord<1>& coord, const Coord<1>& dim)
    {
        return Coord<1>(
            wrap(coord[0], dim[0]));
    }

    Coord<3> operator()(const Coord<2>& coord, const Coord<2>& dim)
    {
        return Coord<3>(
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
    int wrap(int x, int dim)
    {
        return (x + dim) % dim;
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

template<int DIMENSIONS, bool WRAP_DIM0, bool WRAP_DIM1, bool WRAP_DIM2>
class RawTopology
{
public:
    const static int DIM = DIMENSIONS;
    const static bool WRAP_AXIS0 = WRAP_DIM0;
    const static bool WRAP_AXIS1 = WRAP_DIM1;
    const static bool WRAP_AXIS2 = WRAP_DIM2;

    template<typename STORAGE, typename VALUE>
    static void access(STORAGE& storage, VALUE *value, const Coord<1>& coord)
    {
        *value = (*storage)[coord.x()];
    }

    template<typename STORAGE, typename VALUE>
    static void access(const STORAGE& storage, const VALUE *value, const Coord<1>& coord)
    {
        *value = (*storage)[coord.x()];
    }

    template<typename STORAGE, typename VALUE>
    static void access(STORAGE& storage, VALUE *value, const Coord<2>& coord)
    {
        *value = (*storage)[coord.x()][coord.y()];
    }

    template<typename STORAGE, typename VALUE>
    static void access(const STORAGE& storage, const VALUE *value, const Coord<2>& coord)
    {
        *value = (*storage)[coord.x()][coord.y()];
    }

    template<typename STORAGE, typename VALUE>
    static void access(STORAGE& storage, VALUE *value, const Coord<3>& coord)
    {
        *value = (*storage)[coord.x()][coord.y()][coord.z()];
    }

    template<typename STORAGE, typename VALUE>
    static void access(const STORAGE& storage, const VALUE *value, const Coord<3>& coord)
    {
        *value = (*storage)[coord.x()][coord.y()][coord.z()];
    }
};

template<int DIMENSIONS, bool WRAP_DIM0, bool WRAP_DIM1, bool WRAP_DIM2>
class NewTopology
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
        MyRawTopology::access(grid, NormalizeEdges<MyRawTopology>()(coord, ret, dim));
        return *ret;
    }

};

}

// fixme: kill this
template<int DIM, class TOPOLOGY>
class WrapsAxisHelper
{
public:
    typedef typename TOPOLOGY::ParentTopology ParentTopology;
    static const bool VALUE = WrapsAxisHelper<DIM - 1, ParentTopology>::VALUE;
};

// fixme: kill this
template<class TOPOLOGY>
class WrapsAxisHelper<0, TOPOLOGY>
{
public:
    static const bool VALUE = TOPOLOGY::WRAP_EDGES;
};

/**
 * provides a compile-time random access interface to topology
 * features. Meant as an efficient alternative to
 * ZeroDimensional::wrapsAxis().
 */
// fixme: kill this
template<int DIM, class TOPOLOGY>
class WrapsAxis
{
public:
    static const bool VALUE = WrapsAxisHelper<TOPOLOGY::DIMENSIONS - DIM - 1, TOPOLOGY>::VALUE;

};

/**
 * This class is a crutch to borrow functionality from the topologies
 * in order to map neighboring coordinates correctly on the borders of
 * the grid.
 */
// fixme: kill this
template<int DIM, int DIMINDEX>
class CoordNormalizer
{
public:
    typedef Coord<DIM> CellType;

    CoordNormalizer(Coord<DIM> *c, Coord<DIM> dim) :
        target(c),
        dimensions(dim),
        edge(Coord<DIM>::diagonal(-1))
    {}

    const Coord<DIM>& getDimensions() const
    {
        return dimensions;
    }

    Coord<DIM>& getEdgeCell() 
    {
        return edge;
    }

    CoordNormalizer<DIM, DIMINDEX - 1> operator[](const int& i)
    {
        (*target)[DIMINDEX - 1] = i;
        return CoordNormalizer<DIM, DIMINDEX - 1>(target, dimensions);
    }

private:
    Coord<DIM> *target;
    Coord<DIM> dimensions;
    Coord<DIM> edge;
};

// fixme: kill this
template<int DIM>
class CoordNormalizer<DIM, 1>
{
public:
    typedef Coord<DIM> CellType;

    CoordNormalizer(Coord<DIM> *c, Coord<DIM> dim) :
        target(c),
        dimensions(dim)
    {}

    const Coord<DIM>& getDimensions() const
    {
        return dimensions;
    }

    Coord<DIM>& getEdgeCell() 
    {
        return edge;
    }

    Coord<DIM>& operator[](const int& i)
    {
        (*target)[0] = i;
        return *target;
    }

private:
    Coord<DIM> *target;
    Coord<DIM> dimensions;
    Coord<DIM> edge;
};

class Topologies
{
public:



    // fixme: kill this
    template<typename N_MINUS_1_DIMENSIONAL, bool WRAP_EDGES>
    class NDimensional;

    // fixme: kill this
    template<bool WRAP_EDGES, int DIMENSION, typename COORD> 
    class NormalizeCoordElement;

    // fixme: kill this
    template<int DIM, typename COORD> 
    class NormalizeCoordElement<true, DIM, COORD>
    {
    public:
        inline int operator()(
            const COORD& coord, 
            const COORD& dimensions) const
        {
            if ((coord[DIM] < 0) || (coord[DIM] >= dimensions[DIM])) { 
                return (dimensions[DIM] + coord[DIM]) % dimensions[DIM];
            }

            return coord[DIM];
        }
    };

    // fixme: kill this
    template<int DIM, typename COORD> 
    class NormalizeCoordElement<false, DIM, COORD>
    {
    public:
        inline int operator()(
            const COORD& coord, 
            const COORD& boundingBox) const
        {
            return coord[DIM];
        }
    };

    // fixme: kill this
    template<bool WRAP_EDGES, int DIM, typename COORD> 
    class IsOutOfBounds;

    // fixme: kill this
    template<int DIM, typename COORD> 
    class IsOutOfBounds<true, DIM, COORD>
    {
    public:
        inline bool operator()(
            const COORD& coord, 
            const COORD& boundingBox) const
        {
            return false;
        }
    };

    // fixme: kill this
    template<int DIM, typename COORD> 
    class IsOutOfBounds<false, DIM, COORD>
    {
    public:
        inline bool operator()(
            const COORD& coord, 
            const COORD& boundingBox) const
        {
            return 
                (coord[DIM] < 0) || 
                (coord[DIM] >= boundingBox[DIM]);
        }
    };

    // fixme: kill this
    template<int DIM, typename COORD, typename TOPOLOGY>
    class IsOutOfBoundsHelper
    {
    public:
        inline bool operator()(
            const COORD& coord, 
            const COORD& boundingBox) const
        {
            return 
                IsOutOfBounds<TOPOLOGY::WRAP_EDGES, DIM, COORD>()(
                    coord, boundingBox) ||
                IsOutOfBoundsHelper<DIM - 1, COORD, 
                                    typename TOPOLOGY::ParentTopology>()(
                                        coord, boundingBox);
        }
        
    };

    // fixme: kill this
    class ZeroDimensional;

    // fixme: kill this
    template<int DIM, typename COORD>
    class IsOutOfBoundsHelper<DIM, COORD, ZeroDimensional>
    {
    public:
        inline bool operator()(
            const COORD& coord, 
            const COORD& boundingBox) const
        {
            return false;
        }
        
    };

    template<int DIM>
    class Cube : public Cube<DIM - 1>
    {
    private:
        typedef Cube<DIM - 1> Parent;
        typedef typename Parent::Topology ParentTopology;

    public:
        typedef NDimensional<ParentTopology, false> Topology;
    };

    template<int DIM>
    class Torus : public Torus<DIM - 1>
    {
    private:
        typedef Torus<DIM - 1> Parent;
        typedef typename Parent::Topology ParentTopology;

    public:
        typedef NDimensional<ParentTopology, true> Topology;
    };

    // fixme: kill this
    class ZeroDimensional
    {
    public:
        const static int DIMENSIONS = 0;

        static inline bool wrapsAxis(const int& dim)
        {
            return false;
        }
    };

    // fixme: kill this
    template<typename N_MINUS_1_DIMENSIONAL, bool WRAP_EDGES_FLAG>
    class NDimensional
    {
    public:
        // fixme: rename to DIM
        const static int DIMENSIONS = N_MINUS_1_DIMENSIONAL::DIMENSIONS + 1;
        const static bool WRAP_EDGES = WRAP_EDGES_FLAG;
        typedef N_MINUS_1_DIMENSIONAL ParentTopology;

        /**
         * This class facilitates the computation of the actual
         * indices (frame/row/column) and at the same time fetches the
         * data directly from the grid. Combining these two operations
         * allows us (together with this template hell) to use the
         * same code, no matter which topology is being used, while
         * achieving a zero abstraction penalty.
         */
        template<int DIM, typename CELL> 
        class LocateHelper;

        template<typename CELL> 
        class LocateHelper<1, CELL>
        {
        public:
            template<typename STORAGE>
            inline const CELL& operator()(
                const STORAGE& storage, 
                const Coord<1>& coord,
                const Coord<1>& boundingBox) const
            {
                return (*this)(
                    const_cast<STORAGE&>(storage),
                    coord,
                    boundingBox);
            }

            template<typename STORAGE>
            inline CELL& operator()(
                STORAGE& storage, 
                const Coord<1>& coord,
                const Coord<1>& boundingBox) const
            {
                return storage\
                    [ NormalizeCoordElement<
                            WRAP_EDGES, 
                            0,
                            Coord<1> >()(coord, boundingBox)];
            }
        };

        template<typename CELL> 
        class LocateHelper<2, CELL>
        {
        public:
            template<typename STORAGE>
            inline const CELL& operator()(
                const STORAGE& storage, 
                const Coord<2>& coord,
                const Coord<2>& boundingBox) const
            {
                return (*this)(
                    const_cast<STORAGE&>(storage),
                    coord,
                    boundingBox);
            }

            template<typename STORAGE>
            inline CELL& operator()(
                STORAGE& storage, 
                const Coord<2>& coord,
                const Coord<2>& boundingBox) const
            {
                return storage\
                    [ NormalizeCoordElement<
                            WRAP_EDGES, 
                            1,
                            Coord<2> >()(coord, boundingBox)]\
                    [ NormalizeCoordElement<
                            ParentTopology::WRAP_EDGES, 
                            0,
                            Coord<2> >()(coord, boundingBox)];
            }
        };

        template<typename CELL> 
        class LocateHelper<3, CELL>
        {
        public:
            template<typename STORAGE>
            inline const CELL& operator()(
                const STORAGE& storage, 
                const Coord<3>& coord,
                const Coord<3>& boundingBox) const
            {
                return (*this)(
                    const_cast<STORAGE&>(storage),
                    coord,
                    boundingBox);
            }

            template<typename STORAGE>
            inline CELL& operator()(
                STORAGE& storage, 
                const Coord<3>& coord,
                const Coord<3>& boundingBox) const
            {
                return storage\
                    [ NormalizeCoordElement<
                            WRAP_EDGES, 
                            2,
                            Coord<3> >()(coord, boundingBox)
                     ]\
                    [ NormalizeCoordElement<
                            ParentTopology::WRAP_EDGES, 
                            1,
                            Coord<3> >()(coord, boundingBox)]\
                    [ NormalizeCoordElement<
                            ParentTopology::ParentTopology::WRAP_EDGES, 
                            0,
                            Coord<3> >()(coord, boundingBox)];
            }
        };

        static Coord<DIMENSIONS> normalize(
            const Coord<DIMENSIONS>& coord,
            const Coord<DIMENSIONS>& dimensions)
        {
            Coord<DIMENSIONS> res = Coord<DIMENSIONS>::diagonal(-1);
            CoordNormalizer<DIMENSIONS, DIMENSIONS> normalizer(
                &res, dimensions);
            locate(normalizer, coord);
            return res;
        }

        template<typename GRID, int DIM>
        static inline const typename GRID::CellType& locate(
            const GRID& grid, 
            const Coord<DIM>& coord) 
        {
            if (IsOutOfBoundsHelper<
                DIM - 1, 
                Coord<DIM>, 
                NDimensional >()(coord, grid.getDimensions()))
                return grid.getEdgeCell();

            return LocateHelper<DIM, typename GRID::CellType>()(
                grid, coord, grid.getDimensions());
        }

        template<typename GRID, int DIM>
        static inline typename GRID::CellType& locate(
            GRID& grid, 
            const Coord<DIM>& coord) 
        {
            if (IsOutOfBoundsHelper<
                DIM - 1, 
                Coord<DIM>, 
                NDimensional >()(coord, grid.getDimensions()))
                return grid.getEdgeCell();

            return LocateHelper<DIM, typename GRID::CellType>()(
                grid, coord, grid.getDimensions());
        }

        static inline bool wrapsAxis(const int& dim)
        {
            if (dim == DIMENSIONS - 1) {
                return WRAP_EDGES;
            } else {
                return ParentTopology::wrapsAxis(dim);
            }
        }
    };
};

template<>
class Topologies::Cube<0>
{
public:
    typedef ZeroDimensional Topology;
};

template<>
class Topologies::Torus<0>
{
public:
    typedef ZeroDimensional Topology;
};


}
#endif
