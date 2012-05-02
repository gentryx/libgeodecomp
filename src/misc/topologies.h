#ifndef _libgeodecomp_misc_topologies_h_
#define _libgeodecomp_misc_topologies_h_

#include <stdexcept>
#include <iostream>
#include <libgeodecomp/misc/coord.h>

namespace LibGeoDecomp
{

/**
 * This class is a crutch to borrow functionality from the topologies
 * in order to map neighboring coordinates correctly on the borders of
 * the grid.
 */
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
        target->c[DIMINDEX - 1] = i;
        return CoordNormalizer<DIM, DIMINDEX - 1>(target, dimensions);
    }

private:
    Coord<DIM> *target;
    Coord<DIM> dimensions;
    Coord<DIM> edge;
};

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
        target->c[0] = i;
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
    template<typename N_MINUS_1_DIMENSIONAL, bool WRAP_EDGES>
    class NDimensional;

    template<bool WRAP_EDGES, int DIMENSION, typename COORD> 
    class NormalizeCoordElement;

    template<int DIMENSION, typename COORD> 
    class NormalizeCoordElement<true, DIMENSION, COORD>
    {
    public:
        inline int operator()(
            const COORD& coord, 
            const COORD& boundingBox) const
        {
            return 
                (boundingBox.c[DIMENSION] + coord.c[DIMENSION]) %
                boundingBox.c[DIMENSION];
        }
    };

    template<int DIMENSION, typename COORD> 
    class NormalizeCoordElement<false, DIMENSION, COORD>
    {
    public:
        inline int operator()(
            const COORD& coord, 
            const COORD& boundingBox) const
        {
            return coord.c[DIMENSION];
        }
    };

    template<bool WRAP_EDGES, int DIMENSION, typename COORD> 
    class IsOutOfBounds;

    template<int DIMENSION, typename COORD> 
    class IsOutOfBounds<true, DIMENSION, COORD>
    {
    public:
        inline bool operator()(
            const COORD& coord, 
            const COORD& boundingBox) const
        {
            return false;
        }
    };

    template<int DIMENSION, typename COORD> 
    class IsOutOfBounds<false, DIMENSION, COORD>
    {
    public:
        inline bool operator()(
            const COORD& coord, 
            const COORD& boundingBox) const
        {
            return
                (coord.c[DIMENSION] < 0) || 
                (coord.c[DIMENSION] >= 
                 boundingBox.c[DIMENSION]);
        }
    };

    template<int DIMENSION, typename COORD, typename TOPOLOGY>
    class IsOutOfBoundsHelper
    {
    public:
        inline bool operator()(
            const COORD& coord, 
            const COORD& boundingBox) const
        {
            return 
                IsOutOfBounds<TOPOLOGY::WrapEdges, DIMENSION, COORD>()(
                    coord, boundingBox) ||
                IsOutOfBoundsHelper<DIMENSION - 1, COORD, 
                                    typename TOPOLOGY::ParentTopology>()(
                                        coord, boundingBox);
        }
        
    };

    class ZeroDimensional;

    template<int DIMENSION, typename COORD>
    class IsOutOfBoundsHelper<DIMENSION, COORD, ZeroDimensional>
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

    class ZeroDimensional
    {
    public:
        const static int DIMENSIONS = 0;

        static inline bool wrapsAxis(const int& dim)
        {
            return false;
        }
    };

    template<typename N_MINUS_1_DIMENSIONAL, bool WRAP_EDGES>
    class NDimensional
    {
    public:
        // fixme: rename to DIM
        const static int DIMENSIONS = N_MINUS_1_DIMENSIONAL::DIMENSIONS + 1;
        const static bool WrapEdges = WRAP_EDGES;
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
                            ParentTopology::WrapEdges, 
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
                            ParentTopology::WrapEdges, 
                            1,
                            Coord<3> >()(coord, boundingBox)]\
                    [ NormalizeCoordElement<
                            ParentTopology::ParentTopology::WrapEdges, 
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
                return WrapEdges;
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
