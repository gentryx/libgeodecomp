#ifndef _libgeodecomp_misc_linepointerassembly_h_
#define _libgeodecomp_misc_linepointerassembly_h_

#include <libgeodecomp/misc/stencils.h>
#include <libgeodecomp/misc/streak.h>

namespace LibGeoDecomp {

namespace LinePointerAssemblyHelpers {
/**
 * Utility class to deduce the X coordinate from a streak using the
 * original stencil's shape. See LinePointerUpdateFunctor for a
 * description of the individual pointers' meaning.
 */
template<int X>
class CalcXCoord;

template<>
class CalcXCoord<-1>
{
public:
    template<int DIM>
    int operator()(const Streak<DIM>& streak)
    {
        return streak.origin.x() - 1;
    }
};

template<>
class CalcXCoord<0>
{
public:
    template<int DIM>
    int operator()(const Streak<DIM>& streak)
    {
        return streak.origin.x() + 0;
    }
};

template<>
class CalcXCoord<1>
{
public:
    template<int DIM>
    int operator()(const Streak<DIM>& streak)
    {
        return streak.endX;
    }
};

template<int DIM>
class DetermineLinePointerCoord;

template<>
class DetermineLinePointerCoord<1>
{
public:
    template<int X, int Y, int Z>
    Coord<1> operator()(Streak<1> streak, FixedCoord<X, Y, Z>)
    {
        return Coord<1>(CalcXCoord<X>()(streak));
    }
};

template<>
class DetermineLinePointerCoord<2>
{
public:
    template<int X, int Y, int Z>
    Coord<2> operator()(Streak<2> streak, FixedCoord<X, Y, Z>)
    {
        return Coord<2>(CalcXCoord<X>()(streak), 
                        streak.origin.y() + Y);
    }
};

template<>
class DetermineLinePointerCoord<3>
{
public:
    template<int X, int Y, int Z>
    Coord<3> operator()(Streak<3> streak, FixedCoord<X, Y, Z>)
    {
        return Coord<2>(CalcXCoord<X>()(streak), 
                        streak.origin.y() + Y,
                        streak.origin.z() + Z);
    }
};

template<class STENCIL, int INDEX>
class CopyCellPointer
{
public:
    typedef typename STENCIL::template Coords<INDEX> RelCoord;

    template<typename CELL_TYPE, int DIM, typename GRID_TYPE>
    void operator()(CELL_TYPE **pointers, const Streak<DIM>& streak, GRID_TYPE *grid)
    {
        Coord<DIM> c(DetermineLinePointerCoord<DIM>()(
                         streak, typename STENCIL::template Coords<INDEX>()));
        pointers[INDEX] = &(*grid)[c];
    }
};

}

/**
 * will initialize an array of pointers so that it can be used with
 * LinePointerNeighborhood.
 */
template<class STENCIL>
class LinePointerAssembly
{};

template<>
class LinePointerAssembly<Stencils::Moore<2, 1> >
{
public:
    template<typename CELL, typename GRID>
    void operator()(CELL *pointers[9], const Streak<2>& streak, GRID& grid)
    {
        Stencils::Repeat<Stencils::Moore<2, 1>::VOLUME, 
                         LinePointerAssemblyHelpers::CopyCellPointer, 
                         Stencils::Moore<2, 1> >()(pointers, streak, &grid);
    }
};

template<>
class LinePointerAssembly<Stencils::Moore<3, 1> >
{
public:
    template<typename CELL, typename GRID>
    void operator()(CELL *pointers[27], const Streak<3>& streak, GRID& grid)
    {
        pointers[ 0] = &grid[Coord<3>(streak.origin.x() - 1, streak.origin.y() - 1, streak.origin.z() - 1)];
        pointers[ 1] = &grid[Coord<3>(streak.origin.x() + 0, streak.origin.y() - 1, streak.origin.z() - 1)];
        pointers[ 2] = &grid[Coord<3>(streak.endX,           streak.origin.y() - 1, streak.origin.z() - 1)];
        pointers[ 3] = &grid[Coord<3>(streak.origin.x() - 1, streak.origin.y() + 0, streak.origin.z() - 1)];
        pointers[ 4] = &grid[Coord<3>(streak.origin.x() + 0, streak.origin.y() + 0, streak.origin.z() - 1)];
        pointers[ 5] = &grid[Coord<3>(streak.endX,           streak.origin.y() + 0, streak.origin.z() - 1)];
        pointers[ 6] = &grid[Coord<3>(streak.origin.x() - 1, streak.origin.y() + 1, streak.origin.z() - 1)];
        pointers[ 7] = &grid[Coord<3>(streak.origin.x() + 0, streak.origin.y() + 1, streak.origin.z() - 1)];
        pointers[ 8] = &grid[Coord<3>(streak.endX,           streak.origin.y() + 1, streak.origin.z() - 1)];

        pointers[ 9] = &grid[Coord<3>(streak.origin.x() - 1, streak.origin.y() - 1, streak.origin.z() + 0)];
        pointers[10] = &grid[Coord<3>(streak.origin.x() + 0, streak.origin.y() - 1, streak.origin.z() + 0)];
        pointers[11] = &grid[Coord<3>(streak.endX,           streak.origin.y() - 1, streak.origin.z() + 0)];
        pointers[12] = &grid[Coord<3>(streak.origin.x() - 1, streak.origin.y() + 0, streak.origin.z() + 0)];
        pointers[13] = &grid[Coord<3>(streak.origin.x() + 0, streak.origin.y() + 0, streak.origin.z() + 0)];
        pointers[14] = &grid[Coord<3>(streak.endX,           streak.origin.y() + 0, streak.origin.z() + 0)];
        pointers[15] = &grid[Coord<3>(streak.origin.x() - 1, streak.origin.y() + 1, streak.origin.z() + 0)];
        pointers[16] = &grid[Coord<3>(streak.origin.x() + 0, streak.origin.y() + 1, streak.origin.z() + 0)];
        pointers[17] = &grid[Coord<3>(streak.endX,           streak.origin.y() + 1, streak.origin.z() + 0)];

        pointers[18] = &grid[Coord<3>(streak.origin.x() - 1, streak.origin.y() - 1, streak.origin.z() + 1)];
        pointers[19] = &grid[Coord<3>(streak.origin.x() + 0, streak.origin.y() - 1, streak.origin.z() + 1)];
        pointers[20] = &grid[Coord<3>(streak.endX,           streak.origin.y() - 1, streak.origin.z() + 1)];
        pointers[21] = &grid[Coord<3>(streak.origin.x() - 1, streak.origin.y() + 0, streak.origin.z() + 1)];
        pointers[22] = &grid[Coord<3>(streak.origin.x() + 0, streak.origin.y() + 0, streak.origin.z() + 1)];
        pointers[23] = &grid[Coord<3>(streak.endX,           streak.origin.y() + 0, streak.origin.z() + 1)];
        pointers[24] = &grid[Coord<3>(streak.origin.x() - 1, streak.origin.y() + 1, streak.origin.z() + 1)];
        pointers[25] = &grid[Coord<3>(streak.origin.x() + 0, streak.origin.y() + 1, streak.origin.z() + 1)];
        pointers[26] = &grid[Coord<3>(streak.endX,           streak.origin.y() + 1, streak.origin.z() + 1)];
    }
};

template<>
class LinePointerAssembly<Stencils::VonNeumann<2, 1> >
{
public:
    template<typename CELL, typename GRID>
    void operator()(CELL *pointers[5], const Streak<2>& streak, GRID& grid)
    {
        pointers[0] = &grid[Coord<2>(streak.origin.x() + 0, streak.origin.y() - 1)];
        pointers[1] = &grid[Coord<2>(streak.origin.x() - 1, streak.origin.y() + 0)];
        pointers[2] = &grid[Coord<2>(streak.origin.x() + 0, streak.origin.y() + 0)];
        pointers[3] = &grid[Coord<2>(streak.endX,           streak.origin.y() + 0)];
        pointers[4] = &grid[Coord<2>(streak.origin.x() + 0, streak.origin.y() + 1)];
    }
};

template<>
class LinePointerAssembly<Stencils::VonNeumann<3, 1> >
{
public:
    template<typename CELL, typename GRID>
    void operator()(CELL *pointers[7], const Streak<3>& streak, GRID& grid)
    {
        pointers[ 0] = &grid[Coord<3>(streak.origin.x() + 0, streak.origin.y() + 0, streak.origin.z() - 1)];
        pointers[ 1] = &grid[Coord<3>(streak.origin.x() + 0, streak.origin.y() - 1, streak.origin.z() + 0)];
        pointers[ 2] = &grid[Coord<3>(streak.origin.x() - 1, streak.origin.y() + 0, streak.origin.z() + 0)];
        pointers[ 3] = &grid[Coord<3>(streak.origin.x() + 0, streak.origin.y() + 0, streak.origin.z() + 0)];
        pointers[ 4] = &grid[Coord<3>(streak.endX,           streak.origin.y() + 0, streak.origin.z() + 0)];
        pointers[ 5] = &grid[Coord<3>(streak.origin.x() + 0, streak.origin.y() + 1, streak.origin.z() + 0)];
        pointers[ 6] = &grid[Coord<3>(streak.origin.x() + 0, streak.origin.y() + 0, streak.origin.z() + 1)];
    }
};


}

#endif
