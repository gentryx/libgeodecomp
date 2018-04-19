#ifndef LIBGEODECOMP_STORAGE_LINEPOINTERASSEMBLY_H
#define LIBGEODECOMP_STORAGE_LINEPOINTERASSEMBLY_H

#include <libgeodecomp/geometry/stencils.h>
#include <libgeodecomp/geometry/streak.h>

namespace LibGeoDecomp {

namespace LinePointerAssemblyHelpers {

/**
 * Utility class to deduce the X coordinate from a streak using the
 * original stencil's shape. See LinePointerUpdateFunctor for a
 * description of the individual pointers' meaning.
 */
template<int X>
class CalcXCoord;

/**
 * see above
 */
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

/**
 * see above
 */
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

/**
 * see above
 */
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

/**
 * Another helper class for looking up the correct pointer depending
 * on a (compile-time constant) coordinate specifier. This specifier
 * is important to distinguish left and right boundary of a streak, as
 * wrapping around the x-axis may happen here.
 */
template<int DIM>
class DetermineLinePointerCoord;

/**
 * see above
 */
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

/**
 * see above
 */
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

/**
 * see above
 */
template<>
class DetermineLinePointerCoord<3>
{
public:
    template<int X, int Y, int Z>
    Coord<3> operator()(Streak<3> streak, FixedCoord<X, Y, Z>)
    {
        return Coord<3>(CalcXCoord<X>()(streak),
                        streak.origin.y() + Y,
                        streak.origin.z() + Z);
    }
};

/**
 * Sample all pointers as speficied by a stencil:
 */
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

#ifdef _MSC_BUILD
#pragma warning( push )
#pragma warning( disable : 4710 4711 )
#endif

/**
 * will initialize an array of pointers so that it can be used with
 * LinePointerNeighborhood.
 */
template<class STENCIL>
class LinePointerAssembly
{
public:
    template<typename CELL, typename GRID>
    void operator()(const CELL *pointers[STENCIL::VOLUME], const Streak<STENCIL::DIM>& streak, GRID& grid)
    {
        Stencils::Repeat<STENCIL::VOLUME,
                         LinePointerAssemblyHelpers::CopyCellPointer,
                         STENCIL>()(pointers, streak, &grid);
    }
};

#ifdef _MSC_BUILD
#pragma warning( pop )
#endif

}

#endif
