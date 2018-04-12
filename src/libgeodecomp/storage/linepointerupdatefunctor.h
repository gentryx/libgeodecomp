#ifndef LIBGEODECOMP_STORAGE_LINEPOINTERUPDATEFUNCTOR_H
#define LIBGEODECOMP_STORAGE_LINEPOINTERUPDATEFUNCTOR_H

#include <libgeodecomp/misc/apitraits.h>
#include <libgeodecomp/geometry/coord.h>
#include <libgeodecomp/geometry/coordbox.h>
#include <libgeodecomp/geometry/stencils.h>
#include <libgeodecomp/geometry/streak.h>
#include <libgeodecomp/storage/linepointerneighborhood.h>

namespace LibGeoDecomp {

#ifdef _MSC_BUILD
#pragma warning( push )
#pragma warning( disable : 4710 )
#endif

/**
 * is a functor which (tada) updates a single row (or the fraction
 * described by the Streak) in the grid. It will manufacture suitable
 * LinePointerNeighborhood objects, depending on the line's location
 * with respect to the grid dimensions and topology. It requires the
 * line pointers to be arranged as expected by the
 * PointerNeighborhood, but with a twist. Here is an 2D example with
 * the 9-point Moore stencil:
 *
 *  12                              3
 *  4X----------------------------->6
 *  78                              9
 *
 * Legend:
 *
 * - Streak: X---------->
 * - Pointers:
 *   TW: 1
 *   T:  2
 *   TE: 3
 *   W:  4
 *   C:  5
 *   ...
 *
 * This layout allows us to capture the western and eastern boundary
 * conditions for the first and last cell in the row via dedicated
 * pointers (1, 4, 7 and 3, 6, 9). For the cells in between only the
 * middle pointers (2, X, 8) are used, but with an offset. Exception:
 * when 2 or 8 point to a cell outside of the grid (i.e.
 * Grid::edgeCell) then no offset is used. This is only possible with
 * constant boundary conditions as periodic boundary conditions would
 * automatically wrap accesses to cells within the grid.
 */
template<
    typename CELL,
    int DIM = APITraits::SelectTopology<CELL>::Value::DIM,
    bool HIGH = true,
    int CUR_DIM = (DIM - 1),
    bool BOUNDARY_TOP = false,
    bool BOUNDARY_BOTTOM = false,
    bool BOUNDARY_SOUTH = false,
    bool BOUNDARY_NORTH = false
>
class LinePointerUpdateFunctor
{
public:
    void operator()(
        const Streak<DIM>& streak,
        const CoordBox<DIM>& box,
        const CELL **pointers,
        CELL *newLine,
        unsigned nanoStep)
    {
        typedef typename APITraits::SelectTopology<CELL>::Value Topology;
        const Coord<DIM>& c = streak.origin;

        // Constant conditional expressions are fine here, the
        // compiler will be smart enough to optimize this away.
        // They're there so we don't have to write redundant code for
        // every combination of flags:
#ifdef _MSC_BUILD
#pragma warning( push )
#pragma warning( disable : 4127 )
#endif
        if ((CUR_DIM == 2) && (HIGH == true)) {
            if ((!Topology::template WrapsAxis<CUR_DIM>::VALUE) &&
                (c[CUR_DIM] == (box.origin[CUR_DIM] + box.dimensions[CUR_DIM] - 1))) {
                LinePointerUpdateFunctor<CELL, DIM, false, CUR_DIM,     BOUNDARY_TOP, BOUNDARY_BOTTOM, BOUNDARY_SOUTH, true          >()(streak, box, pointers, newLine, nanoStep);
            } else {
                LinePointerUpdateFunctor<CELL, DIM, false, CUR_DIM,     BOUNDARY_TOP, BOUNDARY_BOTTOM, BOUNDARY_SOUTH, false         >()(streak, box, pointers, newLine, nanoStep);
            }
        }

        if ((CUR_DIM == 2) && (HIGH == false)) {
            if ((!Topology::template WrapsAxis<CUR_DIM>::VALUE) &&
                (c[CUR_DIM] == box.origin[CUR_DIM])) {
                LinePointerUpdateFunctor<CELL, DIM, true,  CUR_DIM - 1, BOUNDARY_TOP, BOUNDARY_BOTTOM, true,           BOUNDARY_NORTH>()(streak, box, pointers, newLine, nanoStep);
            } else {
                LinePointerUpdateFunctor<CELL, DIM, true,  CUR_DIM - 1, BOUNDARY_TOP, BOUNDARY_BOTTOM, false,          BOUNDARY_NORTH>()(streak, box, pointers, newLine, nanoStep);
            }
        }

        if ((CUR_DIM == 1) && (HIGH == true)) {
            if ((!Topology::template WrapsAxis<CUR_DIM>::VALUE) &&
                (c[CUR_DIM] == (box.origin[CUR_DIM] + box.dimensions[CUR_DIM] - 1))) {
                LinePointerUpdateFunctor<CELL, DIM, false, CUR_DIM,     BOUNDARY_TOP, true,            BOUNDARY_SOUTH, BOUNDARY_NORTH>()(streak, box, pointers, newLine, nanoStep);
            } else {
                LinePointerUpdateFunctor<CELL, DIM, false, CUR_DIM,     BOUNDARY_TOP, false,           BOUNDARY_SOUTH, BOUNDARY_NORTH>()(streak, box, pointers, newLine, nanoStep);
            }
        }

        if ((CUR_DIM == 1) && (HIGH == false)) {
            if ((!Topology::template WrapsAxis<CUR_DIM>::VALUE) &&
                (c[CUR_DIM] == box.origin[CUR_DIM])) {
                LinePointerUpdateFunctor<CELL, DIM, true,  CUR_DIM - 1, true,         BOUNDARY_BOTTOM, BOUNDARY_SOUTH, BOUNDARY_NORTH>()(streak, box, pointers, newLine, nanoStep);
            } else {
                LinePointerUpdateFunctor<CELL, DIM, true,  CUR_DIM - 1, false,        BOUNDARY_BOTTOM, BOUNDARY_SOUTH, BOUNDARY_NORTH>()(streak, box, pointers, newLine, nanoStep);
            }
        }
#ifdef _MSC_BUILD
#pragma warning( pop )
#endif

    }
};

/**
 * see class above for documentation
 */
template<typename CELL, int DIM, bool BOUNDARY_TOP, bool BOUNDARY_BOTTOM, bool BOUNDARY_SOUTH, bool BOUNDARY_NORTH>
class LinePointerUpdateFunctor<CELL, DIM, true, 0, BOUNDARY_TOP, BOUNDARY_BOTTOM, BOUNDARY_SOUTH, BOUNDARY_NORTH>
{
public:
    void operator()(
        const Streak<DIM>& streak,
        const CoordBox<DIM>& /* box */,
        const CELL **pointers,
        CELL *newLine,
        unsigned nanoStep)
    {
        typedef typename APITraits::SelectStencil<CELL>::Value Stencil;
        typedef typename APITraits::SelectUpdateLineX<CELL>::Value UpdateLineXFlag;

        long x = 0;
        long endX = streak.endX - streak.origin.x();

        if (streak.endX == (streak.origin.x() + 1)) {
            LinePointerNeighborhood<
                CELL, Stencil, true, true,
                BOUNDARY_TOP, BOUNDARY_BOTTOM, BOUNDARY_SOUTH, BOUNDARY_NORTH> hood(pointers, &x);
            updateWrapper(newLine, &x, long(endX), hood, nanoStep, UpdateLineXFlag());
            return;
        }

        LinePointerNeighborhood<
            CELL, Stencil, true, false,
            BOUNDARY_TOP, BOUNDARY_BOTTOM, BOUNDARY_SOUTH, BOUNDARY_NORTH> hoodWest(pointers, &x);
        updateWrapper(newLine, &x, 1, hoodWest, nanoStep, UpdateLineXFlag());

        LinePointerNeighborhood<
            CELL, Stencil, false, false,
            BOUNDARY_TOP, BOUNDARY_BOTTOM, BOUNDARY_SOUTH, BOUNDARY_NORTH> hood(pointers, &x);
        updateWrapper(newLine, &x, long(endX - 1), hood, nanoStep, UpdateLineXFlag());

        LinePointerNeighborhood<
            CELL, Stencil, false, true,
            BOUNDARY_TOP, BOUNDARY_BOTTOM, BOUNDARY_SOUTH, BOUNDARY_NORTH> hoodEast(pointers, &x);
        updateWrapper(newLine, &x, endX, hoodEast, nanoStep, UpdateLineXFlag());
    }

private:
    /**
     * serves as a fork to select update() and updateLineX(),
     * depending on the Cell's API. This specialization will delegate
     * to update() calls for single cells.
     */
    template<typename NEIGHBORHOOD>
    void updateWrapper(CELL *newLine, long *x, long endX, NEIGHBORHOOD hood, unsigned nanoStep, APITraits::FalseType)
    {
        for (; *x < endX; ++(*x)) {
            newLine[*x].update(hood, nanoStep);
        }
    }

    /**
     * serves as a fork to select update() and updateLineX(),
     * depending on the Cell's API. This specialization will delegate
     * to updateLineX() for streaks of cells.
     */
    template<typename NEIGHBORHOOD>
    void updateWrapper(CELL *newLine, long *x, long endX, NEIGHBORHOOD hood, unsigned nanoStep, APITraits::TrueType)
    {
        CELL::updateLineX(newLine, x, endX, hood, nanoStep);
    }
};

#ifdef _MSC_BUILD
#pragma warning( pop )
#endif

}

#endif

