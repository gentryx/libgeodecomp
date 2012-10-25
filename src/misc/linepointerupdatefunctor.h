#ifndef _libgeodecomp_misc_linepointerupdatefunctor_h_
#define _libgeodecomp_misc_linepointerupdatefunctor_h_

#include <libgeodecomp/misc/apis.h>
#include <libgeodecomp/misc/coord.h>
#include <libgeodecomp/misc/coordbox.h>
#include <libgeodecomp/misc/linepointerneighborhood.h>
#include <libgeodecomp/misc/stencils.h>
#include <libgeodecomp/misc/streak.h>

namespace LibGeoDecomp {

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
template<typename CELL, int DIM=CELL::Topology::DIMENSIONS, bool HIGH=true, int CUR_DIM=(DIM - 1), bool BOUNDARY_TOP=false, bool BOUNDARY_BOTTOM=false, bool BOUNDARY_SOUTH=false, bool BOUNDARY_NORTH=false> 
class LinePointerUpdateFunctor 
{
public:
    void operator()(
        const Streak<DIM>& streak,
        const CoordBox<DIM>& box,
        const CELL **pointers,
        CELL *newLine,
        int nanoStep)
    {
        typedef typename CELL::Topology Topology;
        const Coord<DIM>& c = streak.origin;

        if ((CUR_DIM == 2) && (HIGH == true)) {
            if ((!WrapsAxis<CUR_DIM, Topology>::VALUE) && 
                (c[CUR_DIM] == (box.origin[CUR_DIM] + box.dimensions[CUR_DIM] - 1))) {
                LinePointerUpdateFunctor<CELL, DIM, false, CUR_DIM,     BOUNDARY_TOP, BOUNDARY_BOTTOM, BOUNDARY_SOUTH, true          >()(streak, box, pointers, newLine, nanoStep);
            } else {
                LinePointerUpdateFunctor<CELL, DIM, false, CUR_DIM,     BOUNDARY_TOP, BOUNDARY_BOTTOM, BOUNDARY_SOUTH, false         >()(streak, box, pointers, newLine, nanoStep);
            }
        }
            
        if ((CUR_DIM == 2) && (HIGH == false)) {
            if ((!WrapsAxis<CUR_DIM, Topology>::VALUE) && 
                (c[CUR_DIM] == box.origin[CUR_DIM])) {
                LinePointerUpdateFunctor<CELL, DIM, true,  CUR_DIM - 1, BOUNDARY_TOP, BOUNDARY_BOTTOM, true,           BOUNDARY_NORTH>()(streak, box, pointers, newLine, nanoStep);
            } else {
                LinePointerUpdateFunctor<CELL, DIM, true,  CUR_DIM - 1, BOUNDARY_TOP, BOUNDARY_BOTTOM, false,          BOUNDARY_NORTH>()(streak, box, pointers, newLine, nanoStep);
            }
        }
            
        if ((CUR_DIM == 1) && (HIGH == true)) {
            if ((!WrapsAxis<CUR_DIM, Topology>::VALUE) && 
                (c[CUR_DIM] == (box.origin[CUR_DIM] + box.dimensions[CUR_DIM] - 1))) {
                LinePointerUpdateFunctor<CELL, DIM, false, CUR_DIM,     BOUNDARY_TOP, true,            BOUNDARY_SOUTH, BOUNDARY_NORTH>()(streak, box, pointers, newLine, nanoStep);
            } else {
                LinePointerUpdateFunctor<CELL, DIM, false, CUR_DIM,     BOUNDARY_TOP, false,           BOUNDARY_SOUTH, BOUNDARY_NORTH>()(streak, box, pointers, newLine, nanoStep);
            }
        }
            
        if ((CUR_DIM == 1) && (HIGH == false)) {
            if ((!WrapsAxis<CUR_DIM, Topology>::VALUE) && 
                (c[CUR_DIM] == box.origin[CUR_DIM])) {
                LinePointerUpdateFunctor<CELL, DIM, true,  CUR_DIM - 1, true,         BOUNDARY_BOTTOM, BOUNDARY_SOUTH, BOUNDARY_NORTH>()(streak, box, pointers, newLine, nanoStep);
            } else {
                LinePointerUpdateFunctor<CELL, DIM, true,  CUR_DIM - 1, false,        BOUNDARY_BOTTOM, BOUNDARY_SOUTH, BOUNDARY_NORTH>()(streak, box, pointers, newLine, nanoStep);
            }
        }
            
    }
};

template<typename CELL, int DIM, bool BOUNDARY_TOP, bool BOUNDARY_BOTTOM, bool BOUNDARY_SOUTH, bool BOUNDARY_NORTH> 
class LinePointerUpdateFunctor<CELL, DIM, true, 0, BOUNDARY_TOP, BOUNDARY_BOTTOM, BOUNDARY_SOUTH, BOUNDARY_NORTH>
{
public:
    void operator()(
        const Streak<DIM>& streak,
        const CoordBox<DIM>& box,
        const CELL **pointers,
        CELL *newLine,
        int nanoStep)
    {
        typedef typename CELL::Stencil Stencil;
        
        long x = streak.origin.x();
            
        if (streak.endX == (streak.origin.x() + 1)) {
            LinePointerNeighborhood<CELL, Stencil, true, true, BOUNDARY_TOP, BOUNDARY_BOTTOM, BOUNDARY_SOUTH, BOUNDARY_NORTH> hood(pointers, &x);
            newLine[x].update(hood, nanoStep);
            return;
        }

        LinePointerNeighborhood<CELL, Stencil, true, false, BOUNDARY_TOP, BOUNDARY_BOTTOM, BOUNDARY_SOUTH, BOUNDARY_NORTH> hoodWest(pointers, &x);
        newLine[x].update(hoodWest, nanoStep);

        LinePointerNeighborhood<CELL, Stencil, false, false, BOUNDARY_TOP, BOUNDARY_BOTTOM, BOUNDARY_SOUTH, BOUNDARY_NORTH> hood(pointers, &x);
        updateMain(newLine, &x, (long)(streak.endX - 1), hood, nanoStep, typename CELL::API());
        
        LinePointerNeighborhood<CELL, Stencil, false, true, BOUNDARY_TOP, BOUNDARY_BOTTOM, BOUNDARY_SOUTH, BOUNDARY_NORTH> hoodEast(pointers, &x);
        newLine[x].update(hoodEast, nanoStep);
    }

private:
    template<typename NEIGHBORHOOD>
    void updateMain(CELL *newLine, long *x, long endX, NEIGHBORHOOD hood, int nanoStep, APIs::Base)
    {
        for ((*x) += 1; (*x) < endX; ++(*x)) {
            newLine[(*x)].update(hood, nanoStep);
        }
    }

    template<typename NEIGHBORHOOD>
    void updateMain(CELL *newLine, long *x, long endX, NEIGHBORHOOD hood, int nanoStep, APIs::Line)
    {
        CELL::updateLine(newLine, x, endX, hood, nanoStep);
    }
};

}

#endif

