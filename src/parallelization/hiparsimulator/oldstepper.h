#include <libgeodecomp/config.h>
#ifdef LIBGEODECOMP_FEATURE_MPI
#ifndef _libgeodecomp_parallelization_hiparsimulator_oldstepper_h_
#define _libgeodecomp_parallelization_hiparsimulator_oldstepper_h_

#include <libgeodecomp/parallelization/hiparsimulator/patchaccepter.h>
#include <libgeodecomp/misc/typetraits.h>

namespace LibGeoDecomp {
namespace HiParSimulator {

// fixme: oldstepper is currently out of order and now deprecated
template<class CELL_TYPE>
class OldStepper 
{
    friend class OldStepperTest;

public:
    template<class GRID_POINTER, class REGION_MARKER>
    void update(
        GRID_POINTER sourceGrid, 
        GRID_POINTER workGrid1, 
        GRID_POINTER workGrid2, 
        GRID_POINTER targetGrid, 
        const REGION_MARKER& marker, 
        const unsigned& startStep, 
        const unsigned& endStep, 
        const unsigned& startNanoStep) const
    {
        update(sourceGrid, workGrid1, workGrid2, targetGrid, marker, startStep, endStep, startNanoStep, VoidCallback<GRID_POINTER>());
    }

    template<class GRID_POINTER, class REGION_MARKER, class GRID_TYPE>
    void update(
        GRID_POINTER sourceGrid, 
        GRID_POINTER workGrid1, 
        GRID_POINTER workGrid2, 
        GRID_POINTER targetGrid, 
        const REGION_MARKER& marker, 
        const unsigned& startStep, 
        const unsigned& endStep, 
        const unsigned& startNanoStep,
        PatchAccepter<GRID_TYPE> *patchAccepter) const
    {
        // update(sourceGrid, workGrid1, workGrid2, targetGrid, marker, startStep, endStep, startNanoStep, PatchAccepterCallback<GRID_TYPE>(patchAccepter));
    }

private:
    template<typename GRID_TYPE>
    class VoidCallback
    {
    public:
        template<class REGION_MARKER>
        void operator()(GRID_TYPE, const REGION_MARKER&, const unsigned&, const unsigned&) const
        {}
    };

    template<typename GRID_TYPE>
    class PatchAccepterCallback
    {
    public:
        PatchAccepterCallback(PatchAccepter<GRID_TYPE> *patchAccepter_) :
            patchAccepter(patchAccepter_)
        {}

        template<class REGION_MARKER>
        void operator()(GRID_TYPE grid, const REGION_MARKER& marker, const unsigned& nanoStep, const unsigned step) const
        {
            patchAccepter->put(grid, marker.region(step), nanoStep);
        }
    private:
        PatchAccepter<GRID_TYPE> *patchAccepter;
    };

    template<class GRID_POINTER, class REGION_MARKER, class CALLBACK>
    void update(
        GRID_POINTER sourceGrid, 
        GRID_POINTER workGrid1, 
        GRID_POINTER workGrid2, 
        GRID_POINTER targetGrid, 
        const REGION_MARKER& marker, 
        const unsigned& startStep, 
        const unsigned& endStep, 
        const unsigned& startNanoStep,
        const CALLBACK& callback) const
    {
        unsigned nanoStep = startNanoStep;
        for (unsigned step = startStep; step < endStep; ++step) {
            GRID_POINTER oldGrid; 
            GRID_POINTER newGrid; 
            if (step % 2) {
                oldGrid = workGrid1;
                newGrid = workGrid2;
            } else {
                oldGrid = workGrid2;
                newGrid = workGrid1;
            }
            if (step == startStep)
                oldGrid = sourceGrid;
            if (step == (endStep - 1))
                newGrid = targetGrid;
            // callback(*oldGrid, marker, nanoStep, step);
            bump(oldGrid, newGrid, marker, step, nanoStep % CELL_TYPE::nanoSteps());
            nanoStep++;
        }
    }

    template<class GRID_POINTER, class REGION_MARKER>
    void bump(GRID_POINTER oldGrid, GRID_POINTER newGrid, const REGION_MARKER& marker, const unsigned& step, const unsigned& nanoStep)const
    {
        bump(oldGrid, newGrid, marker, step, nanoStep, ProvidesStreakIterator<REGION_MARKER>());
    }

    template<class GRID_POINTER, class REGION_MARKER>
    void bump(GRID_POINTER oldGrid, GRID_POINTER newGrid, const REGION_MARKER& marker, const unsigned& step, const unsigned& nanoStep, const boost::false_type&) const
    {
        bumpImpl(oldGrid, newGrid, marker.begin(step), marker.end(step), nanoStep);
    }

    template<class GRID_POINTER, class REGION_MARKER>
    void bump(GRID_POINTER oldGrid, GRID_POINTER newGrid, const REGION_MARKER& marker, const unsigned& step, const unsigned& nanoStep, const boost::true_type&) const
    {
        bumpImplStreaks(oldGrid, newGrid, marker.beginStreak(step), marker.endStreak(step), nanoStep, ProvidesDirectUpdate<CELL_TYPE>());
    }

    template<class GRID_POINTER, class ITERATOR1, class ITERATOR2>
    void bumpImpl(GRID_POINTER oldGrid, GRID_POINTER newGrid, const ITERATOR1& begin, const ITERATOR2& end, const unsigned& nanoStep) const
    {
        for (ITERATOR1 i = begin; i != end; ++i) {
            Coord<2> c = *i;
            (*newGrid)[c].update(oldGrid->getNeighborhood(c), nanoStep);
        }
    }

    template<class GRID_POINTER, class ITERATOR1, class ITERATOR2>
    void bumpImplStreaks(GRID_POINTER oldGrid, GRID_POINTER newGrid, const ITERATOR1& begin, const ITERATOR2& end, const unsigned& nanoStep, const boost::false_type&) const
    {
        for (ITERATOR1 i = begin; i != end; ++i) {
            int y = i->origin.y();
            int endX = i->endX;
            for (int x = i->origin.x(); x < endX; ++x) {
                Coord<2> c(x, y);
                (*newGrid)[c].update(oldGrid->getNeighborhood(c), nanoStep);
            }
        }
    }

    // fixme: needs test 
    template<class GRID_POINTER, class ITERATOR1, class ITERATOR2>
    void bumpImplStreaks(GRID_POINTER oldGrid, GRID_POINTER newGrid, const ITERATOR1& begin, const ITERATOR2& end, const unsigned& nanoStep, const boost::true_type&) const
    {
        int minX = oldGrid->getOrigin().x();
        int minY = oldGrid->getOrigin().y();
        int maxY = oldGrid->getOrigin().y() + oldGrid->height() - 1;
        int maxXRelative = oldGrid->width() - 1;

        for (ITERATOR1 i = begin; i != end; ++i) {
            int y = i->origin.y();
            bool upperRim = y == minY;
            bool lowerRim = y == maxY;
            int absoluteBeginX = i->origin.x() - minX;
            int absoluteEndX   = i->endX     - minX;

            for (int x = absoluteBeginX; x < absoluteEndX; ++x) {
                const CELL_TYPE& up    = (upperRim?          oldGrid->getEdgeCell() : (*oldGrid)[y-1][x+0]);
                const CELL_TYPE& left  = (x == 0?            oldGrid->getEdgeCell() : (*oldGrid)[y+0][x-1]);
                const CELL_TYPE& self  =                                              (*oldGrid)[y+0][x+0];
                const CELL_TYPE& right = (x == maxXRelative? oldGrid->getEdgeCell() : (*oldGrid)[y+0][x+1]);
                const CELL_TYPE& down  = (lowerRim?          oldGrid->getEdgeCell() : (*oldGrid)[y+1][x+0]);
                (*newGrid)[y][x].update(up, left, self, right, down, nanoStep);
            }
        }
    }


};

}
}

#endif
#endif
