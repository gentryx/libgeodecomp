#include <libgeodecomp/config.h>
#ifdef LIBGEODECOMP_FEATURE_MPI
#ifndef _libgeodecomp_parallelization_hiparsimulator_vanillastepper_h_
#define _libgeodecomp_parallelization_hiparsimulator_vanillastepper_h_

#include <libgeodecomp/misc/grid.h>
#include <libgeodecomp/parallelization/hiparsimulator/partitionmanager.h>
#include <libgeodecomp/parallelization/hiparsimulator/stepperhelper.h>

namespace LibGeoDecomp {
namespace HiParSimulator {

// fixme: deduce DIM from partition?!
template<typename CELL_TYPE, int DIM>
class VanillaStepper : 
        public StepperHelper<CELL_TYPE, DIM, Grid<CELL_TYPE, typename CELL_TYPE::Topology> >
{
public:
    friend class VanillaStepperRegionTest;
    friend class VanillaStepperBasicTest;
    typedef Grid<CELL_TYPE, typename CELL_TYPE::Topology> GridType;
    typedef class StepperHelper<CELL_TYPE, DIM, GridType> ParentType;

    inline VanillaStepper(
        boost::shared_ptr<PartitionManager<DIM> > _partitionManager,
        boost::shared_ptr<Initializer<CELL_TYPE> > _initializer) :
        ParentType(_partitionManager, _initializer)
    {
        initGrids();
        curStep = this->getInitializer().startStep();
        curNanoStep = 0;
    }

    inline virtual void update(int nanoSteps) 
    {
        for (int i = 0; i < nanoSteps; ++i)
            update();
    }

    inline virtual const GridType& grid() const
    {
        return *oldGrid;
    }

    inline virtual std::pair<int, int> currentStep() const
    {
        return std::make_pair(curStep, curNanoStep);
    }

private:
    int curStep;
    int curNanoStep;
    int validGhostZoneWidth;
    boost::shared_ptr<GridType> oldGrid;
    boost::shared_ptr<GridType> newGrid;

    inline void update()
    {
        unsigned index = --validGhostZoneWidth;
        const Region<DIM>& region = this->getPartitionManager().ownRegion(index);
        for (typename Region<DIM>::Iterator i = region.begin(); i != region.end(); ++i) 
            (*newGrid)[*i].update(oldGrid->getNeighborhood(*i), curNanoStep);
        std::swap(oldGrid, newGrid);
        curNanoStep++;
        if (curNanoStep == CELL_TYPE::nanoSteps()) {
            curNanoStep = 0;
            curStep++;
        }

        for (class ParentType::PatchAccepterList::iterator i = 
                 this->patchAccepters.begin();
             i != this->patchAccepters.end();
             ++i)
            if (globalNanoStep() == (*i)->nextRequiredNanoStep()) 
                (*i)->put(*oldGrid, region, globalNanoStep());

        if (validGhostZoneWidth == 0) {
            for (typename ParentType::PatchProviderList::iterator i = 
                     this->patchProviders.begin();
                 i != this->patchProviders.end();
                 ++i)
                (*i)->get(
                    *oldGrid,
                    this->getPartitionManager().getOuterRim(),
                    globalNanoStep());
            resetValidGhostZoneWidth();
        }
    }

    // fixme: use only one global nano step counter, and deduce the
    // nano step from it, rather than vice versa
    inline long globalNanoStep() const
    {
        return curStep * CELL_TYPE::nanoSteps() + curNanoStep;
    }

    inline void initGrids()
    {
        Coord<DIM> dim = this->getInitializer().gridBox().dimensions;
        oldGrid.reset(new GridType(dim));
        newGrid.reset(new GridType(dim));
        this->getInitializer().grid(&*oldGrid);
        newGrid->getEdgeCell() = oldGrid->getEdgeCell();
        resetValidGhostZoneWidth();
    }

    inline void resetValidGhostZoneWidth()
    {
        validGhostZoneWidth = this->getPartitionManager().getGhostZoneWidth();
    }
};

}
}

#endif
#endif
