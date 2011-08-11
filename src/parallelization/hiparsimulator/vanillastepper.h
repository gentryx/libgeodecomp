#include <libgeodecomp/config.h>
#ifdef LIBGEODECOMP_FEATURE_MPI
#ifndef _libgeodecomp_parallelization_hiparsimulator_vanillastepper_h_
#define _libgeodecomp_parallelization_hiparsimulator_vanillastepper_h_

#include <libgeodecomp/misc/displacedgrid.h>
#include <libgeodecomp/parallelization/hiparsimulator/partitionmanager.h>
#include <libgeodecomp/parallelization/hiparsimulator/patchbuffer.h>
#include <libgeodecomp/parallelization/hiparsimulator/stepperhelper.h>

namespace LibGeoDecomp {
namespace HiParSimulator {

template<int INDEX, int DIM, typename TOPOLOGY>
class OffsetHelper
{
public:
    void operator()(
        Coord<DIM> *offset,
        Coord<DIM> *dimensions,
        const CoordBox<DIM>& ownBoundingBox, 
        const CoordBox<DIM>& simulationArea,
        const int& ghostZoneWidth)
    {
        offset->c[INDEX] = 0;
        if (TOPOLOGY::WrapEdges) {
            int enlargedWidth = 
                ownBoundingBox.dimensions.c[INDEX] + 2 * ghostZoneWidth;
            if (enlargedWidth < simulationArea.dimensions.c[INDEX]) {
                offset->c[INDEX] = 
                    ownBoundingBox.origin.c[INDEX] - ghostZoneWidth;
            } else {
                offset->c[INDEX] = 0;
            }
            dimensions->c[INDEX] = 
                std::min(enlargedWidth, simulationArea.dimensions.c[INDEX]);
        } else {
            offset->c[INDEX] = 
                std::max(0, ownBoundingBox.origin.c[INDEX] - ghostZoneWidth);
            int end = std::min(simulationArea.origin.c[INDEX] + 
                               simulationArea.dimensions.c[INDEX],
                               ownBoundingBox.origin.c[INDEX] + 
                               ownBoundingBox.dimensions.c[INDEX] + 
                               ghostZoneWidth);
            dimensions->c[INDEX] = end - offset->c[INDEX];
        } 

        OffsetHelper<INDEX - 1, DIM, typename TOPOLOGY::ParentTopology>()(
            offset, 
            dimensions, 
            ownBoundingBox, 
            simulationArea, 
            ghostZoneWidth);
    }
};

template<int DIM, typename TOPOLOGY>
class OffsetHelper<-1, DIM, TOPOLOGY>
{
public:
    void operator()(
        Coord<DIM> *offset,
        Coord<DIM> *dimensions,
        const CoordBox<DIM>& ownBoundingBox, 
        const CoordBox<DIM>& simulationArea,
        const int& ghostZoneWidth)
    {}
};

// fixme: deduce DIM from CELL_TYPE?!
template<typename CELL_TYPE, int DIM>
class VanillaStepper : 
        public StepperHelper<CELL_TYPE, DIM, DisplacedGrid<CELL_TYPE, typename CELL_TYPE::Topology> >
{
public:
    friend class VanillaStepperRegionTest;
    friend class VanillaStepperBasicTest;
    friend class VanillaStepperTest;
    typedef DisplacedGrid<CELL_TYPE, typename CELL_TYPE::Topology> GridType;
    typedef class StepperHelper<CELL_TYPE, DIM, GridType> ParentType;
    typedef PartitionManager<DIM, typename CELL_TYPE::Topology> MyPartitionManager;

    inline VanillaStepper(
        boost::shared_ptr<MyPartitionManager> _partitionManager,
        boost::shared_ptr<Initializer<CELL_TYPE> > _initializer) :
        ParentType(_partitionManager, _initializer)
    {
        curStep = this->getInitializer().startStep();
        curNanoStep = 0;
        initGrids();
    }

    inline virtual void update(int nanoSteps) 
    {
        for (int i = 0; i < nanoSteps; ++i)
            update();
    }

    inline virtual const Grid<CELL_TYPE, typename CELL_TYPE::Topology>& grid() const
    {
        return *oldGrid->vanillaGrid();
    }

    inline virtual std::pair<int, int> currentStep() const
    {
        return std::make_pair(curStep, curNanoStep);
    }

private:
    int curStep;
    int curNanoStep;
    int validGhostZoneWidth;
    // fixme: do we need these two everywhere?
    Coord<DIM> offset;
    Coord<DIM> dimensions;
    boost::shared_ptr<GridType> oldGrid;
    boost::shared_ptr<GridType> newGrid;
    PatchBuffer<GridType, GridType, CELL_TYPE> patchBuffer;

    inline void update()
    {
        unsigned index = --validGhostZoneWidth;
        const Region<DIM>& region = this->getPartitionManager().ownRegion(index);
        for (typename Region<DIM>::Iterator i = region.begin(); i != region.end(); ++i) 
            (*newGrid)[*i].update(oldGrid->getNeighborhood(*i), curNanoStep);
        std::swap(oldGrid, newGrid);

        ++curNanoStep;
        if (curNanoStep == CELL_TYPE::nanoSteps()) {
            curNanoStep = 0;
            curStep++;
        }

        notifyPatchAccepters(region);



        // if (validGhostZoneWidth == 0) {
        //     notifyPatchProviders();
        //     resetValidGhostZoneWidth();
        // }
    }

    inline void notifyPatchAccepters(const Region<DIM>& region)
    {
        for (class ParentType::PatchAccepterList::iterator i = 
                 this->patchAccepters.begin();
             i != this->patchAccepters.end();
             ++i)
            if (globalNanoStep() == (*i)->nextRequiredNanoStep()) 
                (*i)->put(*oldGrid, region, globalNanoStep());
    }

    inline void notifyPatchProviders()
    {
        for (typename ParentType::PatchProviderList::iterator i = 
                 this->patchProviders.begin();
             i != this->patchProviders.end();
             ++i)
            (*i)->get(
                *oldGrid,
                this->getPartitionManager().getOuterRim(),
                globalNanoStep());
    }

    inline long globalNanoStep() const
    {
        return curStep * CELL_TYPE::nanoSteps() + curNanoStep;
    }

    inline void initGrids()
    {
        guessOffset();
        CoordBox<DIM> gridBox = 
            this->getPartitionManager().ownExpandedRegion().boundingBox();
        
        // std::cout << "my gridBox: " << gridBox << "\n";
        oldGrid.reset(new GridType(gridBox));
        newGrid.reset(new GridType(gridBox));
        this->getInitializer().grid(&*oldGrid);
        newGrid->getEdgeCell() = oldGrid->getEdgeCell();
        resetValidGhostZoneWidth();

        const Region<DIM> rim = this->getPartitionManager().rim(
            this->getPartitionManager().getGhostZoneWidth());
        patchBuffer.pushRequest(&rim, 0);
        patchBuffer.put(*oldGrid, 
                        this->getPartitionManager().ownExpandedRegion(), globalNanoStep());
    }

    inline void resetValidGhostZoneWidth()
    {
        validGhostZoneWidth = this->getPartitionManager().getGhostZoneWidth();
    }

    /**
     * calculates a (mostly) suitable offset which (in conjuction with
     * a DisplacedGrid) avoids having grids with a size equal to the
     * whole simulation area on torus topologies.
     */
    inline void guessOffset()
    {
        const CoordBox<DIM>& boundingBox = 
            this->getPartitionManager().ownRegion().boundingBox();
        OffsetHelper<DIM - 1, DIM, typename CELL_TYPE::Topology>()(
            &offset,
            &dimensions,
            boundingBox,
            this->getInitializer().gridBox(),
            this->getPartitionManager().getGhostZoneWidth());
    }
};

}
}

#endif
#endif
