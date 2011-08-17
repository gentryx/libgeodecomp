#include <libgeodecomp/config.h>
#ifdef LIBGEODECOMP_FEATURE_MPI
#ifndef _libgeodecomp_parallelization_hiparsimulator_vanillastepper_h_
#define _libgeodecomp_parallelization_hiparsimulator_vanillastepper_h_

// fixme: remove this dependency
#include <libgeodecomp/mpilayer/mpilayer.h>

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
    public StepperHelper<CELL_TYPE, DIM, 
                         DisplacedGrid<CELL_TYPE, typename CELL_TYPE::Topology, true> >
{
public:
    friend class VanillaStepperRegionTest;
    friend class VanillaStepperBasicTest;
    friend class VanillaStepperTest;
    typedef DisplacedGrid<CELL_TYPE, typename CELL_TYPE::Topology, true> GridType;
    typedef class StepperHelper<CELL_TYPE, DIM, GridType> ParentType;
    typedef PartitionManager<DIM, typename CELL_TYPE::Topology> MyPartitionManager;

    inline VanillaStepper(
        boost::shared_ptr<MyPartitionManager> _partitionManager,
        boost::shared_ptr<Initializer<CELL_TYPE> > _initializer) :
        ParentType(_partitionManager, _initializer)
    {
        curStep = initializer().startStep();
        curNanoStep = 0;
        initGrids();
    }

    inline virtual void update(int nanoSteps) 
    {
        for (int i = 0; i < nanoSteps; ++i)
            update();
    }

    inline virtual std::pair<int, int> currentStep() const
    {
        return std::make_pair(curStep, curNanoStep);
    }

    inline virtual const GridType& grid() const
    {
        return *oldGrid;
    }

private:
    int curStep;
    int curNanoStep;
    int validGhostZoneWidth;
    boost::shared_ptr<GridType> oldGrid;
    boost::shared_ptr<GridType> newGrid;
    PatchBuffer<GridType, GridType, CELL_TYPE> rimBuffer;
    Region<DIM> kernelFraction;

    inline void update()
    {
        unsigned index = ghostZoneWidth() - --validGhostZoneWidth;
        const Region<DIM>& region = partitionManager().innerSet(index);
        // fixme: honor streak updaters here, akin to StripingSimulator
        for (typename Region<DIM>::Iterator i = region.begin(); 
             i != region.end(); 
             ++i) 
            (*newGrid)[*i].update(oldGrid->getNeighborhood(*i), curNanoStep);
        std::swap(oldGrid, newGrid);

        ++curNanoStep;
        if (curNanoStep == CELL_TYPE::nanoSteps()) {
            curNanoStep = 0;
            curStep++;
        }

        // fixme: we'll have to distinguish between
        // accepters/providers for the ghost zones and the kernel
        // here...
        notifyPatchAccepters(region);

        if (validGhostZoneWidth == 0) {
            notifyPatchProviders();
            resetValidGhostZoneWidth();
        }
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
                partitionManager().getOuterRim(),
                globalNanoStep());
    }

    inline long globalNanoStep() const
    {
        return curStep * CELL_TYPE::nanoSteps() + curNanoStep;
    }

    inline void initGrids()
    {
        Coord<DIM> topoDim = initializer().gridDimensions();
        CoordBox<DIM> gridBox;
        guessOffset(&gridBox.origin, &gridBox.dimensions);

        oldGrid.reset(new GridType(gridBox, CELL_TYPE(), topoDim));
        newGrid.reset(new GridType(gridBox, CELL_TYPE(), topoDim));
        initializer().grid(&*oldGrid);
        newGrid->getEdgeCell() = oldGrid->getEdgeCell();
        resetValidGhostZoneWidth();

        // fixme
        // kernelFraction = partitionManager().

        // save inner rim
        rimBuffer.pushRequest(&rim(), globalNanoStep());
        rimBuffer.put(
            *oldGrid, partitionManager().ownRegion(), globalNanoStep());
        updateGhost();
    }
    
    /**
     * computes the next ghost zone at time "t_1 = globalNanoStep() +
     * ghostZoneWidth()". Expects that oldGrid has its kernel updated
     * to time "globalNanoStep()" and that outer and inner ghostzones
     * at time t_1 can be retrieved from various patch providers. Will
     * leave oldgrid in a state so that its whole ownRegion() will be
     * at time t_1 and the inner ghostzone (rim) will be saved at "t2
     * = t1 + ghostZoneWidth()".
     */
    inline void updateGhost() 
    {
        // 1: Prepare grid. The following update of the ghostzone will
        // destroy parts of the kernel, which is why we'll
        // save/restore those.
        PatchBuffer<GridType, GridType, CELL_TYPE> kernelBuffer;
        kernelBuffer.pushRequest(&partitionManager().getVolatileKernel(), 
                                 globalNanoStep());
        kernelBuffer.put(*oldGrid, 
                         partitionManager().innerSet(ghostZoneWidth()),
                         globalNanoStep());

        // We need to restore the rim since it got destroyed while the
        // kernel was updated.
        rimBuffer.get(*oldGrid, rim(), globalNanoStep(), false);

        // fixme: missing: external ghost zones from ghostzoneprovider 

        // 2: actual ghostzone update
        int nextNanoStep = curNanoStep;
        for (int t = 0; t < ghostZoneWidth(); ++t) {
            const Region<DIM>& region = partitionManager().rim(t + 1);
            for (typename Region<DIM>::Iterator i = region.begin(); 
                 i != region.end(); 
                 ++i) {
                (*newGrid)[*i].update(oldGrid->getNeighborhood(*i), 
                                              nextNanoStep);
            }
            ++nextNanoStep;
            std::swap(oldGrid, newGrid);
        }

        rimBuffer.pushRequest(&rim(), globalNanoStep() + ghostZoneWidth());
        rimBuffer.put(*oldGrid, rim(), globalNanoStep() + ghostZoneWidth());
        if (ghostZoneWidth() % 2)
            std::swap(oldGrid, newGrid);

        // 3: restore grid for kernel update
        rimBuffer.get(
            *oldGrid, 
            rim(), 
            globalNanoStep(), 
            true);
        kernelBuffer.get(
            *oldGrid, 
            partitionManager().getVolatileKernel(), 
            globalNanoStep(), 
            true);
    }

    inline const unsigned& ghostZoneWidth() const
    {
        return partitionManager().getGhostZoneWidth();
    }
    
    inline const Region<DIM>& rim() const
    {
        return partitionManager().rim(ghostZoneWidth());
    }

    inline void resetValidGhostZoneWidth()
    {
        validGhostZoneWidth = ghostZoneWidth();
    }

    /**
     * calculates a (mostly) suitable offset which (in conjuction with
     * a DisplacedGrid) avoids having grids with a size equal to the
     * whole simulation area on torus topologies.
     */
    inline void guessOffset(Coord<DIM> *offset, Coord<DIM> *dimensions)
    {
        const CoordBox<DIM>& boundingBox = 
            partitionManager().ownRegion().boundingBox();
        OffsetHelper<DIM - 1, DIM, typename CELL_TYPE::Topology>()(
            offset,
            dimensions,
            boundingBox,
            initializer().gridBox(),
            partitionManager().getGhostZoneWidth());
    }

    inline MyPartitionManager& partitionManager() 
    {
        return this->getPartitionManager();
    }

    inline const MyPartitionManager& partitionManager() const
    {
        return this->getPartitionManager();
    }

    inline Initializer<CELL_TYPE>& initializer() 
    {
        return this->getInitializer();
    }

    inline const Initializer<CELL_TYPE>& initializer() const
    {
        return this->getInitializer();
    }
};

}
}

#endif
#endif
