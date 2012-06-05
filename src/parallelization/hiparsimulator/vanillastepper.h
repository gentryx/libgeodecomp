#ifndef _libgeodecomp_parallelization_hiparsimulator_vanillastepper_h_
#define _libgeodecomp_parallelization_hiparsimulator_vanillastepper_h_

#include <libgeodecomp/misc/displacedgrid.h>
#include <libgeodecomp/parallelization/hiparsimulator/patchbufferfixed.h>
#include <libgeodecomp/parallelization/hiparsimulator/stepper.h>

namespace LibGeoDecomp {
namespace HiParSimulator {

template<typename CELL_TYPE>
class VanillaStepper : public Stepper<CELL_TYPE>
{
    friend class VanillaStepperRegionTest;
    friend class VanillaStepperBasicTest;
    friend class VanillaStepperTest;
public:
    const static int DIM = CELL_TYPE::Topology::DIMENSIONS;

    typedef class Stepper<CELL_TYPE> ParentType;
    typedef typename ParentType::GridType GridType;
    typedef PartitionManager<
        DIM, typename CELL_TYPE::Topology> MyPartitionManager;
    typedef PatchBufferFixed<GridType, GridType, 1> MyPatchBuffer1;
    typedef PatchBufferFixed<GridType, GridType, 2> MyPatchBuffer2;
    typedef typename ParentType::PatchAccepterVec PatchAccepterVec;

    using Stepper<CELL_TYPE>::addPatchAccepter;
    using Stepper<CELL_TYPE>::initializer;
    using Stepper<CELL_TYPE>::guessOffset;
    using Stepper<CELL_TYPE>::patchAccepters;
    using Stepper<CELL_TYPE>::patchProviders;
    using Stepper<CELL_TYPE>::partitionManager;

    inline VanillaStepper(
        boost::shared_ptr<MyPartitionManager> _partitionManager,
        Initializer<CELL_TYPE> *_initializer,
        const PatchAccepterVec ghostZonePatchAccepters = PatchAccepterVec(),
        const PatchAccepterVec innerSetPatchAccepters = PatchAccepterVec()) :
        ParentType(_partitionManager, _initializer)
    {
        curStep = initializer->startStep();
        curNanoStep = 0;

        for (int i = 0; i < ghostZonePatchAccepters.size(); ++i) {
            addPatchAccepter(ghostZonePatchAccepters[i], ParentType::GHOST);
        }
        for (int i = 0; i < innerSetPatchAccepters.size(); ++i) {
            addPatchAccepter(innerSetPatchAccepters[i], ParentType::INNER_SET);
        }

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
    MyPatchBuffer2 rimBuffer;
    MyPatchBuffer1 kernelBuffer;
    Region<DIM> kernelFraction;

    inline void update()
    {
        unsigned index = ghostZoneWidth() - --validGhostZoneWidth;
        const Region<DIM>& region = partitionManager->innerSet(index);
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

        notifyPatchAccepters(region, ParentType::INNER_SET, globalNanoStep());
        notifyPatchProviders(region, ParentType::INNER_SET, globalNanoStep());
        
        if (validGhostZoneWidth == 0) {
            notifyPatchProviders(
                partitionManager->rim(0), ParentType::GHOST, globalNanoStep());
            updateGhost();
            resetValidGhostZoneWidth();
        }
    }

    inline void notifyPatchAccepters(
        const Region<DIM>& region, 
        const typename ParentType::PatchType& patchType,
        const long& nanoStep)
    {
        for (typename ParentType::PatchAccepterList::iterator i = 
	       patchAccepters[patchType].begin();
             i != patchAccepters[patchType].end();
             ++i) {
            if (nanoStep == (*i)->nextRequiredNanoStep()) {
                (*i)->put(*oldGrid, region, nanoStep);
            }
        }
    }

    inline void notifyPatchProviders(
        const Region<DIM>& region, 
        const typename ParentType::PatchType& patchType,
        const long& nanoStep)
    {
        for (typename ParentType::PatchProviderList::iterator i = 
                 patchProviders[patchType].begin();
             i != patchProviders[patchType].end();
             ++i) {
            (*i)->get(
                &*oldGrid,
                region,
                nanoStep);
        }
    }

    inline long globalNanoStep() const
    {
        return curStep * CELL_TYPE::nanoSteps() + curNanoStep;
    }

    inline void initGrids()
    {
        Coord<DIM> topoDim = initializer->gridDimensions();
        CoordBox<DIM> gridBox;
        guessOffset(&gridBox.origin, &gridBox.dimensions);

        oldGrid.reset(new GridType(gridBox, CELL_TYPE(), topoDim));
        newGrid.reset(new GridType(gridBox, CELL_TYPE(), topoDim));
        initializer->grid(&*oldGrid);
        newGrid->getEdgeCell() = oldGrid->getEdgeCell();
        resetValidGhostZoneWidth();

        notifyPatchAccepters(
            rim(),     
            ParentType::GHOST,     
            globalNanoStep());
        notifyPatchAccepters(
            partitionManager->innerSet(0), 
            ParentType::INNER_SET, 
            globalNanoStep());

        kernelBuffer = MyPatchBuffer1(partitionManager->getVolatileKernel());
        rimBuffer = MyPatchBuffer2(rim());
        saveRim(globalNanoStep());
        updateGhost();
    }
    
    /**
     * computes the next ghost zone at time "t_1 = globalNanoStep() +
     * ghostZoneWidth()". Expects that oldGrid has its kernel and its
     * outer ghostzone updated to time "globalNanoStep()" and that the
     * inner ghostzones (rim) at time t_1 can be retrieved from the
     * internal patch buffer. Will leave oldgrid in a state so that
     * its whole ownRegion() will be at time t_1 and the rim will be
     * saved to the patchBuffer at "t2 = t1 + ghostZoneWidth()".
     */
    inline void updateGhost() 
    {
        // fixme: skip all this ghost zone buffering for
        // ghostZoneWidth == 1?             

        // 1: Prepare grid. The following update of the ghostzone will
        // destroy parts of the kernel, which is why we'll
        // save/restore those.
        saveKernel();
        // We need to restore the rim since it got destroyed while the
        // kernel was updated.
        restoreRim(false);

        // 2: actual ghostzone update
        int oldNanoStep = curNanoStep;
        int oldStep = curStep;
        int curGlobalNanoStep = globalNanoStep();

        for (int t = 0; t < ghostZoneWidth(); ++t) {
            const Region<DIM>& region = partitionManager->rim(t + 1);
            for (typename Region<DIM>::Iterator i = region.begin(); 
                 i != region.end(); 
                 ++i) {
                (*newGrid)[*i].update(oldGrid->getNeighborhood(*i), 
                                      curNanoStep);
            }
            ++curNanoStep;
            if (curNanoStep == CELL_TYPE::nanoSteps()) {
                curNanoStep = 0;
                curStep++;
            }

            std::swap(oldGrid, newGrid);

            ++curGlobalNanoStep;
            notifyPatchAccepters(rim(), ParentType::GHOST, curGlobalNanoStep);
        }
        curNanoStep = oldNanoStep;
        curStep = oldStep;

        saveRim(curGlobalNanoStep);
        if (ghostZoneWidth() % 2)
            std::swap(oldGrid, newGrid);

        // 3: restore grid for kernel update
        restoreRim(true);
        restoreKernel();
    }

    inline const unsigned& ghostZoneWidth() const
    {
        return partitionManager->getGhostZoneWidth();
    }
    
    inline const Region<DIM>& rim() const
    {
        return partitionManager->rim(ghostZoneWidth());
    }

    inline void resetValidGhostZoneWidth()
    {
        validGhostZoneWidth = ghostZoneWidth();
    }

    inline void saveRim(const long& nanoStep)
    {
        rimBuffer.pushRequest(nanoStep);
        rimBuffer.put(*oldGrid, rim(), nanoStep);
    }

    inline void restoreRim(const bool& remove)
    {
        rimBuffer.get(&*oldGrid, rim(), globalNanoStep(), remove);
    }

    inline void saveKernel()
    {
        kernelBuffer.pushRequest(globalNanoStep());
        kernelBuffer.put(*oldGrid, 
                         partitionManager->innerSet(ghostZoneWidth()),
                         globalNanoStep());
    }

    inline void restoreKernel()
    {
        kernelBuffer.get(
            &*oldGrid, 
            partitionManager->getVolatileKernel(), 
            globalNanoStep(), 
            true);
    }
};

}
}

#endif
