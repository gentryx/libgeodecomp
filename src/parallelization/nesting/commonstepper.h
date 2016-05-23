#ifndef LIBGEODECOMP_PARALLELIZATION_NESTING_COMMONSTEPPER_H
#define LIBGEODECOMP_PARALLELIZATION_NESTING_COMMONSTEPPER_H

#include <libgeodecomp/parallelization/nesting/stepper.h>
#include <libgeodecomp/storage/patchbufferfixed.h>

namespace LibGeoDecomp {

/**
 * This class bundles functionality which is commonly required within
 * Stepper implementations, but not necessarily part of a Stepper's
 * public interface.
 */
template<typename CELL_TYPE>
class CommonStepper : public Stepper<CELL_TYPE>
{
public:
    typedef typename Stepper<CELL_TYPE>::Topology Topology;
    const static int DIM = Topology::DIM;
    const static unsigned NANO_STEPS = APITraits::SelectNanoSteps<CELL_TYPE>::VALUE;

    typedef class CommonStepper<CELL_TYPE> ParentType;
    typedef typename ParentType::GridType GridType;
    typedef PartitionManager<Topology> PartitionManagerType;
    typedef PatchBufferFixed<GridType, GridType, 1> PatchBufferType1;
    typedef PatchBufferFixed<GridType, GridType, 2> PatchBufferType2;
    typedef typename ParentType::PatchAccepterVec PatchAccepterVec;
    typedef typename ParentType::PatchProviderVec PatchProviderVec;

    using Stepper<CELL_TYPE>::guessOffset;
    using Stepper<CELL_TYPE>::addPatchAccepter;
    using Stepper<CELL_TYPE>::addPatchProvider;
    using Stepper<CELL_TYPE>::chronometer;
    using Stepper<CELL_TYPE>::initializer;
    using Stepper<CELL_TYPE>::partitionManager;
    using Stepper<CELL_TYPE>::patchAccepters;
    using Stepper<CELL_TYPE>::patchProviders;

    CommonStepper(
        boost::shared_ptr<PartitionManagerType> partitionManager,
        boost::shared_ptr<Initializer<CELL_TYPE> > initializer,
        const PatchAccepterVec& ghostZonePatchAccepters = PatchAccepterVec(),
        const PatchAccepterVec& innerSetPatchAccepters  = PatchAccepterVec(),
        const PatchProviderVec& ghostZonePatchProviders = PatchProviderVec(),
        const PatchProviderVec& innerSetPatchProviders  = PatchProviderVec(),
        bool enableFineGrainedParallelism = false) :
        Stepper<CELL_TYPE>(
            partitionManager,
            initializer),
        enableFineGrainedParallelism(enableFineGrainedParallelism)
    {
        curStep = initializer->startStep();
        curNanoStep = 0;

        for (std::size_t i = 0; i < ghostZonePatchAccepters.size(); ++i) {
            addPatchAccepter(ghostZonePatchAccepters[i], ParentType::GHOST);
        }
        for (std::size_t i = 0; i < innerSetPatchAccepters.size(); ++i) {
            addPatchAccepter(innerSetPatchAccepters[i], ParentType::INNER_SET);
        }

        for (std::size_t i = 0; i < ghostZonePatchProviders.size(); ++i) {
            addPatchProvider(ghostZonePatchProviders[i], ParentType::GHOST);
        }
        for (std::size_t i = 0; i < innerSetPatchProviders.size(); ++i) {
            addPatchProvider(innerSetPatchProviders[i], ParentType::INNER_SET);
        }
    }

    inline virtual void update(std::size_t nanoSteps)
    {
        for (std::size_t i = 0; i < nanoSteps; ++i)
        {
            update1();
        }
    }

    inline std::pair<std::size_t, std::size_t> currentStep() const
    {
        return std::make_pair(curStep, curNanoStep);
    }

    inline const GridType& grid() const
    {
        return *oldGrid;
    }

    /**
     * Proceed the simulation exactly one nano step
     */
    virtual void update1() = 0;

protected:
    std::size_t curStep;
    std::size_t curNanoStep;
    unsigned validGhostZoneWidth;
    boost::shared_ptr<GridType> oldGrid;
    boost::shared_ptr<GridType> newGrid;
    PatchBufferType2 rimBuffer;
    PatchBufferType1 kernelBuffer;
    Region<DIM> kernelFraction;
    bool enableFineGrainedParallelism;

    inline void notifyPatchAccepters(
        const Region<DIM>& region,
        const typename ParentType::PatchType& patchType,
        std::size_t nanoStep)
    {
        TimePatchAccepters t(&chronometer);

        for (typename ParentType::PatchAccepterList::iterator i =
                 patchAccepters[patchType].begin();
             i != patchAccepters[patchType].end();
             ++i) {
            if (nanoStep == (*i)->nextRequiredNanoStep()) {
                (*i)->put(
                    *oldGrid,
                    region,
                    partitionManager->getSimulationArea(),
                    nanoStep,
                    partitionManager->rank());
            }
        }
    }

    inline void notifyPatchProviders(
        const Region<DIM>& region,
        const typename ParentType::PatchType& patchType,
        std::size_t nanoStep)
    {
        TimePatchProviders t(&chronometer);

        for (typename ParentType::PatchProviderList::iterator i =
                 patchProviders[patchType].begin();
             i != patchProviders[patchType].end();
             ++i) {
            if (nanoStep == (*i)->nextAvailableNanoStep()) {
                (*i)->get(
                    &*oldGrid,
                    region,
                    partitionManager->getSimulationArea(),
                    nanoStep,
                    partitionManager->rank(),
                    true);
            }
        }
    }

    inline std::size_t globalNanoStep() const
    {
        return curStep * NANO_STEPS + curNanoStep;
    }

    inline CoordBox<DIM> initGridsCommon()
    {
        Coord<DIM> topoDim = initializer->gridDimensions();
        CoordBox<DIM> gridBox;
        guessOffset(&gridBox.origin, &gridBox.dimensions);

        oldGrid.reset(new GridType(gridBox, CELL_TYPE(), CELL_TYPE(), topoDim));
        newGrid.reset(new GridType(gridBox, CELL_TYPE(), CELL_TYPE(), topoDim));

        initializer->grid(&*oldGrid);
        *newGrid = *oldGrid;

        notifyPatchProviders(partitionManager->getOuterRim(), ParentType::GHOST,     globalNanoStep());
        notifyPatchProviders(partitionManager->ownRegion(),   ParentType::INNER_SET, globalNanoStep());

        newGrid->setEdge(oldGrid->getEdge());

        resetValidGhostZoneWidth();
        kernelBuffer = PatchBufferType1(getVolatileKernel());
        rimBuffer = PatchBufferType2(rim());

        return gridBox;
    }

    inline unsigned ghostZoneWidth() const
    {
        return partitionManager->getGhostZoneWidth();
    }

    inline const Region<DIM>& rim(unsigned offset) const
    {
        return partitionManager->rim(offset);
    }

    inline const Region<DIM>& rim() const
    {
        return rim(ghostZoneWidth());
    }

    inline const Region<DIM>& innerSet(unsigned offset) const
    {
        return partitionManager->innerSet(offset);
    }

    inline const Region<DIM>& getVolatileKernel() const
    {
        return partitionManager->getVolatileKernel();
    }

    inline const Region<DIM>& getInnerRim() const
    {
        return partitionManager->getInnerRim();
    }

    inline void resetValidGhostZoneWidth()
    {
        validGhostZoneWidth = ghostZoneWidth();
    }

    inline void saveRim(std::size_t nanoStep)
    {
        rimBuffer.pushRequest(nanoStep);
        rimBuffer.put(
            *oldGrid,
            rim(),
            partitionManager->getSimulationArea(),
            nanoStep,
            partitionManager->rank());
    }

    inline void restoreRim(bool remove)
    {
        rimBuffer.get(
            &*oldGrid,
            rim(),
            partitionManager->getSimulationArea(),
            globalNanoStep(),
            partitionManager->rank(),
            remove);
    }

    inline void saveKernel()
    {
        kernelBuffer.pushRequest(globalNanoStep());
        kernelBuffer.put(
            *oldGrid,
            innerSet(ghostZoneWidth()),
            partitionManager->getSimulationArea(),
            globalNanoStep(),
            partitionManager->rank());
    }

    inline void restoreKernel()
    {
        kernelBuffer.get(
            &*oldGrid,
            getVolatileKernel(),
            partitionManager->getSimulationArea(),
            globalNanoStep(),
            partitionManager->rank(),
            true);
    }
};

}

#endif
