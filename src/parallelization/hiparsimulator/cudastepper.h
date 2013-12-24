#ifndef LIBGEODECOMP_PARALLELIZATION_HIPARSIMULATOR_CUDASTEPPER_H
#define LIBGEODECOMP_PARALLELIZATION_HIPARSIMULATOR_CUDASTEPPER_H

#include <libgeodecomp/parallelization/hiparsimulator/stepper.h>
#include <libgeodecomp/storage/patchbufferfixed.h>
#include <libgeodecomp/storage/updatefunctor.h>

namespace LibGeoDecomp {
namespace HiParSimulator {

/**
 * The CUDAStepper offloads cell updates to a CUDA enabled GPU.
 *
 * FIXME: add option to select CUDA device in c-tor
 */
template<typename CELL_TYPE>
class CUDAStepper : public Stepper<CELL_TYPE>
{
public:
    friend class CUDAStepperRegionTest;
    friend class CUDAStepperBasicTest;
    friend class CUDAStepperTest;

    typedef typename Stepper<CELL_TYPE>::Topology Topology;
    const static int DIM = Topology::DIM;
    const static unsigned NANO_STEPS = APITraits::SelectNanoSteps<CELL_TYPE>::VALUE;

    typedef class Stepper<CELL_TYPE> ParentType;
    typedef typename ParentType::GridType GridType;
    typedef PartitionManager<Topology> PartitionManagerType;
    typedef PatchBufferFixed<GridType, GridType, 1> PatchBufferType1;
    typedef PatchBufferFixed<GridType, GridType, 2> PatchBufferType2;
    typedef typename ParentType::PatchAccepterVec PatchAccepterVec;

    using Stepper<CELL_TYPE>::addPatchAccepter;
    using Stepper<CELL_TYPE>::initializer;
    using Stepper<CELL_TYPE>::guessOffset;
    using Stepper<CELL_TYPE>::patchAccepters;
    using Stepper<CELL_TYPE>::patchProviders;
    using Stepper<CELL_TYPE>::partitionManager;

    using Stepper<CELL_TYPE>::chronometer;

    inline CUDAStepper(
        boost::shared_ptr<PartitionManagerType> partitionManager,
        boost::shared_ptr<Initializer<CELL_TYPE> > initializer,
        const PatchAccepterVec& ghostZonePatchAccepters = PatchAccepterVec(),
        const PatchAccepterVec& innerSetPatchAccepters = PatchAccepterVec()) :
        ParentType(partitionManager, initializer)
    {
        curStep = initializer->startStep();
        curNanoStep = 0;

        for (std::size_t i = 0; i < ghostZonePatchAccepters.size(); ++i) {
            addPatchAccepter(ghostZonePatchAccepters[i], ParentType::GHOST);
        }
        for (std::size_t i = 0; i < innerSetPatchAccepters.size(); ++i) {
            addPatchAccepter(innerSetPatchAccepters[i], ParentType::INNER_SET);
        }

        initGrids();
    }

    inline void update(std::size_t nanoSteps)
    {
        for (std::size_t i = 0; i < nanoSteps; ++i)
        {
            update();
        }
    }

    inline virtual std::pair<std::size_t, std::size_t> currentStep() const
    {
        return std::make_pair(curStep, curNanoStep);
    }

    inline virtual const GridType& grid() const
    {
        return *oldGrid;
    }

private:
    std::size_t curStep;
    std::size_t curNanoStep;
    unsigned validGhostZoneWidth;
    boost::shared_ptr<GridType> oldGrid;
    boost::shared_ptr<GridType> newGrid;
    PatchBufferType2 rimBuffer;
    PatchBufferType1 kernelBuffer;
    Region<DIM> kernelFraction;

    inline void update()
    {
        std::cout << "update()\n";
        unsigned index = ghostZoneWidth() - --validGhostZoneWidth;
        const Region<DIM>& region = innerSet(index);
        {
            TimeComputeInner t(&chronometer);

            UpdateFunctor<CELL_TYPE>()(
                region,
                Coord<DIM>(),
                Coord<DIM>(),
                *oldGrid,
                &*newGrid,
                curNanoStep);
            std::swap(oldGrid, newGrid);

            ++curNanoStep;
            if (curNanoStep == NANO_STEPS) {
                curNanoStep = 0;
                curStep++;
            }
        }

        notifyPatchAccepters(region, ParentType::INNER_SET, globalNanoStep());

        if (validGhostZoneWidth == 0) {
            updateGhost();
            resetValidGhostZoneWidth();
        }

        notifyPatchProviders(region, ParentType::INNER_SET, globalNanoStep());
    }

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
                (*i)->put(*oldGrid, region, nanoStep);
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
            (*i)->get(
                &*oldGrid,
                region,
                nanoStep);
        }
    }

    inline std::size_t globalNanoStep() const
    {
        return curStep * NANO_STEPS + curNanoStep;
    }

    inline void initGrids()
    {
        Coord<DIM> topoDim = initializer->gridDimensions();
        CoordBox<DIM> gridBox;
        guessOffset(&gridBox.origin, &gridBox.dimensions);
        oldGrid.reset(new GridType(gridBox, CELL_TYPE(), CELL_TYPE(), topoDim));
        newGrid.reset(new GridType(gridBox, CELL_TYPE(), CELL_TYPE(), topoDim));
        initializer->grid(&*oldGrid);
        newGrid->getEdgeCell() = oldGrid->getEdgeCell();
        resetValidGhostZoneWidth();

        notifyPatchAccepters(
            rim(),
            ParentType::GHOST,
            globalNanoStep());
        notifyPatchAccepters(
            innerSet(0),
            ParentType::INNER_SET,
            globalNanoStep());

        kernelBuffer = PatchBufferType1(getVolatileKernel());
        rimBuffer = PatchBufferType2(rim());
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
        std::cout << "updateGhost()\n";
        {
            TimeComputeGhost t(&chronometer);

            // fixme: skip all this ghost zone buffering for
            // ghostZoneWidth == 1?

            // 1: Prepare grid. The following update of the ghostzone will
            // destroy parts of the kernel, which is why we'll
            // save/restore those.
            saveKernel();
            // We need to restore the rim since it got destroyed while the
            // kernel was updated.
            restoreRim(false);
        }

        // 2: actual ghostzone update
        std::size_t oldNanoStep = curNanoStep;
        std::size_t oldStep = curStep;
        std::size_t curGlobalNanoStep = globalNanoStep();

        for (std::size_t t = 0; t < ghostZoneWidth(); ++t) {
            notifyPatchProviders(rim(t), ParentType::GHOST, globalNanoStep());

            {
                TimeComputeGhost timer(&chronometer);

                const Region<DIM>& region = rim(t + 1);
                UpdateFunctor<CELL_TYPE>()(
                    region,
                    Coord<DIM>(),
                    Coord<DIM>(),
                    *oldGrid,
                    &*newGrid,
                    curNanoStep);

                ++curNanoStep;
                if (curNanoStep == NANO_STEPS) {
                    curNanoStep = 0;
                    curStep++;
                }

                std::swap(oldGrid, newGrid);

                ++curGlobalNanoStep;
            }

            notifyPatchAccepters(rim(), ParentType::GHOST, curGlobalNanoStep);
        }

        {
            TimeComputeGhost t(&chronometer);
            curNanoStep = oldNanoStep;
            curStep = oldStep;

            saveRim(curGlobalNanoStep);
            if (ghostZoneWidth() % 2) {
                std::swap(oldGrid, newGrid);
            }

            // 3: restore grid for kernel update
            restoreRim(true);
            restoreKernel();
        }
    }

private:

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

    inline void resetValidGhostZoneWidth()
    {
        validGhostZoneWidth = ghostZoneWidth();
    }

    inline void saveRim(std::size_t nanoStep)
    {
        rimBuffer.pushRequest(nanoStep);
        rimBuffer.put(*oldGrid, rim(), nanoStep);
    }

    inline void restoreRim(bool remove)
    {
        rimBuffer.get(&*oldGrid, rim(), globalNanoStep(), remove);
    }

    inline void saveKernel()
    {
        kernelBuffer.pushRequest(globalNanoStep());
        kernelBuffer.put(*oldGrid,
                         innerSet(ghostZoneWidth()),
                         globalNanoStep());
    }

    inline void restoreKernel()
    {
        kernelBuffer.get(
            &*oldGrid,
            getVolatileKernel(),
            globalNanoStep(),
            true);
    }
};

}
}

#endif
