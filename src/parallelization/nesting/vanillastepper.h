#ifndef LIBGEODECOMP_PARALLELIZATION_NESTING_VANILLASTEPPER_H
#define LIBGEODECOMP_PARALLELIZATION_NESTING_VANILLASTEPPER_H

#include <libgeodecomp/parallelization/nesting/commonstepper.h>
#include <libgeodecomp/storage/updatefunctor.h>

namespace LibGeoDecomp {

/**
 * As its name implies, the VanillaStepper is the simplest
 * implementation of the Stepper concept: single threaded and no
 * accelerator offloading. It does however overlap communication and
 * calculation and support wide halos (halos = ghostzones). Ghost
 * zones of width k mean that synchronization only needs to be done
 * every k'th (nano) step.
 */
template<typename CELL_TYPE, typename CONCURRENCY_SPEC>
class VanillaStepper : public CommonStepper<CELL_TYPE>
{
public:
    friend class VanillaStepperRegionTest;
    friend class VanillaStepperBasicTest;
    friend class VanillaStepperTest;

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

    using ParentType::initializer;
    using ParentType::patchAccepters;
    using ParentType::patchProviders;
    using ParentType::partitionManager;
    using ParentType::chronometer;
    using ParentType::notifyPatchAccepters;
    using ParentType::notifyPatchProviders;

    using ParentType::innerSet;
    using ParentType::saveKernel;
    using ParentType::restoreRim;
    using ParentType::globalNanoStep;
    using ParentType::rim;
    using ParentType::resetValidGhostZoneWidth;
    using ParentType::initGridsCommon;
    using ParentType::getVolatileKernel;
    using ParentType::saveRim;
    using ParentType::getInnerRim;
    using ParentType::restoreKernel;

    using ParentType::curStep;
    using ParentType::curNanoStep;
    using ParentType::validGhostZoneWidth;
    using ParentType::ghostZoneWidth;
    using ParentType::oldGrid;
    using ParentType::newGrid;
    using ParentType::rimBuffer;
    using ParentType::kernelBuffer;
    using ParentType::kernelFraction;
    using ParentType::enableFineGrainedParallelism;

    inline VanillaStepper(
        boost::shared_ptr<PartitionManagerType> partitionManager,
        boost::shared_ptr<Initializer<CELL_TYPE> > initializer,
        const PatchAccepterVec& ghostZonePatchAccepters = PatchAccepterVec(),
        const PatchAccepterVec& innerSetPatchAccepters = PatchAccepterVec(),
        const PatchProviderVec& ghostZonePatchProviders = PatchProviderVec(),
        const PatchProviderVec& innerSetPatchProviders = PatchProviderVec(),
        bool enableFineGrainedParallelism = false) :
        ParentType(
            partitionManager,
            initializer,
            ghostZonePatchAccepters,
            innerSetPatchAccepters,
            ghostZonePatchProviders,
            innerSetPatchProviders,
            enableFineGrainedParallelism)
    {
        initGrids();
    }

private:
    inline void update1()
    {
        using std::swap;
        TimeTotal t(&chronometer);
        unsigned index = ghostZoneWidth() - --validGhostZoneWidth;
        const Region<DIM>& region = innerSet(index);
        {
            TimeComputeInner t(&chronometer);

            UpdateFunctor<CELL_TYPE, CONCURRENCY_SPEC>()(
                region,
                Coord<DIM>(),
                Coord<DIM>(),
                *oldGrid,
                &*newGrid,
                curNanoStep,
                CONCURRENCY_SPEC(false, enableFineGrainedParallelism));
            swap(oldGrid, newGrid);

            ++curNanoStep;
            if (curNanoStep == NANO_STEPS) {
                curNanoStep = 0;
                ++curStep;
            }
        }

        notifyPatchAccepters(innerSet(ghostZoneWidth()), ParentType::INNER_SET, globalNanoStep());

        if (validGhostZoneWidth == 0) {
            updateGhost();
            resetValidGhostZoneWidth();
        }

        index = ghostZoneWidth() - validGhostZoneWidth;
        const Region<DIM>& nextRegion = innerSet(index);
        notifyPatchProviders(nextRegion, ParentType::INNER_SET, globalNanoStep());
    }

    inline void initGrids()
    {
        initGridsCommon();

        notifyPatchAccepters(
            rim(),
            ParentType::GHOST,
            globalNanoStep());
        notifyPatchAccepters(
            innerSet(ghostZoneWidth()),
            ParentType::INNER_SET,
            globalNanoStep());

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
        using std::swap;
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
                UpdateFunctor<CELL_TYPE, CONCURRENCY_SPEC>()(
                    region,
                    Coord<DIM>(),
                    Coord<DIM>(),
                    *oldGrid,
                    &*newGrid,
                    curNanoStep,
                    CONCURRENCY_SPEC(true, enableFineGrainedParallelism));

                ++curNanoStep;
                if (curNanoStep == NANO_STEPS) {
                    curNanoStep = 0;
                    curStep++;
                }

                swap(oldGrid, newGrid);

                ++curGlobalNanoStep;
            }

            notifyPatchAccepters(rim(ghostZoneWidth()), ParentType::GHOST, curGlobalNanoStep);
        }

        {
            TimeComputeGhost t(&chronometer);

            saveRim(curGlobalNanoStep);
            if (ghostZoneWidth() % 2) {
                swap(oldGrid, newGrid);
            }

            // 3: restore grid for kernel update
            curNanoStep = oldNanoStep;
            curStep = oldStep;
            restoreRim(true);
            restoreKernel();
        }
    }
};

}

#endif
