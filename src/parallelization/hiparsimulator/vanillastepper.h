#ifndef LIBGEODECOMP_PARALLELIZATION_HIPARSIMULATOR_VANILLASTEPPER_H
#define LIBGEODECOMP_PARALLELIZATION_HIPARSIMULATOR_VANILLASTEPPER_H

#include <libgeodecomp/parallelization/hiparsimulator/commonstepper.h>
#include <libgeodecomp/storage/updatefunctor.h>

namespace LibGeoDecomp {
namespace HiParSimulator {

/**
 * As its name implies, the VanillaStepper is the simplest
 * implementation of the Stepper concept: single threaded and no
 * accelerator offloading. It does however overlap communication and
 * calculation and support wide halos (halos = ghostzones). Ghost
 * zones of width k mean that synchronization only needs to be done
 * every k'th (nano) step.
 */
template<typename CELL_TYPE>
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

    using CommonStepper<CELL_TYPE>::initializer;
    using CommonStepper<CELL_TYPE>::patchAccepters;
    using CommonStepper<CELL_TYPE>::patchProviders;
    using CommonStepper<CELL_TYPE>::partitionManager;
    using CommonStepper<CELL_TYPE>::chronometer;
    using CommonStepper<CELL_TYPE>::notifyPatchAccepters;
    using CommonStepper<CELL_TYPE>::notifyPatchProviders;

    using CommonStepper<CELL_TYPE>::innerSet;
    using CommonStepper<CELL_TYPE>::saveKernel;
    using CommonStepper<CELL_TYPE>::restoreRim;
    using CommonStepper<CELL_TYPE>::globalNanoStep;
    using CommonStepper<CELL_TYPE>::rim;
    using CommonStepper<CELL_TYPE>::resetValidGhostZoneWidth;
    using CommonStepper<CELL_TYPE>::initGridsCommon;
    using CommonStepper<CELL_TYPE>::getVolatileKernel;
    using CommonStepper<CELL_TYPE>::saveRim;
    using CommonStepper<CELL_TYPE>::getInnerRim;
    using CommonStepper<CELL_TYPE>::restoreKernel;

    using CommonStepper<CELL_TYPE>::curStep;
    using CommonStepper<CELL_TYPE>::curNanoStep;
    using CommonStepper<CELL_TYPE>::validGhostZoneWidth;
    using CommonStepper<CELL_TYPE>::ghostZoneWidth;
    using CommonStepper<CELL_TYPE>::oldGrid;
    using CommonStepper<CELL_TYPE>::newGrid;
    using CommonStepper<CELL_TYPE>::rimBuffer;
    using CommonStepper<CELL_TYPE>::kernelBuffer;
    using CommonStepper<CELL_TYPE>::kernelFraction;

    inline VanillaStepper(
        boost::shared_ptr<PartitionManagerType> partitionManager,
        boost::shared_ptr<Initializer<CELL_TYPE> > initializer,
        const PatchAccepterVec& ghostZonePatchAccepters = PatchAccepterVec(),
        const PatchAccepterVec& innerSetPatchAccepters = PatchAccepterVec()) :
        CommonStepper<CELL_TYPE>(
            partitionManager,
            initializer,
            ghostZonePatchAccepters,
            innerSetPatchAccepters)
    {
        initGrids();
    }

private:
    inline void update1()
    {
        TimeTotal t(&chronometer);
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
                ++curStep;
            }
        }

        notifyPatchAccepters(region, ParentType::INNER_SET, globalNanoStep());

        if (validGhostZoneWidth == 0) {
            updateGhost();
            resetValidGhostZoneWidth();
        }

        notifyPatchProviders(region, ParentType::INNER_SET, globalNanoStep());
    }

    inline void initGrids()
    {
        initGridsCommon();

        notifyPatchAccepters(
            rim(),
            ParentType::GHOST,
            globalNanoStep());
        notifyPatchAccepters(
            innerSet(0),
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
};

}
}

#endif
