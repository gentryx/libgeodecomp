#ifndef LIBGEODECOMP_PARALLELIZATION_HPXSIMULATOR_HPXSTEPPER_H
#define LIBGEODECOMP_PARALLELIZATION_HPXSIMULATOR_HPXSTEPPER_H

#include <libgeodecomp/parallelization/hiparsimulator/commonstepper.h>
#include <libgeodecomp/storage/patchbufferfixed.h>
#include <libgeodecomp/storage/updatefunctor.h>

#include <hpx/async.hpp>
#include <hpx/lcos/wait_all.hpp>

namespace LibGeoDecomp {
namespace HiParSimulator {

template <typename CELL_TYPE>
class HpxStepper : public CommonStepper<CELL_TYPE>
{
public:
    friend class HpxStepperRegionTest;
    friend class HpxStepperBasicTest;
    friend class HpxStepperTest;

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

    inline HpxStepper(
        boost::shared_ptr<PartitionManagerType> partitionManager,
        boost::shared_ptr<Initializer<CELL_TYPE> > initializer,
        const PatchAccepterVec& ghostZonePatchAccepters = PatchAccepterVec(),
        const PatchAccepterVec& innerSetPatchAccepters = PatchAccepterVec()) :
        CommonStepper<CELL_TYPE>(
            partitionManager,
            initializer,
            ghostZonePatchAccepters,
            innerSetPatchAccepters),
        asyncThreshold(boost::lexical_cast<std::size_t>(
                           hpx::get_config_entry("LibGeoDecomp.asyncThreshold", "0")))
    {
        hpx::async(&HpxStepper::initGrids, this).wait();
    }
private:
    hpx::lcos::local::spinlock gridMutex;
    std::size_t asyncThreshold;

    void update1()
    {
        TimeTotal t(&chronometer);
        unsigned index = ghostZoneWidth() - --validGhostZoneWidth;
        const Region<DIM>& region = innerSet(index);
        {
            TimeComputeInner t(&chronometer);

            updateRegion(innerSet(index), curNanoStep);
            std::swap(oldGrid, newGrid);

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
            innerSet(ghostZoneWidth()),
            ParentType::INNER_SET,
            globalNanoStep());

        saveRim(globalNanoStep());
        updateGhost();
    }

    inline void notifyPatchAccepters(
        const Region<DIM>& region,
        const typename ParentType::PatchType& patchType,
        std::size_t nanoStep)
    {
        TimePatchAccepters t(&chronometer);

        std::vector<hpx::future<void> > patchAcceptersFutures;
        patchAcceptersFutures.reserve(patchAccepters[patchType].size());

        for (typename ParentType::PatchAccepterList::iterator i =
                 patchAccepters[patchType].begin();
             i != patchAccepters[patchType].end();
             ++i) {
            if (nanoStep == (*i)->nextRequiredNanoStep()) {
                patchAcceptersFutures.push_back(
                    hpx::async(
                        [this, i, region, nanoStep]()
                        {
                            (*i)->put(*oldGrid, region, nanoStep);
                        }
                    )
                );
            }
        }

        hpx::wait_all(patchAcceptersFutures);
    }

    inline void notifyPatchProviders(
        const Region<DIM>& region,
        const typename ParentType::PatchType& patchType,
        std::size_t nanoStep)
    {
        TimePatchProviders t(&chronometer);

        std::vector<hpx::future<void> > patchProvidersFutures;
        patchProvidersFutures.reserve(patchProviders[patchType].size());

        for (typename ParentType::PatchProviderList::iterator i =
                 patchProviders[patchType].begin();
             i != patchProviders[patchType].end();
             ++i) {
                patchProvidersFutures.push_back(
                    hpx::async(
                        [this, i, region, nanoStep]() mutable
                        {
                            (*i)->get(&*oldGrid, region, nanoStep, gridMutex, true);
                        }
                    )
                );
        }

        hpx::wait_all(patchProvidersFutures);
    }

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

                updateRegion(rim(t + 1), curNanoStep);

                ++curNanoStep;
                if (curNanoStep == NANO_STEPS) {
                    curNanoStep = 0;
                    curStep++;
                }

                std::swap(oldGrid, newGrid);

                ++curGlobalNanoStep;
            }

            notifyPatchAccepters(rim(ghostZoneWidth()), ParentType::GHOST, curGlobalNanoStep);
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

    void updateRegion(const Region<DIM>& region, std::size_t curNanoStep)
    {
        std::vector<hpx::future<void> > updateFutures;
        updateFutures.reserve(region.numStreaks());
        Region<DIM> small_region;
        for (typename Region<DIM>::StreakIterator i = region.beginStreak();
             i != region.endStreak();
             ++i) {

            if(std::size_t(i->length()) > asyncThreshold) {
                Region<DIM> r;
                r << *i;
                updateFutures.push_back(
                    hpx::async(
                        hpx::util::bind(
                            UpdateFunctor<CELL_TYPE>(),
                            r,
                            Coord<DIM>(),
                            Coord<DIM>(),
                            boost::cref(*oldGrid),
                            &*newGrid,
                            curNanoStep
                        )
                    )
                );
            }
            else {
                small_region << *i;
                if(small_region.size() > asyncThreshold) {
                    updateFutures.push_back(
                        hpx::async(
                            hpx::util::bind(
                                UpdateFunctor<CELL_TYPE>(),
                                small_region,
                                Coord<DIM>(),
                                Coord<DIM>(),
                                boost::cref(*oldGrid),
                                &*newGrid,
                                curNanoStep
                            )
                        )
                    );
                    small_region.clear();
                }
            }
        }

        if(!small_region.empty()) {
            UpdateFunctor<CELL_TYPE>()(
                small_region,
                Coord<DIM>(),
                Coord<DIM>(),
                *oldGrid,
                &*newGrid,
                curNanoStep);
        }

        hpx::wait_all(updateFutures);
    }
};

}
}

#endif
