#ifndef LIBGEODECOMP_PARALLELIZATION_NESTING_HPXSTEPPER_H
#define LIBGEODECOMP_PARALLELIZATION_NESTING_HPXSTEPPER_H

#include <libgeodecomp/parallelization/nesting/vanillastepper.h>
#include <libgeodecomp/storage/updatefunctor.h>

namespace LibGeoDecomp {

/**
 * The HPXStepper is a implementation of the Stepper concept: mulit threaded
 * and no accelerator offloading. It does however overlap communication and
 * calculation and support wide halos (halos = ghostzones). Ghost
 * zones of width k mean that synchronization only needs to be done
 * every k'th (nano) step.
 */
template<typename CELL_TYPE, typename CONCURRENCY_SPEC>
class HPXStepper : public VanillaStepper<CELL_TYPE, CONCURRENCY_SPEC>
{
public:
    typedef typename Stepper<CELL_TYPE>::Topology Topology;
    const static int DIM = Topology::DIM;
    const static unsigned NANO_STEPS = APITraits::SelectNanoSteps<CELL_TYPE>::VALUE;

    typedef VanillaStepper<CELL_TYPE, CONCURRENCY_SPEC> ParentType;
    typedef typename ParentType::GridType GridType;
    typedef PartitionManager<Topology> PartitionManagerType;
    typedef PatchBufferFixed<GridType, GridType, 1> PatchBufferType1;
    typedef PatchBufferFixed<GridType, GridType, 2> PatchBufferType2;
    typedef typename ParentType::PatchAccepterVec PatchAccepterVec;
    typedef typename ParentType::PatchProviderVec PatchProviderVec;

    using ParentType::patchAccepters;
    using ParentType::patchProviders;
    using ParentType::partitionManager;
    using ParentType::chronometer;

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

    inline HPXStepper(
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
    {}

    inline hpx::future<void> notifyPatchAcceptersAsync(
        const Region<DIM>& region,
        const typename ParentType::PatchType& patchType,
        std::size_t nanoStep)
    {
        TimePatchAccepters t(&chronometer);
        std::vector<hpx::future<void> > putFutures;
        putFutures.reserve(patchAccepters[patchType].size());

        for (typename ParentType::PatchAccepterList::iterator i =
                 patchAccepters[patchType].begin();
             i != patchAccepters[patchType].end();
             ++i) {
            if (nanoStep == (*i)->nextRequiredNanoStep()) {
                putFutures << (*i)->putAsync(
                    *oldGrid,
                    region,
                    partitionManager->getSimulationArea(),
                    nanoStep,
                    partitionManager->rank());
            }
        }
        return hpx::when_all(putFutures);
    }

    inline void notifyPatchAccepters(
        const Region<DIM>& region,
        const typename ParentType::PatchType& patchType,
        std::size_t nanoStep)
    {
        return notifyPatchAcceptersAsync(region, patchType, nanoStep).get();
    }

    inline hpx::future<void> notifyPatchProvidersAsync(
        const Region<DIM>& region,
        const typename ParentType::PatchType& patchType,
        std::size_t nanoStep)
    {
        TimePatchProviders t(&chronometer);
        std::vector<hpx::future<void> > getFutures;
        getFutures.reserve(patchProviders[patchType].size());

        for (typename ParentType::PatchProviderList::iterator i =
                 patchProviders[patchType].begin();
             i != patchProviders[patchType].end();
             ++i) {
            if (nanoStep == (*i)->nextAvailableNanoStep()) {
                getFutures << (*i)->getAsync(
                    &*oldGrid,
                    region,
                    partitionManager->getSimulationArea(),
                    nanoStep,
                    partitionManager->rank(),
                    true);
            }
        }

        return hpx::when_all(getFutures);
    }

    inline void notifyPatchProviders(
        const Region<DIM>& region,
        const typename ParentType::PatchType& patchType,
        std::size_t nanoStep)
    {
        notifyPatchProvidersAsync(region, patchType, nanoStep).get();
    }
};

}

#endif
