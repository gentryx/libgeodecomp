#ifndef LIBGEODECOMP_PARALLELIZATION_HPXSIMULATOR_H
#define LIBGEODECOMP_PARALLELIZATION_HPXSIMULATOR_H

#include <libgeodecomp/config.h>
#ifdef LIBGEODECOMP_WITH_HPX

#include <libgeodecomp/misc/cudaboostworkaround.h>
#include <hpx/config.hpp>
#include <hpx/serialization/set.hpp>
#include <hpx/serialization/string.hpp>
#include <hpx/serialization/vector.hpp>
#include <hpx/collectives/broadcast.hpp>

#include <libgeodecomp/communication/hpxserializationwrapper.h>
#include <libgeodecomp/geometry/partitions/stripingpartition.h>
#include <libgeodecomp/loadbalancer/loadbalancer.h>
#include <libgeodecomp/misc/sharedptr.h>
#include <libgeodecomp/parallelization/hierarchicalsimulator.h>
#include <libgeodecomp/parallelization/nesting/hpxupdategroup.h>
#include <libgeodecomp/parallelization/nesting/parallelwriteradapter.h>
#include <libgeodecomp/parallelization/nesting/steereradapter.h>
#include <libgeodecomp/parallelization/nesting/stepper.h>
#include <libgeodecomp/parallelization/nesting/hpxstepper.h>

namespace LibGeoDecomp {
namespace HpxSimulatorHelpers {

void gatherAndBroadcastLocalityIndices(
    double speedGuide,
    std::vector<double> *globalUpdateGroupSpeeds,
    std::vector<std::size_t> *localityIndices,
    const std::string& basename,
    const std::vector<double> updateGroupSpeeds);

}

template<
    class CELL_TYPE,
    class PARTITION,
    class STEPPER=LibGeoDecomp::HPXStepper<CELL_TYPE, UpdateFunctorHelpers::ConcurrencyEnableHPX>
>
class HpxSimulator : public HierarchicalSimulator<CELL_TYPE>
{
public:
    friend class HpxSimulatorTest;
    using DistributedSimulator<CELL_TYPE>::NANO_STEPS;
    using HierarchicalSimulator<CELL_TYPE>::handleEvents;
    using HierarchicalSimulator<CELL_TYPE>::enableFineGrainedParallelism;
    using HierarchicalSimulator<CELL_TYPE>::events;
    using HierarchicalSimulator<CELL_TYPE>::initEvents;
    using HierarchicalSimulator<CELL_TYPE>::timeToLastEvent;
    using HierarchicalSimulator<CELL_TYPE>::timeToNextEvent;

    using typename DistributedSimulator<CELL_TYPE>::SteererPtr;
    using typename DistributedSimulator<CELL_TYPE>::WriterPtr;
    // fixme:
    typedef typename DistributedSimulator<CELL_TYPE>::Topology Topology;
    typedef LibGeoDecomp::HierarchicalSimulator<CELL_TYPE> ParentType;
    typedef HPXUpdateGroup<CELL_TYPE> UpdateGroupType;
    typedef typename ParentType::GridType GridType;
    typedef LibGeoDecomp::ParallelWriterAdapter<typename UpdateGroupType::GridType, CELL_TYPE> ParallelWriterAdapterType;
    typedef LibGeoDecomp::SteererAdapter<typename UpdateGroupType::GridType, CELL_TYPE> SteererAdapterType;
    typedef typename UpdateGroupType::PatchAccepterVec PatchAccepterVec;
    typedef typename UpdateGroupType::PatchProviderVec PatchProviderVec;
    typedef typename UpdateGroupType::PatchAccepterPtr PatchAccepterPtr;
    typedef typename UpdateGroupType::PatchProviderPtr PatchProviderPtr;
    typedef typename SharedPtr<UpdateGroupType>::Type UpdateGroupPtr;

    static const int DIM = Topology::DIM;

    /**
     * Creates an HpxSimulator. Parameters are essentially the same as
     * for the HiParSimulator. The vector updateGroupSpeeds controls
     * how many UpdateGroups will be created and how large their
     * individual domain should be.
     */
    inline HpxSimulator(
        Initializer<CELL_TYPE> *initializer,
        const std::vector<double> updateGroupSpeeds = std::vector<double>(1, 1.0),
        LoadBalancer *balancer = 0,
        const unsigned loadBalancingPeriod = 1,
        const unsigned ghostZoneWidth = 1,
        bool enableFineGrainedParallelism = false,
        std::string basename = "/HPXSimulator") :
        ParentType(
            initializer,
            loadBalancingPeriod * NANO_STEPS,
            enableFineGrainedParallelism),
        updateGroupSpeeds(updateGroupSpeeds),
        balancer(balancer),
        ghostZoneWidth(ghostZoneWidth),
        basename(basename),
        rank(hpx::get_locality_id())
    {
        HpxSimulatorHelpers::gatherAndBroadcastLocalityIndices(
            APITraits::SelectSpeedGuide<CELL_TYPE>::value(),
            &globalUpdateGroupSpeeds,
            &localityIndices,
            basename,
            updateGroupSpeeds);

        for (std::size_t i = localityIndices[rank + 0]; i < localityIndices[rank + 1]; ++i) {
            steererAdaptersGhost[i] = PatchProviderVec();
            steererAdaptersInner[i] = PatchProviderVec();

            writerAdaptersGhost[i] = PatchAccepterVec();
            writerAdaptersInner[i] = PatchAccepterVec();
        }
    }

    inline void run()
    {
        initSimulation();
        nanoStep(timeToLastEvent());
    }

    inline void step()
    {
        initSimulation();
        nanoStep(NANO_STEPS);
    }

    virtual unsigned getStep() const
    {
        if (updateGroups.size() == 0) {
            return initializer->startStep();
        }

        return updateGroups[0]->currentStep().first;
    }

    virtual void addSteerer(Steerer<CELL_TYPE> *steerer)
    {
        for (std::size_t i = localityIndices[rank + 0]; i < localityIndices[rank + 1]; ++i) {
            SteererPtr steererSharedPointer(steerer);

            // two adapters needed, just as for the writers
            typename UpdateGroupType::PatchProviderPtr adapterGhost(
                new SteererAdapterType(
                    steererSharedPointer,
                    initializer->startStep(),
                    initializer->maxSteps(),
                    false));

            typename UpdateGroupType::PatchProviderPtr adapterInnerSet(
                new SteererAdapterType(
                    steererSharedPointer,
                    initializer->startStep(),
                    initializer->maxSteps(),
                    true));

            steererAdaptersGhost[i].push_back(adapterGhost);
            steererAdaptersInner[i].push_back(adapterInnerSet);

            if ((i + 1) < localityIndices[rank + 1]) {
                steerer = steerer->clone();
            }
        }
    }

    virtual void addWriter(ParallelWriter<CELL_TYPE> *writer)
    {
        for (std::size_t i = localityIndices[rank + 0]; i < localityIndices[rank + 1]; ++i) {
            WriterPtr writerSharedPointer(writer);
            // we need two adapters as each ParallelWriter needs to be
            // notified twice: once for the (inner) ghost zone, and once
            // for the inner set.
            PatchAccepterPtr adapterGhost(
                new ParallelWriterAdapterType(
                    writerSharedPointer,
                    initializer->startStep(),
                    initializer->maxSteps(),
                    false));
            PatchAccepterPtr adapterInnerSet(
                new ParallelWriterAdapterType(
                    writerSharedPointer,
                    initializer->startStep(),
                    initializer->maxSteps(),
                    true));

            writerAdaptersGhost[i].push_back(adapterGhost);
            writerAdaptersInner[i].push_back(adapterInnerSet);

            if ((i + 1) < localityIndices[rank + 1]) {
                writer = writer->clone();
            }
        }
    }

    std::size_t numUpdateGroups() const
    {
        return updateGroups.size();
    }

    std::vector<Chronometer> gatherStatistics()
    {
        // fixme: gather from all updategroups

        std::vector<Chronometer> statistics;
        statistics.reserve(updateGroups.size());

        for (auto& i: updateGroups) {
            statistics << i->statistics();
        }

        return statistics;
    }

private:
    using DistributedSimulator<CELL_TYPE>::initializer;
    using DistributedSimulator<CELL_TYPE>::steerers;
    using DistributedSimulator<CELL_TYPE>::writers;

    std::vector<double> updateGroupSpeeds;
    SharedPtr<LoadBalancer>::Type balancer;
    unsigned ghostZoneWidth;
    std::string basename;

    std::vector<UpdateGroupPtr> updateGroups;
    std::vector<double> globalUpdateGroupSpeeds;
    std::vector<std::size_t> localityIndices;
    std::size_t rank;

    std::map<std::size_t, typename UpdateGroupType::PatchProviderVec> steererAdaptersGhost;
    std::map<std::size_t, typename UpdateGroupType::PatchProviderVec> steererAdaptersInner;
    std::map<std::size_t, typename UpdateGroupType::PatchAccepterVec> writerAdaptersGhost;
    std::map<std::size_t, typename UpdateGroupType::PatchAccepterVec> writerAdaptersInner;

    inline void initSimulation()
    {
        if (updateGroups.size() != 0) {
            return;
        }

        CoordBox<DIM> box = initializer->gridBox();
        Region<DIM> globalRegion;
        globalRegion << box;

        std::vector<std::size_t> weights = LoadBalancer::initialWeights(
            box.dimensions.prod(),
            globalUpdateGroupSpeeds);

        typename SharedPtr<PARTITION>::Type partition(
            new PARTITION(
                box.origin,
                box.dimensions,
                0,
                weights,
                initializer->getAdjacency(globalRegion)));

        std::vector<hpx::future<UpdateGroupPtr> > updateGroupCreationFutures;

        for (std::size_t i = localityIndices[rank + 0]; i < localityIndices[rank + 1]; ++i) {
            updateGroupCreationFutures << hpx::async(&HpxSimulator::createUpdateGroup, this, i, partition);
        }
        updateGroups = hpx::util::unwrap(std::move(updateGroupCreationFutures));

        for (std::size_t i = localityIndices[rank + 0]; i < localityIndices[rank + 1]; ++i) {
            writerAdaptersGhost[i].clear();
            writerAdaptersInner[i].clear();
            steererAdaptersGhost[i].clear();
            steererAdaptersInner[i].clear();
        }

        initEvents();
    }

    inline long currentNanoStep() const
    {
        std::pair<int, int> now = updateGroups[0]->currentStep();
        return (long)now.first * NANO_STEPS + now.second;
    }

    inline void balanceLoad()
    {
        // fixme: do we need this after all?
    }

    void nanoStep(std::size_t remainingNanoSteps)
    {
        std::vector<hpx::future<void> > updateFutures;
        updateFutures.reserve(updateGroups.size());

        for (auto& i: updateGroups) {
            updateFutures << hpx::async(&UpdateGroupType::update, i, remainingNanoSteps);
        }

        hpx::lcos::wait_all(std::move(updateFutures));
    }

    UpdateGroupPtr createUpdateGroup(
        std::size_t rank,
        typename SharedPtr<PARTITION>::Type partition)
    {
        CoordBox<DIM> box = initializer->gridBox();

        return UpdateGroupPtr(
            new UpdateGroupType(
                partition,
                box,
                ghostZoneWidth,
                initializer,
                reinterpret_cast<STEPPER*>(0),
                writerAdaptersGhost[rank],
                writerAdaptersInner[rank],
                steererAdaptersGhost[rank],
                steererAdaptersInner[rank],
                enableFineGrainedParallelism,
                basename,
                rank));
    }
};

}

#endif
#endif
