#ifndef LIBGEODECOMP_PARALLELIZATION_HIPARSIMULATOR_H
#define LIBGEODECOMP_PARALLELIZATION_HIPARSIMULATOR_H

#include <libgeodecomp/config.h>
#ifdef LIBGEODECOMP_WITH_MPI

#include <libgeodecomp/communication/mpilayer.h>
#include <libgeodecomp/geometry/partitions/stripingpartition.h>
#include <libgeodecomp/geometry/partitions/ptscotchunstructuredpartition.h>
#include <libgeodecomp/geometry/partitions/unstructuredstripingpartition.h>
#include <libgeodecomp/geometry/partitions/distributedptscotchunstructuredpartition.h>
#include <libgeodecomp/loadbalancer/loadbalancer.h>
#include <libgeodecomp/misc/sharedptr.h>
#include <libgeodecomp/parallelization/hierarchicalsimulator.h>
#include <libgeodecomp/parallelization/nesting/parallelwriteradapter.h>
#include <libgeodecomp/parallelization/nesting/steereradapter.h>
#include <libgeodecomp/parallelization/nesting/mpiupdategroup.h>
#include <cmath>
#include <stdexcept>

namespace LibGeoDecomp {

/**
 * The HiParSimulator implements our hierarchical parallelization
 * algorithm which delivers best-of-breed latency hiding (wide ghost
 * zones combined with overlapping communication and calculation)
 * while remaining flexible with regard to the domain decomposition
 * method. It can combine different types of parallelization: MPI for
 * inter-node or inter-NUMA-domain communication and OpenMP and/or
 * CUDA for local paralelism.
 *
 * fixme: check if code runs with a communicator which is merely a subset of MPI_COMM_WORLD
 */
template<
    typename CELL_TYPE,
    typename PARTITION,
    typename STEPPER = VanillaStepper<CELL_TYPE, UpdateFunctorHelpers::ConcurrencyEnableOpenMP> >
class HiParSimulator : public HierarchicalSimulator<CELL_TYPE>
{
public:
    friend class HiParSimulatorTest;
    using DistributedSimulator<CELL_TYPE>::NANO_STEPS;
    using DistributedSimulator<CELL_TYPE>::chronometer;
    using HierarchicalSimulator<CELL_TYPE>::handleEvents;
    using HierarchicalSimulator<CELL_TYPE>::enableFineGrainedParallelism;
    using HierarchicalSimulator<CELL_TYPE>::events;
    using HierarchicalSimulator<CELL_TYPE>::initEvents;
    using HierarchicalSimulator<CELL_TYPE>::timeToLastEvent;
    using HierarchicalSimulator<CELL_TYPE>::timeToNextEvent;

    typedef typename DistributedSimulator<CELL_TYPE>::Topology Topology;
    typedef HierarchicalSimulator<CELL_TYPE> ParentType;
    typedef MPIUpdateGroup<CELL_TYPE> UpdateGroupType;
    typedef typename ParentType::GridType GridType;
    typedef ParallelWriterAdapter<typename UpdateGroupType::GridType, CELL_TYPE> ParallelWriterAdapterType;
    typedef SteererAdapter<typename UpdateGroupType::GridType, CELL_TYPE> SteererAdapterType;

    static const int DIM = Topology::DIM;

    inline explicit HiParSimulator(
        Initializer<CELL_TYPE> *initializer,
        LoadBalancer *balancer = 0,
        unsigned loadBalancingPeriod = 1,
        unsigned ghostZoneWidth = 1,
        bool enableFineGrainedParallelism = false,
        MPI_Comm communicator = MPI_COMM_WORLD) :
        ParentType(
            initializer,
            loadBalancingPeriod * NANO_STEPS,
            enableFineGrainedParallelism),
        balancer(balancer),
        ghostZoneWidth(ghostZoneWidth),
        mpiLayer(communicator)
    {}

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
        if (updateGroup) {
            return updateGroup->currentStep().first;
        } else {
            return initializer->startStep();
        }
    }

    virtual void addSteerer(Steerer<CELL_TYPE> *steerer)
    {
        DistributedSimulator<CELL_TYPE>::addSteerer(steerer);

        // two adapters needed, just as for the writers
        typename UpdateGroupType::PatchProviderPtr adapterGhost(
            new SteererAdapterType(
                steerers.back(),
                initializer->startStep(),
                initializer->maxSteps(),
                false));

        typename UpdateGroupType::PatchProviderPtr adapterInnerSet(
            new SteererAdapterType(
                steerers.back(),
                initializer->startStep(),
                initializer->maxSteps(),
                true));

        steererAdaptersGhost.push_back(adapterGhost);
        steererAdaptersInner.push_back(adapterInnerSet);
    }

    virtual void addWriter(ParallelWriter<CELL_TYPE> *writer)
    {
        DistributedSimulator<CELL_TYPE>::addWriter(writer);

        // we need two adapters as each ParallelWriter needs to be
        // notified twice: once for the (inner) ghost zone, and once
        // for the inner set.
        typename UpdateGroupType::PatchAccepterPtr adapterGhost(
            new ParallelWriterAdapterType(
                writers.back(),
                initializer->startStep(),
                initializer->maxSteps(),
                false));
        typename UpdateGroupType::PatchAccepterPtr adapterInnerSet(
            new ParallelWriterAdapterType(
                writers.back(),
                initializer->startStep(),
                initializer->maxSteps(),
                true));

        writerAdaptersGhost.push_back(adapterGhost);
        writerAdaptersInner.push_back(adapterInnerSet);
    }

    std::vector<Chronometer> gatherStatistics()
    {
        Chronometer stats = chronometer + updateGroup->statistics();
        return mpiLayer.gather(stats, 0);
    }

private:
    using DistributedSimulator<CELL_TYPE>::initializer;
    using DistributedSimulator<CELL_TYPE>::steerers;
    using DistributedSimulator<CELL_TYPE>::writers;

    SharedPtr<LoadBalancer>::Type balancer;
    unsigned ghostZoneWidth;
    MPILayer mpiLayer;
    typename SharedPtr<UpdateGroupType>::Type updateGroup;

    typename UpdateGroupType::PatchProviderVec steererAdaptersGhost;
    typename UpdateGroupType::PatchProviderVec steererAdaptersInner;
    typename UpdateGroupType::PatchAccepterVec writerAdaptersGhost;
    typename UpdateGroupType::PatchAccepterVec writerAdaptersInner;

    inline void nanoStep(long s)
    {
        long remainingNanoSteps = s;
        while (remainingNanoSteps > 0) {
            long hop = std::min(remainingNanoSteps, timeToNextEvent());
            updateGroup->update(hop);
            handleEvents();
            remainingNanoSteps -= hop;
        }
    }

    /**
     * We need to do late/lazy initialization to give the user time to
     * add ParallelWriter objects before calling run(). Writers may
     * only be added savely to an UpdateGroup upon creation because of
     * the way the Stepper handles ghostzone updates. It's a long
     * story... At the end of the day this remains the best compromise
     * of hiding complexity (in the Stepper) and a convenient API of
     * the Simulator on the one hand, and avoiding objects with an
     * uninitialized state on the other.
     */
    inline void initSimulation()
    {
        if (updateGroup) {
            return;
        }

        CoordBox<DIM> box = initializer->gridBox();
        Region<DIM> globalRegion;
        globalRegion << box;

        double mySpeed = APITraits::SelectSpeedGuide<CELL_TYPE>::value();
        std::vector<double> rankSpeeds = mpiLayer.allGather(mySpeed);
        std::vector<std::size_t> weights = LoadBalancer::initialWeights(
            box.dimensions.prod(),
            rankSpeeds);

        typename SharedPtr<PARTITION>::Type partition(
            new PARTITION(
                box.origin,
                box.dimensions,
                0,
                weights,
                initializer->getAdjacency(globalRegion)));

        updateGroup.reset(
            new UpdateGroupType(
                partition,
                box,
                ghostZoneWidth,
                initializer,
                static_cast<STEPPER*>(0),
                writerAdaptersGhost,
                writerAdaptersInner,
                steererAdaptersGhost,
                steererAdaptersInner,
                enableFineGrainedParallelism,
                mpiLayer.communicator()));

        writerAdaptersGhost.clear();
        writerAdaptersInner.clear();
        steererAdaptersGhost.clear();
        steererAdaptersInner.clear();

        initEvents();
    }

    inline long currentNanoStep() const
    {
        std::pair<int, int> now = updateGroup->currentStep();
        return (long)now.first * NANO_STEPS + now.second;
    }

    inline void balanceLoad()
    {
        if (mpiLayer.rank() == 0) {
            if (!balancer) {
                return;
            }

            LoadBalancer::LoadVec loads(mpiLayer.size(), 1.0);
            LoadBalancer::WeightVec newWeights =
                balancer->balance(updateGroup->getWeights(), loads);
            // fixme: actually balance the load!
        }
    }
};

}

#endif
#endif
