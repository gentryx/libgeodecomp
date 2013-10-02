#include <libgeodecomp/config.h>
#ifdef LIBGEODECOMP_FEATURE_MPI
#ifndef LIBGEODECOMP_PARALLELIZATION_HIPARSIMULATOR_H
#define LIBGEODECOMP_PARALLELIZATION_HIPARSIMULATOR_H

#include <cmath>
#include <stdexcept>
#include <libgeodecomp/loadbalancer/loadbalancer.h>
#include <libgeodecomp/misc/supermap.h>
#include <libgeodecomp/misc/statistics.h>
#include <libgeodecomp/mpilayer/mpilayer.h>
#include <libgeodecomp/parallelization/distributedsimulator.h>
#include <libgeodecomp/parallelization/hiparsimulator/partitions/stripingpartition.h>
#include <libgeodecomp/parallelization/hiparsimulator/parallelwriteradapter.h>
#include <libgeodecomp/parallelization/hiparsimulator/steereradapter.h>
#include <libgeodecomp/parallelization/hiparsimulator/updategroup.h>

namespace LibGeoDecomp {
namespace HiParSimulator {

enum EventPoint {LOAD_BALANCING, END};
typedef SuperSet<EventPoint> EventSet;
typedef SuperMap<long, EventSet> EventMap;

template<class CELL_TYPE, class PARTITION>
class HiParSimulator : public DistributedSimulator<CELL_TYPE>
{
public:
    friend class HiParSimulatorTest;
    using DistributedSimulator<CELL_TYPE>::NANO_STEPS;
    typedef typename DistributedSimulator<CELL_TYPE>::Topology Topology;
    typedef DistributedSimulator<CELL_TYPE> ParentType;
    typedef UpdateGroup<CELL_TYPE> UpdateGroupType;
    typedef typename ParentType::GridType GridType;
    typedef ParallelWriterAdapter<typename UpdateGroupType::GridType, CELL_TYPE> ParallelWriterAdapterType;
    typedef SteererAdapter<typename UpdateGroupType::GridType, CELL_TYPE> SteererAdapterType;
    static const int DIM = Topology::DIM;

    inline HiParSimulator(
        Initializer<CELL_TYPE> *initializer,
        LoadBalancer *balancer = 0,
        unsigned loadBalancingPeriod = 1,
        unsigned ghostZoneWidth = 1,
        MPI_Datatype cellMPIDatatype = Typemaps::lookup<CELL_TYPE>(),
        MPI_Comm communicator = MPI_COMM_WORLD) :
        ParentType(initializer),
        balancer(balancer),
        loadBalancingPeriod(loadBalancingPeriod * NANO_STEPS),
        ghostZoneWidth(ghostZoneWidth),
        mpiLayer(communicator),
        cellMPIDatatype(cellMPIDatatype)
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

    inline std::vector<Statistics> gatherStatistics()
    {
        Statistics stat =
        {
            totalTime,
            updateGroup->stepper->computeTimeInner,
            updateGroup->stepper->computeTimeGhost,
            updateGroup->stepper->patchAcceptersTime,
            updateGroup->stepper->patchProvidersTime,
        };

        MPI::Aint displacements[] = {0};
        MPI::Datatype memberTypes[] = {MPI_CHAR};
        int lengths[] = {sizeof(Statistics)};
        MPI::Datatype objType
            = MPI::Datatype::Create_struct(1, lengths, displacements, memberTypes);
        objType.Commit();
    
        return mpiLayer.gather(stat, 0, objType);
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
        boost::shared_ptr<Steerer<CELL_TYPE> > steererPointer(steerer);
        steerers << steererPointer;

        // two adapters needed, just as for the writers
        typename UpdateGroupType::PatchProviderPtr adapterGhost(
            new SteererAdapterType(
                steererPointer,
                initializer->startStep(),
                initializer->maxSteps(),
                initializer->gridDimensions(),
                mpiLayer.rank(),
                false));
        typename UpdateGroupType::PatchProviderPtr adapterInnerSet(
            new SteererAdapterType(
                steererPointer,
                initializer->startStep(),
                initializer->maxSteps(),
                initializer->gridDimensions(),
                mpiLayer.rank(),
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
                initializer->gridDimensions(),
                mpiLayer.rank(),
                false));
        typename UpdateGroupType::PatchAccepterPtr adapterInnerSet(
            new ParallelWriterAdapterType(
                writers.back(),
                initializer->startStep(),
                initializer->maxSteps(),
                initializer->gridDimensions(),
                mpiLayer.rank(),
                true));

        writerAdaptersGhost.push_back(adapterGhost);
        writerAdaptersInner.push_back(adapterInnerSet);
    }

private:
    using DistributedSimulator<CELL_TYPE>::initializer;
    using DistributedSimulator<CELL_TYPE>::steerers;
    using DistributedSimulator<CELL_TYPE>::writers;

    boost::shared_ptr<LoadBalancer> balancer;
    unsigned loadBalancingPeriod;
    unsigned ghostZoneWidth;
    EventMap events;
    PartitionManager<Topology> partitionManager;
    MPILayer mpiLayer;
    MPI_Datatype cellMPIDatatype;
    boost::shared_ptr<UpdateGroupType> updateGroup;
    typename UpdateGroupType::PatchProviderVec steererAdaptersGhost;
    typename UpdateGroupType::PatchProviderVec steererAdaptersInner;
    typename UpdateGroupType::PatchAccepterVec writerAdaptersGhost;
    typename UpdateGroupType::PatchAccepterVec writerAdaptersInner;

    double totalTime;

    double getCellSpeed(APITraits::FalseType) const
    {
        return 1.0;
    }

    double getCellSpeed(APITraits::TrueType) const
    {
        return CELL_TYPE::speed();
    }

    std::vector<std::size_t> initialWeights(std::size_t items, std::size_t size) const
    {
        double mySpeed = getCellSpeed(typename APITraits::SelectSpeed<CELL_TYPE>::Value());
        std::vector<double> speeds = mpiLayer.allGather(mySpeed);
        double s = sum(speeds);
        std::vector<std::size_t> ret(size);

        std::size_t lastPos = 0;
        double partialSum = 0.0;
        if(size > 1) {
            for (std::size_t i = 0; i < size -1; i++) {
                partialSum += speeds[i];
                std::size_t nextPos = items * partialSum / s;
                ret[i] = nextPos - lastPos;
                lastPos = nextPos;
            }
        }
        ret[size-1] = items - lastPos;

        return ret;
    }

    inline void nanoStep(const long& s)
    {
#ifdef LIBGEODECOMP_FEATURE_HPX
        hpx::util::high_resolution_timer timer;
#endif
        long remainingNanoSteps = s;
        while (remainingNanoSteps > 0) {
            long hop = std::min(remainingNanoSteps, timeToNextEvent());
            updateGroup->update(hop);
            handleEvents();
            remainingNanoSteps -= hop;
        }
#ifdef LIBGEODECOMP_FEATURE_HPX
        totalTime = timer.elapsed();
#endif
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

        boost::shared_ptr<PARTITION> partition(
            new PARTITION(
                box.origin,
                box.dimensions,
                0,
                initialWeights(box.dimensions.prod(),
                               mpiLayer.size())));
        

        updateGroup.reset(
            new UpdateGroupType(
                partition,
                box,
                ghostZoneWidth,
                initializer,
                static_cast<VanillaStepper<CELL_TYPE>*>(0),
                writerAdaptersGhost,
                writerAdaptersInner,
                steererAdaptersGhost,
                steererAdaptersInner,
                cellMPIDatatype,
                mpiLayer.communicator()));

        writerAdaptersGhost.clear();
        writerAdaptersInner.clear();

        initEvents();
    }

    inline void initEvents()
    {
        events.clear();
        long lastNanoStep = initializer->maxSteps() * NANO_STEPS;
        events[lastNanoStep] << END;

        insertNextLoadBalancingEvent();
    }

    inline void handleEvents()
    {
        if (currentNanoStep() > events.begin()->first) {
            throw std::logic_error("stale event found, should have been handled previously");
        }
        if (currentNanoStep() < events.begin()->first) {
            // don't need to handle future events now
            return;
        }

        const EventSet& curEvents = events.begin()->second;
        for (EventSet::const_iterator i = curEvents.begin(); i != curEvents.end(); ++i) {
            if (*i == LOAD_BALANCING) {
                balanceLoad();
                insertNextLoadBalancingEvent();
            }
        }
        events.erase(events.begin());
    }

    inline void insertNextLoadBalancingEvent()
    {
        long nextLoadBalancing = currentNanoStep() + loadBalancingPeriod;
        events[nextLoadBalancing] << LOAD_BALANCING;
    }

    inline long currentNanoStep() const
    {
        std::pair<int, int> now = updateGroup->currentStep();
        return (long)now.first * NANO_STEPS + now.second;
    }

    /**
     * returns the number of nano steps until the next event needs to be handled.
     */
    inline long timeToNextEvent() const
    {
        return events.begin()->first - currentNanoStep();
    }

    /**
     * returns the number of nano steps until simulation end.
     */
    inline long timeToLastEvent() const
    {
        return  events.rbegin()->first - currentNanoStep();
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
}

#endif
#endif
