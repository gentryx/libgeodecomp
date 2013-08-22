#include <libgeodecomp/config.h>
#ifdef LIBGEODECOMP_FEATURE_HPX
#ifndef LIBGEODECOMP_PARALLELIZATION_HPXSIMULATOR_H
#define LIBGEODECOMP_PARALLELIZATION_HPXSIMULATOR_H

#include <hpx/config.hpp>
#include <hpx/lcos/broadcast.hpp>
#include <libgeodecomp/loadbalancer/loadbalancer.h>
#include <libgeodecomp/misc/supermap.h>
#include <libgeodecomp/parallelization/distributedsimulator.h>
#include <libgeodecomp/parallelization/hiparsimulator/partitions/stripingpartition.h>
#include <libgeodecomp/parallelization/hpxsimulator/hpxstepper.h>
#include <libgeodecomp/parallelization/hpxsimulator/updategroup.h>
#include <libgeodecomp/parallelization/hpxsimulator/createupdategroups.h>

#include <boost/serialization/shared_ptr.hpp>

#define LIBGEODECOMP_REGISTER_HPX_SIMULATOR_DECLARATION(SIMULATOR, NAME)        \
    typedef                                                                     \
        SIMULATOR ::UpdateGroupType::ComponentType                              \
        BOOST_PP_CAT(NAME, UpdateGroupType);                                    \
    HPX_REGISTER_ACTION_DECLARATION(                                            \
        BOOST_PP_CAT(NAME, UpdateGroupType)::InitAction,                        \
        BOOST_PP_CAT(BOOST_PP_CAT(NAME, UpdateGroup), InitAction)               \
    );                                                                          \
    HPX_ACTION_USES_HUGE_STACK(                                                 \
        BOOST_PP_CAT(NAME, UpdateGroupType)::InitAction                         \
    );                                                                          \
    HPX_REGISTER_ACTION_DECLARATION(                                            \
        BOOST_PP_CAT(NAME, UpdateGroupType)::CurrentStepAction,                 \
        BOOST_PP_CAT(BOOST_PP_CAT(NAME, UpdateGroup), CurrentStepAction)        \
    );                                                                          \
    HPX_REGISTER_ACTION_DECLARATION(                                            \
        BOOST_PP_CAT(NAME, UpdateGroupType)::NanoStepAction,                    \
        BOOST_PP_CAT(BOOST_PP_CAT(NAME, UpdateGroup), NanoStepAction)           \
    );                                                                          \
    HPX_REGISTER_ACTION_DECLARATION(                                            \
        BOOST_PP_CAT(NAME, UpdateGroupType)::StopAction,                        \
        BOOST_PP_CAT(BOOST_PP_CAT(NAME, UpdateGroup), StopAction)               \
    );                                                                          \
    HPX_REGISTER_ACTION_DECLARATION(                                            \
        BOOST_PP_CAT(NAME, UpdateGroupType)::SetOuterGhostZoneAction,           \
        BOOST_PP_CAT(BOOST_PP_CAT(NAME, UpdateGroup), SetOuterGhostZoneAction)  \
    );                                                                          \
    HPX_REGISTER_ACTION_DECLARATION(                                            \
        BOOST_PP_CAT(NAME, UpdateGroupType)::SpeedAction,                       \
        BOOST_PP_CAT(BOOST_PP_CAT(NAME, UpdateGroup), SpeedZoneAction)          \
    );                                                                          \
    HPX_REGISTER_BROADCAST_ACTION_DECLARATION_2(                                \
        BOOST_PP_CAT(NAME, UpdateGroupType)::InitAction,                        \
        BOOST_PP_CAT(BOOST_PP_CAT(NAME, UpdateGroupType), BroadcastInitAction)  \
    );                                                                          \
    HPX_REGISTER_BROADCAST_ACTION_DECLARATION_2(                                \
        BOOST_PP_CAT(NAME, UpdateGroupType)::NanoStepAction,                    \
        BOOST_PP_CAT(BOOST_PP_CAT(NAME, UpdateGroupType), BroadcastNanoStepAction)\
    );                                                                          \
    HPX_REGISTER_BROADCAST_ACTION_DECLARATION_2(                                \
        BOOST_PP_CAT(NAME, UpdateGroupType)::SpeedAction,                       \
        BOOST_PP_CAT(BOOST_PP_CAT(NAME, UpdateGroupType), BroadcastSpeedAction) \
    );                                                                          \
/**/

#define LIBGEODECOMP_REGISTER_HPX_SIMULATOR(SIMULATOR, NAME)                    \
    typedef                                                                     \
        hpx::components::managed_component<                                     \
            BOOST_PP_CAT(NAME, UpdateGroupType)                                 \
        >                                                                       \
        BOOST_PP_CAT(NAME, UpdateGroupComponentType);                           \
    HPX_REGISTER_MINIMAL_COMPONENT_FACTORY(                                     \
        BOOST_PP_CAT(NAME, UpdateGroupComponentType),                           \
        BOOST_PP_CAT(NAME, UpdateGroupComponentType)                            \
    );                                                                          \
    HPX_REGISTER_ACTION(                                                        \
        BOOST_PP_CAT(NAME, UpdateGroupType)::InitAction,                        \
        BOOST_PP_CAT(BOOST_PP_CAT(NAME, UpdateGroup), InitAction)               \
    );                                                                          \
    HPX_REGISTER_ACTION(                                                        \
        BOOST_PP_CAT(NAME, UpdateGroupType)::CurrentStepAction,                 \
        BOOST_PP_CAT(BOOST_PP_CAT(NAME, UpdateGroup), CurrentStepAction)        \
    );                                                                          \
    HPX_REGISTER_ACTION(                                                        \
        BOOST_PP_CAT(NAME, UpdateGroupType)::NanoStepAction,                    \
        BOOST_PP_CAT(BOOST_PP_CAT(NAME, UpdateGroup), NanoStepAction)           \
    );                                                                          \
    HPX_REGISTER_ACTION(                                                        \
        BOOST_PP_CAT(NAME, UpdateGroupType)::StopAction,                        \
        BOOST_PP_CAT(BOOST_PP_CAT(NAME, UpdateGroup), StopAction)               \
    );                                                                          \
    HPX_REGISTER_ACTION(                                                        \
        BOOST_PP_CAT(NAME, UpdateGroupType)::SetOuterGhostZoneAction,           \
        BOOST_PP_CAT(BOOST_PP_CAT(NAME, UpdateGroup), SetOuterGhostZoneAction)  \
    );                                                                          \
    HPX_REGISTER_ACTION(                                                        \
        BOOST_PP_CAT(NAME, UpdateGroupType)::SpeedAction,                       \
        BOOST_PP_CAT(BOOST_PP_CAT(NAME, UpdateGroup), SpeedZoneAction)          \
    );                                                                          \
    HPX_REGISTER_BROADCAST_ACTION_2(                                            \
        BOOST_PP_CAT(NAME, UpdateGroupType)::InitAction,                        \
        BOOST_PP_CAT(BOOST_PP_CAT(NAME, UpdateGroupType), BroadcastInitAction)  \
    );                                                                          \
    HPX_REGISTER_BROADCAST_ACTION_2(                                            \
        BOOST_PP_CAT(NAME, UpdateGroupType)::NanoStepAction,                    \
        BOOST_PP_CAT(BOOST_PP_CAT(NAME, UpdateGroupType), BroadcastNanoStepAction)\
    );                                                                          \
    HPX_REGISTER_BROADCAST_ACTION_2(                                            \
        BOOST_PP_CAT(NAME, UpdateGroupType)::SpeedAction,                       \
        BOOST_PP_CAT(BOOST_PP_CAT(NAME, UpdateGroupType), BroadcastSpeedAction) \
    );                                                                          \
/**/

namespace LibGeoDecomp {
namespace HpxSimulator {

typedef std::pair<std::size_t, std::size_t> StepPairType;

template<
    class CELL_TYPE,
    class PARTITION,
    class STEPPER=LibGeoDecomp::HiParSimulator::HpxStepper<CELL_TYPE>
>
class HpxSimulator : public DistributedSimulator<CELL_TYPE>
{
    friend class HpxSimulatorTest;
public:
    typedef typename CELL_TYPE::Topology Topology;
    typedef LibGeoDecomp::DistributedSimulator<CELL_TYPE> ParentType;
    typedef UpdateGroup<CELL_TYPE, PARTITION, STEPPER> UpdateGroupType;
    typedef typename ParentType::GridType GridType;

    static const int DIM = Topology::DIM;

public:
    inline HpxSimulator(
        Initializer<CELL_TYPE> *initializer,
        const float overcommitFactor,
        LoadBalancer *balancer = 0,
        const unsigned loadBalancingPeriod = 1,
        const unsigned ghostZoneWidth = 1) :
        ParentType(initializer),
        balancer(balancer),
        loadBalancingPeriod(loadBalancingPeriod * CELL_TYPE::nanoSteps()),
        ghostZoneWidth(ghostZoneWidth),
        updateGroups(createUpdateGroups<UpdateGroupType>(overcommitFactor)),
        initialized(false)
    {
        updateGroupsIds.reserve(updateGroups.size());
        BOOST_FOREACH(UpdateGroupType& ug, updateGroups) {
            updateGroupsIds.push_back(ug.gid());
        }
    }
    
    inline HpxSimulator(
        Initializer<CELL_TYPE> *initializer,
        const hpx::util::function<std::size_t()>& numUpdateGroups,
        LoadBalancer *balancer = 0,
        const unsigned loadBalancingPeriod = 1,
        const unsigned ghostZoneWidth = 1) :
        ParentType(initializer),
        balancer(balancer),
        loadBalancingPeriod(loadBalancingPeriod * CELL_TYPE::nanoSteps()),
        ghostZoneWidth(ghostZoneWidth),
        updateGroups(createUpdateGroups<UpdateGroupType>(numUpdateGroups)),
        initialized(false)
    {
        updateGroupsIds.reserve(updateGroups.size());
        BOOST_FOREACH(UpdateGroupType& ug, updateGroups) {
            updateGroupsIds.push_back(ug.gid());
        }
    }

    void calculateBoundingBoxes(
        std::vector<CoordBox<DIM> >& boundingBoxes,
        std::size_t rank_start,
        std::size_t rank_end,
        const CoordBox<DIM>& box,
        const SuperVector<long>& weights)
    {
        std::size_t numPartitions = boundingBoxes.size();
        for(std::size_t rank = rank_start; rank != rank_end; ++rank)
        {
            typedef
                HiParSimulator::PartitionManager<DIM, typename CELL_TYPE::Topology>
                PartitionManagerType;

            PartitionManagerType partitionManager;
            
            boost::shared_ptr<PARTITION> partition(
                new PARTITION(
                    box.origin,
                    box.dimensions,
                    0,
                    weights));
            partitionManager.resetRegions(
                    box,
                    partition,
                    rank,
                    ghostZoneWidth
                );
            boundingBoxes[rank] = boost::move(partitionManager.ownRegion().boundingBox());
        }
    }
    
    void init()
    {
        if(initialized) {
            return;
        }
        std::vector<CoordBox<DIM> > boundingBoxes;
        std::size_t numPartitions = updateGroups.size();
        boundingBoxes.resize(numPartitions);

        CoordBox<DIM> box = initializer->gridBox();

        std::vector<hpx::future<void> > boundingBoxesFutures;
        boundingBoxesFutures.reserve(numPartitions);
        std::size_t steps = numPartitions/hpx::get_os_thread_count() + 1;
        
        SuperVector<long> weights(initialWeights(box.dimensions.prod(), numPartitions));
        for(std::size_t i = 0; i < numPartitions; i += steps)
        {
            boundingBoxesFutures.push_back(
                hpx::async(
                    HPX_STD_BIND(
                        &HpxSimulator::calculateBoundingBoxes,
                        this,
                        boost::ref(boundingBoxes),
                        i, (std::min)(numPartitions, i + steps), box,
                        boost::cref(weights)
                    )
                )
            );
        }
        hpx::wait_all(boundingBoxesFutures);
        for(std::size_t i = 0; i < boundingBoxes.size(); ++i)
        {
            std::cerr << "boundingBox " << i << ": " << boundingBoxes[i] << "\n";
        }
        typename UpdateGroupType::InitData initData =
        {
            updateGroups,
            loadBalancingPeriod,
            ghostZoneWidth,
            initializer,
            writers,
            steerers,
            boundingBoxes,
            weights
        };
        hpx::wait(
            hpx::lcos::broadcast<typename UpdateGroupType::ComponentType::InitAction>(
                updateGroupsIds,
                initData
            )
        );

        initialized = true;
    }

    inline void run()
    {
        runTimed();
    }

    inline std::vector<Statistics> runTimed()
    {
        init();
        std::size_t lastNanoStep = initializer->maxSteps() * CELL_TYPE::nanoSteps();
        return nanoStep(lastNanoStep);
    }

    void stop()
    {
        BOOST_FOREACH(UpdateGroupType& ug, updateGroups) {
            ug.stop();
        }
    }

    inline void step()
    {
        init();
        nanoStep(CELL_TYPE::nanoSteps());
    }

    virtual unsigned getStep() const
    {
        if (initialized) {
            return updateGroups[0].currentStep().first;
        } else {
            return initializer->startStep();
        }
        return 0;
    }

    virtual void addSteerer(Steerer<CELL_TYPE> *steerer)
    {
        DistributedSimulator<CELL_TYPE>::addSteerer(steerer);
    }

    virtual void addWriter(ParallelWriter<CELL_TYPE> *writer)
    {
        DistributedSimulator<CELL_TYPE>::addWriter(writer);
    }

    std::size_t numUpdateGroups() const
    {
        return updateGroups.size();
    }

private:
    using DistributedSimulator<CELL_TYPE>::initializer;
    using DistributedSimulator<CELL_TYPE>::steerers;
    using DistributedSimulator<CELL_TYPE>::writers;

    boost::shared_ptr<LoadBalancer> balancer;
    unsigned loadBalancingPeriod;
    unsigned ghostZoneWidth;
    HiParSimulator::PartitionManager<DIM, Topology> partitionManager;
    std::vector<UpdateGroupType> updateGroups;
    std::vector<hpx::id_type> updateGroupsIds;
    boost::atomic<bool> initialized;

    std::vector<Statistics> nanoStep(std::size_t remainingNanoSteps)
    {
        return
            hpx::lcos::broadcast<typename UpdateGroupType::ComponentType::NanoStepAction>(
                updateGroupsIds,
                remainingNanoSteps
            ).move();
    }

    SuperVector<long> initialWeights(const long items, const long size) const
    {
        SuperVector<double> speeds(
            hpx::lcos::broadcast<typename UpdateGroupType::ComponentType::SpeedAction>(
                updateGroupsIds
            ).move());
        double sum = speeds.sum();
        SuperVector<long> ret(size);

        long lastPos = 0;
        double partialSum = 0.0;
        for (long i = 0; i < size -1; i++) {
            partialSum += speeds[i];
            long nextPos = items * partialSum / sum;
            std::cerr << "i: " << i << " " << nextPos << "\n";
            ret[i] = nextPos - lastPos;
            lastPos = nextPos;
        }
        std::cerr << "i: 1023 " << items << "\n";
        ret[size-1] = items - lastPos;

        std::cerr << size << " " << speeds.size() << "\n";
        std::cerr << speeds << "\n";
        std::cerr << ret << "\n";
        std::cerr << ret.sum() << "\n";
        std::cerr << items << "\n";

        return ret;
    }
};

}
}
HPX_REGISTER_BASE_LCO_WITH_VALUE_DECLARATION(
    LibGeoDecomp::HpxSimulator::StepPairType,
    LibGeoDecomp_BaseLcoStepPair
)

HPX_REGISTER_BASE_LCO_WITH_VALUE_DECLARATION(
    LibGeoDecomp::CoordBox<2>,
    LibGeoDecomp_BaseLcoCoordBox2
)

HPX_REGISTER_BASE_LCO_WITH_VALUE_DECLARATION(
    std::vector<double>,
    LibGeoDecomp_BaseLcovector_double
)

HPX_REGISTER_BASE_LCO_WITH_VALUE_DECLARATION(
    std::vector<LibGeoDecomp::Statistics>,
    LibGeoDecomp_BaseLcovector_Statistics
)

#endif
#endif
