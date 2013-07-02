#include <libgeodecomp/config.h>
#ifdef LIBGEODECOMP_FEATURE_HPX
#ifndef LIBGEODECOMP_PARALLELIZATION_HPXSIMULATOR_H
#define LIBGEODECOMP_PARALLELIZATION_HPXSIMULATOR_H

#include <libgeodecomp/loadbalancer/loadbalancer.h>
#include <libgeodecomp/misc/supermap.h>
#include <libgeodecomp/parallelization/distributedsimulator.h>
#include <libgeodecomp/parallelization/hiparsimulator/partitions/stripingpartition.h>
#include <libgeodecomp/parallelization/hiparsimulator/vanillastepper.h>
#include <libgeodecomp/parallelization/hpxsimulator/hpxupdategroup.h>
#include <libgeodecomp/parallelization/hpxsimulator/createupdategroups.h>

#include <boost/serialization/shared_ptr.hpp>

#define LIBGEDECOMP_REGISTER_HPX_SIMULATOR(CELL_TYPE, PARTITION, TYPE)          \
    typedef                                                                     \
        LibGeoDecomp::HpxSimulator::Server::HpxUpdateGroup<                     \
            CELL_TYPE,                                                          \
            PARTITION,                                                          \
            LibGeoDecomp::HiParSimulator::VanillaStepper<CELL_TYPE>             \
        >                                                                       \
        BOOST_PP_CAT(HpxUpdateGroup, TYPE);                                     \
    HPX_REGISTER_ACTION_DECLARATION(                                            \
        BOOST_PP_CAT(HpxUpdateGroup, TYPE)::InitAction,                         \
        BOOST_PP_CAT(BOOST_PP_CAT(HpxUpdateGroup, TYPE), _InitAction)           \
    );                                                                          \
    HPX_REGISTER_ACTION_DECLARATION(                                            \
        BOOST_PP_CAT(HpxUpdateGroup, TYPE)::CurrentStepAction,                  \
        BOOST_PP_CAT(BOOST_PP_CAT(HpxUpdateGroup, TYPE), _CurrentStepAction)    \
    );                                                                          \
    HPX_REGISTER_ACTION_DECLARATION(                                            \
        BOOST_PP_CAT(HpxUpdateGroup, TYPE)::NanoStepAction,                     \
        BOOST_PP_CAT(BOOST_PP_CAT(HpxUpdateGroup, TYPE), _NanoStepAction)       \
    );                                                                          \
    HPX_REGISTER_ACTION_DECLARATION(                                            \
        BOOST_PP_CAT(HpxUpdateGroup, TYPE)::BoundingBoxAction,                  \
        BOOST_PP_CAT(BOOST_PP_CAT(HpxUpdateGroup, TYPE), _BoundingBoxAction)    \
    );                                                                          \
    HPX_REGISTER_ACTION_DECLARATION(                                            \
        BOOST_PP_CAT(HpxUpdateGroup, TYPE)::SetOuterGhostZoneAction,            \
        BOOST_PP_CAT(BOOST_PP_CAT(HpxUpdateGroup, TYPE), _SetOuterGhostZoneAction)\
    );                                                                          \
                                                                                \
    typedef                                                                     \
        hpx::components::managed_component<                                     \
            BOOST_PP_CAT(HpxUpdateGroup, TYPE)                                  \
        >                                                                       \
        BOOST_PP_CAT(HpxUpdateGroupComponent, TYPE);                            \
    HPX_REGISTER_MINIMAL_COMPONENT_FACTORY(                                     \
        BOOST_PP_CAT(HpxUpdateGroupComponent, TYPE),                            \
        BOOST_PP_CAT(HpxUpdateGroupComponent, TYPE)                             \
    );                                                                          \
    HPX_REGISTER_ACTION(                                                        \
        BOOST_PP_CAT(HpxUpdateGroup, TYPE)::InitAction,                         \
        BOOST_PP_CAT(BOOST_PP_CAT(HpxUpdateGroup, TYPE), _InitAction)           \
    );                                                                          \
    HPX_REGISTER_ACTION(                                                        \
        BOOST_PP_CAT(HpxUpdateGroup, TYPE)::CurrentStepAction,                  \
        BOOST_PP_CAT(BOOST_PP_CAT(HpxUpdateGroup, TYPE), _CurrentStepAction)    \
    );                                                                          \
    HPX_REGISTER_ACTION(                                                        \
        BOOST_PP_CAT(HpxUpdateGroup, TYPE)::NanoStepAction,                     \
        BOOST_PP_CAT(BOOST_PP_CAT(HpxUpdateGroup, TYPE), _NanoStepAction)       \
    );                                                                          \
    HPX_REGISTER_ACTION(                                                        \
        BOOST_PP_CAT(HpxUpdateGroup, TYPE)::BoundingBoxAction,                  \
        BOOST_PP_CAT(BOOST_PP_CAT(HpxUpdateGroup, TYPE), _BoundingBoxAction)    \
    );                                                                          \
    HPX_REGISTER_ACTION(                                                        \
        BOOST_PP_CAT(HpxUpdateGroup, TYPE)::SetOuterGhostZoneAction,            \
        BOOST_PP_CAT(BOOST_PP_CAT(HpxUpdateGroup, TYPE), _SetOuterGhostZoneAction)\
    );                                                                          \
    typedef                                                                     \
        LibGeoDecomp::HpxSimulator::HpxSimulator<CELL_TYPE, PARTITION >         \
        TYPE;                                                                   \
/**/

namespace LibGeoDecomp {
namespace HpxSimulator {
    
typedef std::pair<std::size_t, std::size_t> StepPairType;

enum EventPoint {LOAD_BALANCING, END};
typedef SuperSet<EventPoint> EventSet;
typedef SuperMap<long, EventSet> EventMap;

inline std::string eventToStr(const EventPoint& event)
{
    switch (event) {
    case LOAD_BALANCING:
        return "LOAD_BALANCING";
    case END:
        return "END";
    default:
        return "invalid";
    }
}

template<
    class CELL_TYPE,
    class PARTITION,
    class STEPPER=LibGeoDecomp::HiParSimulator::VanillaStepper<CELL_TYPE>
>
class HpxSimulator : public DistributedSimulator<CELL_TYPE>
{
    friend class HpxSimulatorTest;
public:
    typedef typename CELL_TYPE::Topology Topology;
    typedef LibGeoDecomp::DistributedSimulator<CELL_TYPE> ParentType;
    typedef HpxUpdateGroup<CELL_TYPE, PARTITION, STEPPER> UpdateGroupType;
    typedef typename ParentType::GridType GridType;
    /*
    typedef
        LibGeoDecomp::HiParSimulator::ParallelWriterAdapter<
            typename UpdateGroupType::GridType,
            CELL_TYPE,
            HpxSimulator
        > ParallelWriterAdapterType;
    typedef
        LibGeoDecomp::HiParSimulator::SteererAdapter<
            typename UpdateGroupType::GridType,
            CELL_TYPE
        >
        SteererAdapterType;
    */

    static const int DIM = Topology::DIM;

public:
    inline HpxSimulator(
        Initializer<CELL_TYPE> *initializer,
        const std::size_t overcommitFactor,
        LoadBalancer *balancer = 0,
        const unsigned loadBalancingPeriod = 1,
        const unsigned ghostZoneWidth = 1) :
        ParentType(initializer),
        overcommitFactor(overcommitFactor),
        balancer(balancer),
        loadBalancingPeriod(loadBalancingPeriod * CELL_TYPE::nanoSteps()),
        ghostZoneWidth(ghostZoneWidth)
    {}

    inline void run()
    {
        initSimulation();
        nanoStep(timeToLastEvent());
    }

    inline void step()
    {
        initSimulation();
        nanoStep(CELL_TYPE::nanoSteps());
    }

    virtual unsigned getStep() const
    {
        if (!updateGroups.empty()) {
            return updateGroups[0].currentStep().first;
        } else {
            return initializer->startStep();
        }
        return 0;
    }

    virtual void addSteerer(Steerer<CELL_TYPE> *steerer)
    {
        DistributedSimulator<CELL_TYPE>::addSteerer(steerer);

        /*
        // two adapters needed, just as for the writers
        typename UpdateGroupType::PatchProviderPtr adapterGhost(
            new SteererAdapterType(steererPointer));
        typename UpdateGroupType::PatchProviderPtr adapterInnerSet(
            new SteererAdapterType(steererPointer));

        steererAdaptersGhost.push_back(adapterGhost);
        steererAdaptersInner.push_back(adapterInnerSet);
        */
    }

    virtual void addWriter(ParallelWriter<CELL_TYPE> *writer)
    {
        DistributedSimulator<CELL_TYPE>::addWriter(writer);

        // we need two adapters as each ParallelWriter needs to be
        // notified twice: once for the (inner) ghost zone, and once
        // for the inner set.
        /*
        typename UpdateGroupType::PatchAccepterPtr adapterGhost(
            new ParallelWriterAdapterType(
                this,
                writers.back(),
                initializer->startStep(),
                initializer->maxSteps(),
                initializer->gridDimensions(),
                false));
        typename UpdateGroupType::PatchAccepterPtr adapterInnerSet(
            new ParallelWriterAdapterType(
                this,
                writers.back(),
                initializer->startStep(),
                initializer->maxSteps(),
                initializer->gridDimensions(),
                true));

        writerAdaptersGhost.push_back(adapterGhost);
        writerAdaptersInner.push_back(adapterInnerSet);
        */
    }

private:
    using DistributedSimulator<CELL_TYPE>::initializer;
    using DistributedSimulator<CELL_TYPE>::steerers;
    using DistributedSimulator<CELL_TYPE>::writers;

    std::size_t overcommitFactor;
    boost::shared_ptr<LoadBalancer> balancer;
    unsigned loadBalancingPeriod;
    unsigned ghostZoneWidth;
    EventMap events;
    HiParSimulator::PartitionManager<DIM, Topology> partitionManager;
    std::vector<UpdateGroupType> updateGroups;
    /*
    typename UpdateGroupType::PatchProviderVec steererAdaptersGhost;
    typename UpdateGroupType::PatchProviderVec steererAdaptersInner;
    typename UpdateGroupType::PatchAccepterVec writerAdaptersGhost;
    typename UpdateGroupType::PatchAccepterVec writerAdaptersInner;
    */

    void nanoStep(long remainingNanoSteps)
    {
        std::vector<hpx::future<void> >
            nanoSteps;
        nanoSteps.reserve(updateGroups.size());

        BOOST_FOREACH(UpdateGroupType & ug, updateGroups)
        {
            nanoSteps.push_back(ug.nanoStep(remainingNanoSteps));
        }
        hpx::wait(nanoSteps);
    }

    void initSimulation()
    {
        if(!updateGroups.empty())
        {
            return;
        }

        updateGroups
            = createUpdateGroups<UpdateGroupType>(
                overcommitFactor,
                ghostZoneWidth,
                initializer,
                writers,
                steerers
            );

        initEvents();
    }

    void initEvents()
    {
        events.clear();
        long lastNanoStep = initializer->maxSteps() * CELL_TYPE::nanoSteps();
        events[lastNanoStep] << END;
    }

    long currentNanoStep() const
    {
        std::pair<long, long> now = updateGroups[0].currentStep();
        return now.first * CELL_TYPE::nanoSteps() + now.second;
    }

    long timeToNextEvent()
    {
        return events.begin()->first - currentNanoStep();
    }

    long timeToLastEvent()
    {
        return events.rbegin()->first - currentNanoStep();
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

#endif
#endif
