#include <libgeodecomp/config.h>
#ifdef LIBGEODECOMP_FEATURE_HPX
#ifndef LIBGEODECOMP_PARALLELIZATION_HPXSIMULATOR_H
#define LIBGEODECOMP_PARALLELIZATION_HPXSIMULATOR_H

#include <libgeodecomp/loadbalancer/loadbalancer.h>
#include <libgeodecomp/misc/supermap.h>
#include <libgeodecomp/parallelization/distributedsimulator.h>
#include <libgeodecomp/parallelization/hiparsimulator/partitions/stripingpartition.h>
#include <libgeodecomp/parallelization/hiparsimulator/vanillastepper.h>
#include <libgeodecomp/parallelization/hpxsimulator/updategroup.h>
#include <libgeodecomp/parallelization/hpxsimulator/createupdategroups.h>

#include <boost/serialization/shared_ptr.hpp>

#define LIBGEODECOMP_REGISTER_HPX_SIMULATOR_DECLARATION(SIMULATOR, NAME)         \
    typedef                                                                     \
        SIMULATOR ::UpdateGroupType::ComponentType                              \
        BOOST_PP_CAT(NAME, UpdateGroupType);                                    \
    HPX_REGISTER_ACTION_DECLARATION(                                            \
        BOOST_PP_CAT(NAME, UpdateGroupType)::InitAction,                        \
        BOOST_PP_CAT(BOOST_PP_CAT(NAME, UpdateGroup), InitAction)               \
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
        BOOST_PP_CAT(NAME, UpdateGroupType)::BoundingBoxAction,                 \
        BOOST_PP_CAT(BOOST_PP_CAT(NAME, UpdateGroup), BoundingBoxAction)        \
    );                                                                          \
    HPX_REGISTER_ACTION_DECLARATION(                                            \
        BOOST_PP_CAT(NAME, UpdateGroupType)::SetOuterGhostZoneAction,           \
        BOOST_PP_CAT(BOOST_PP_CAT(NAME, UpdateGroup), SetOuterGhostZoneAction)  \
    );                                                                          \
/**/

#define LIBGEODECOMP_REGISTER_HPX_SIMULATOR(SIMULATOR, NAME)                     \
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
        BOOST_PP_CAT(NAME, UpdateGroupType)::BoundingBoxAction,                 \
        BOOST_PP_CAT(BOOST_PP_CAT(NAME, UpdateGroup), BoundingBoxAction)        \
    );                                                                          \
    HPX_REGISTER_ACTION(                                                        \
        BOOST_PP_CAT(NAME, UpdateGroupType)::SetOuterGhostZoneAction,           \
        BOOST_PP_CAT(BOOST_PP_CAT(NAME, UpdateGroup), SetOuterGhostZoneAction)  \
    );                                                                          \
/**/

namespace LibGeoDecomp {
namespace HpxSimulator {

typedef std::pair<std::size_t, std::size_t> StepPairType;

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
    typedef UpdateGroup<CELL_TYPE, PARTITION, STEPPER> UpdateGroupType;
    typedef typename ParentType::GridType GridType;

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
        ghostZoneWidth(ghostZoneWidth),
        updateGroups(createUpdateGroups<UpdateGroupType>(overcommitFactor)),
        initialized(false)
    {}

    inline void run()
    {
        initSimulation();
        std::size_t lastNanoStep = initializer->maxSteps() * CELL_TYPE::nanoSteps();
        nanoStep(lastNanoStep);
    }

    inline void step()
    {
        initSimulation();
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

    std::size_t overcommitFactor;
    boost::shared_ptr<LoadBalancer> balancer;
    unsigned loadBalancingPeriod;
    unsigned ghostZoneWidth;
    HiParSimulator::PartitionManager<DIM, Topology> partitionManager;
    std::vector<UpdateGroupType> updateGroups;
    boost::atomic<bool> initialized;

    void nanoStep(std::size_t remainingNanoSteps)
    {
        std::vector<hpx::future<void> >
            nanoSteps;
        nanoSteps.reserve(updateGroups.size());

        BOOST_FOREACH(UpdateGroupType & ug, updateGroups) {
            nanoSteps.push_back(ug.nanoStep(remainingNanoSteps));
        }
        hpx::wait(nanoSteps);
    }

    void initSimulation()
    {
        if(initialized) {
            return;
        }

        // TODO: replace with proper broadcast
        BOOST_FOREACH(UpdateGroupType & ug, updateGroups) {
            ug.init(
                updateGroups,
                //balancer,
                loadBalancingPeriod,
                ghostZoneWidth,
                initializer,
                writers,
                steerers
            );
        }
        initialized = true;
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
