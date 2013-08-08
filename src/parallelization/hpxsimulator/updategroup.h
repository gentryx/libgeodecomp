
#include <libgeodecomp/config.h>
#ifdef LIBGEODECOMP_FEATURE_HPX
#ifndef LIBGEODECOMP_PARALLELIZATION_HPXUPDATEGROUP_H
#define LIBGEODECOMP_PARALLELIZATION_HPXUPDATEGROUP_H

#include <libgeodecomp/misc/displacedgrid.h>
#include <libgeodecomp/misc/statistics.h>
#include <libgeodecomp/parallelization/hiparsimulator/partitionmanager.h>
#include <libgeodecomp/parallelization/hiparsimulator/stepper.h>
#include <libgeodecomp/parallelization/hpxsimulator/updategroupserver.h>

#include <hpx/apply.hpp>

namespace LibGeoDecomp {
namespace HpxSimulator {

template <class CELL_TYPE, class PARTITION, class STEPPER>
class UpdateGroup
{
    friend class boost::serialization::access;
    friend class UpdateGroupServer<CELL_TYPE, PARTITION, STEPPER>;
public:
    const static int DIM = CELL_TYPE::Topology::DIM;
    typedef DisplacedGrid<
        CELL_TYPE, typename CELL_TYPE::Topology, true> GridType;
    typedef
        typename DistributedSimulator<CELL_TYPE>::WriterVector
        WriterVector;
    typedef
        typename DistributedSimulator<CELL_TYPE>::SteererVector
        SteererVector;
    typedef typename HiParSimulator::Stepper<CELL_TYPE>::PatchType PatchType;
    typedef
        typename HiParSimulator::Stepper<CELL_TYPE>::PatchProviderPtr
        PatchProviderPtr;
    typedef
        typename HiParSimulator::Stepper<CELL_TYPE>::PatchAccepterPtr
        PatchAccepterPtr;
    typedef
        typename HiParSimulator::Stepper<CELL_TYPE>::PatchAccepterVec
        PatchAccepterVec;
    typedef
        typename HiParSimulator::Stepper<CELL_TYPE>::PatchProviderVec
        PatchProviderVec;

    typedef UpdateGroupServer<CELL_TYPE, PARTITION, STEPPER> ComponentType;

    typedef std::pair<std::size_t, std::size_t> StepPairType;

    UpdateGroup()
    {}

    UpdateGroup(hpx::id_type thisId)
      : thisId(thisId)
    {}

    struct InitData
    {
        std::vector<UpdateGroup> updateGroups;
        unsigned loadBalancingPeriod;
        unsigned ghostZoneWidth;
        boost::shared_ptr<Initializer<CELL_TYPE> > initializer;
        WriterVector writers;
        SteererVector steerers;
        std::vector<CoordBox<DIM> > boundingBoxes;

        template <typename ARCHIVE>
        void serialize(ARCHIVE& ar, unsigned)
        {
            ar & updateGroups;
            ar & loadBalancingPeriod;
            ar & ghostZoneWidth;
            ar & initializer;
            ar & writers;
            ar & steerers;
            ar & boundingBoxes;
        }
    };

    hpx::future<void> init(
        const std::vector<UpdateGroup>& updateGroups,
        //boost::shared_ptr<LoadBalancer> balancer,
        unsigned loadBalancingPeriod,
        unsigned ghostZoneWidth,
        boost::shared_ptr<Initializer<CELL_TYPE> > initializer,
        const WriterVector& writers,
        const SteererVector& steerers,
        std::vector<CoordBox<DIM> > boundingBoxes
    )
    {
        InitData initData =
        {
            updateGroups,
            loadBalancingPeriod,
            ghostzoneWidth,
            initializer,
            writers,
            steerers,
            boundingBoxes
        };
        return
            hpx::async<typename ComponentType::InitAction>(
                thisId,
                boost::move(initData)
            );
    }

    hpx::naming::id_type gid() const
    {
        return thisId;
    }

    StepPairType currentStep() const
    {
        return typename ComponentType::CurrentStepAction()(thisId);
    }

    hpx::future<Statistics> nanoStep(std::size_t remainingNanoSteps)
    {
        return
            hpx::async<typename ComponentType::NanoStepAction>(
                thisId,
                remainingNanoSteps
            );
    }

    void stop() const
    {
        return typename ComponentType::StopAction()(thisId);
    }

    hpx::future<void> setOuterGhostZone(
        std::size_t srcRank,
        boost::shared_ptr<SuperVector<CELL_TYPE> > buffer,
        long nanoStep)
    {
        return
            hpx::async<typename ComponentType::SetOuterGhostZoneAction>(
                thisId,
                srcRank,
                buffer,
                nanoStep
            );
    }


private:
    hpx::naming::id_type thisId;

    template <typename ARCHIVE>
    void serialize(ARCHIVE& ar, unsigned)
    {
        ar & thisId;
    }
};

}}

#endif
#endif
