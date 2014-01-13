#ifndef LIBGEODECOMP_PARALLELIZATION_HPXSIMULATOR_UPDATEGROUP_H
#define LIBGEODECOMP_PARALLELIZATION_HPXSIMULATOR_UPDATEGROUP_H

#include <libgeodecomp/config.h>
#ifdef LIBGEODECOMP_FEATURE_HPX

#include <libgeodecomp/geometry/partitionmanager.h>
#include <libgeodecomp/parallelization/hiparsimulator/stepper.h>
#include <libgeodecomp/parallelization/hpxsimulator/updategroupserver.h>
#include <libgeodecomp/storage/displacedgrid.h>

#include <hpx/apply.hpp>

namespace LibGeoDecomp {
namespace HpxSimulator {

template <class CELL_TYPE, class PARTITION, class STEPPER>
class UpdateGroup
{
    friend class boost::serialization::access;
    friend class UpdateGroupServer<CELL_TYPE, PARTITION, STEPPER>;
public:
    typedef typename STEPPER::Topology Topology;
    typedef DisplacedGrid<CELL_TYPE, Topology, true> GridType;
    typedef typename STEPPER::PatchType PatchType;
    typedef typename STEPPER::PatchProviderPtr PatchProviderPtr;
    typedef typename STEPPER::PatchAccepterPtr PatchAccepterPtr;
    typedef typename STEPPER::PatchAccepterVec PatchAccepterVec;
    typedef typename STEPPER::PatchProviderVec PatchProviderVec;
    const static int DIM = Topology::DIM;

    typedef
        typename DistributedSimulator<CELL_TYPE>::WriterVector
        WriterVector;
    typedef
        typename DistributedSimulator<CELL_TYPE>::SteererVector
        SteererVector;

    typedef UpdateGroupServer<CELL_TYPE, PARTITION, STEPPER> ComponentType;

    typedef std::pair<std::size_t, std::size_t> StepPairType;

    UpdateGroup()
    {}

    UpdateGroup(hpx::id_type thisId)
      : thisId(thisId)
    {}

    struct InitData
    {
        unsigned loadBalancingPeriod;
        unsigned ghostZoneWidth;
        boost::shared_ptr<Initializer<CELL_TYPE> > initializer;
        WriterVector writers;
        SteererVector steerers;
        std::vector<CoordBox<DIM> > boundingBoxes;
        std::vector<std::size_t> initialWeights;

        template <typename ARCHIVE>
        void serialize(ARCHIVE& ar, unsigned)
        {
            ar & loadBalancingPeriod;
            ar & ghostZoneWidth;
            ar & initializer;
            ar & writers;
            ar & steerers;
            ar & boundingBoxes;
            ar & initialWeights;
        }
    };

    hpx::naming::id_type gid() const
    {
        return thisId;
    }

    hpx::unique_future<void> setOuterGhostZone(
        std::size_t srcRank,
        boost::shared_ptr<std::vector<CELL_TYPE> > buffer,
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
