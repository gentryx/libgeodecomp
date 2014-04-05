#ifndef LIBGEODECOMP_PARALLELIZATION_HIPARSIMULATOR_COMMONSTEPPER_H
#define LIBGEODECOMP_PARALLELIZATION_HIPARSIMULATOR_COMMONSTEPPER_H

#include <libgeodecomp/parallelization/hiparsimulator/stepper.h>
#include <libgeodecomp/storage/patchbufferfixed.h>

namespace LibGeoDecomp {

namespace HiParSimulator {

template<typename CELL_TYPE>
class CommonStepper : public Stepper<CELL_TYPE>
{
public:
    typedef typename Stepper<CELL_TYPE>::Topology Topology;
    const static int DIM = Topology::DIM;
    const static unsigned NANO_STEPS = APITraits::SelectNanoSteps<CELL_TYPE>::VALUE;

    typedef class CommonStepper<CELL_TYPE> ParentType;
    typedef typename ParentType::GridType GridType;
    typedef PartitionManager<Topology> PartitionManagerType;
    typedef PatchBufferFixed<GridType, GridType, 1> PatchBufferType1;
    typedef PatchBufferFixed<GridType, GridType, 2> PatchBufferType2;
    typedef typename ParentType::PatchAccepterVec PatchAccepterVec;

    CommonStepper(
        boost::shared_ptr<PartitionManagerType> partitionManager,
        boost::shared_ptr<Initializer<CELL_TYPE> > initializer// ,
        // const PatchAccepterVec& ghostZonePatchAccepters = PatchAccepterVec(),
        // const PatchAccepterVec& innerSetPatchAccepters = PatchAccepterVec()
                  ) :
        Stepper<CELL_TYPE>(
            partitionManager,
            initializer)
    {}
};

}
}

#endif
