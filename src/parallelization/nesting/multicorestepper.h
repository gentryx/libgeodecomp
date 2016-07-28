#ifndef LIBGEODECOMP_PARALLELIZATION_NESTING_MULTICORESTEPPER_H
#define LIBGEODECOMP_PARALLELIZATION_NESTING_MULTICORESTEPPER_H

#include <libgeodecomp/config.h>
#ifdef LIBGEODECOMP_WITH_THREADS

#include <libgeodecomp/parallelization/nesting/commonstepper.h>
#include <libgeodecomp/storage/patchbufferfixed.h>

namespace LibGeoDecomp {

/**
 * MultiCoreStepper is an OpenMP-enabled implementation of the Stepper
 * concept.
 *
 * fixme: how to handle threading if user code has a multithreaded
 *        update() itself? (e.g. n-body codes)
 *
 * fixme: cache blocking?
 *
 * fixme: mpi pacing?
 */
template<typename CELL_TYPE>
class MultiCoreStepper : public CommonStepper<CELL_TYPE>
{
public:
    typedef typename Stepper<CELL_TYPE>::Topology Topology;
    typedef typename Stepper<CELL_TYPE>::InitPtr InitPtr;
    typedef typename Stepper<CELL_TYPE>::PartitionManagerPtr PartitionManagerPtr;
    typedef class Stepper<CELL_TYPE> ParentType;
    typedef typename ParentType::GridType GridType;
    typedef PartitionManager<Topology> PartitionManagerType;
    typedef PatchBufferFixed<GridType, GridType, 1> PatchBufferType1;
    typedef PatchBufferFixed<GridType, GridType, 2> PatchBufferType2;
    typedef typename ParentType::PatchAccepterVec PatchAccepterVec;

    const static int DIM = Topology::DIM;
    const static unsigned NANO_STEPS = APITraits::SelectNanoSteps<CELL_TYPE>::VALUE;

    inline MultiCoreStepper(
        PartitionManagerPtr partitionManager,
        InitPtr initializer,
        const PatchAccepterVec& ghostZonePatchAccepters = PatchAccepterVec(),
        const PatchAccepterVec& innerSetPatchAccepters = PatchAccepterVec()) :
        CommonStepper<CELL_TYPE>(
            partitionManager,
            initializer,
            ghostZonePatchAccepters,
            innerSetPatchAccepters)
    {}
};

}

#endif

#endif
