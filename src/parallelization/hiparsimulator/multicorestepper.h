#ifndef LIBGEODECOMP_PARALLELIZATION_HIPARSIMULATOR_MULTICORESTEPPER_H
#define LIBGEODECOMP_PARALLELIZATION_HIPARSIMULATOR_MULTICORESTEPPER_H

#include <libgeodecomp/parallelization/hiparsimulator/patchbufferfixed.h>
#include <libgeodecomp/parallelization/hiparsimulator/stepper.h>

namespace LibGeoDecomp {
namespace HiParSimulator {

template<typename CELL_TYPE>
class MulticoreStepper 
// fixme: not enabled yet
// : public StepperHelper<
//     DisplacedGrid<CELL_TYPE, typename CELL_TYPE::Topology, true> >
{
};

}
}

#endif
