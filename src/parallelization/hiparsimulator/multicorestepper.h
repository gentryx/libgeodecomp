#ifndef _libgeodecomp_parallelization_hiparsimulator_multicorestepper_h_
#define _libgeodecomp_parallelization_hiparsimulator_multicorestepper_h_

#include <libgeodecomp/misc/displacedgrid.h>
#include <libgeodecomp/parallelization/hiparsimulator/patchbufferfixed.h>
#include <libgeodecomp/parallelization/hiparsimulator/stepperhelper.h>

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
