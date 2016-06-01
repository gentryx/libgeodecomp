#ifndef LIBGEODECOMP_PARALLELIZATION_HPXDATAFLOWSIMULATOR_H
#define LIBGEODECOMP_PARALLELIZATION_HPXDATAFLOWSIMULATOR_H

#include <libgeodecomp/config.h>
#ifdef LIBGEODECOMP_WITH_HPX

namespace LibGeoDecomp {

/**
 * Experimental Simulator based on (surprise surprise) HPX' dataflow
 operator. Primary use case (for now) is DGSWEM.
 */
template<typename CELL>
class HPXDataflowSimulator
{
};

}

#endif

#endif
