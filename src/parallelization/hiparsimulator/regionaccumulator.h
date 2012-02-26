#include <libgeodecomp/config.h>
#ifdef LIBGEODECOMP_FEATURE_MPI
#ifndef _libgeodecomp_parallelization_hiparsimulator_regionaccumulator_h_
#define _libgeodecomp_parallelization_hiparsimulator_regionaccumulator_h_

#include <libgeodecomp/misc/region.h>

namespace LibGeoDecomp {
namespace HiParSimulator {

template<int DIM>
class RegionAccumulator
{
public:
    virtual Region<DIM> getRegion(const long& node) = 0;
};

};
};

#endif
#endif
