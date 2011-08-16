#include <libgeodecomp/config.h>
#ifdef LIBGEODECOMP_FEATURE_MPI
#ifndef _libgeodecomp_parallelization_hiparsimulator_patchprovider_h_
#define _libgeodecomp_parallelization_hiparsimulator_patchprovider_h_

#include <libgeodecomp/misc/region.h>
#include <libgeodecomp/parallelization/hiparsimulator/gridvecconv.h>

namespace LibGeoDecomp {
namespace HiParSimulator {

template<class GRID_TYPE>
class PatchProvider
{
public:
    const static int DIM = GRID_TYPE::DIM;

    virtual ~PatchProvider() {};

    virtual void get(
        GRID_TYPE& destinationGrid, 
        const Region<DIM>& patchRegion, 
        const unsigned& nanoStep) = 0;
};

}
}

#endif
#endif
