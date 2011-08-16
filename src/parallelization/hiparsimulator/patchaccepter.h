#include <libgeodecomp/config.h>
#ifdef LIBGEODECOMP_FEATURE_MPI
#ifndef _libgeodecomp_parallelization_hiparsimulator_patchaccepter_h_
#define _libgeodecomp_parallelization_hiparsimulator_patchaccepter_h_

#include <libgeodecomp/misc/region.h>
#include <libgeodecomp/parallelization/hiparsimulator/gridvecconv.h>

namespace LibGeoDecomp {
namespace HiParSimulator {

template<class GRID_TYPE>
class PatchAccepter
{
public:
    const static int DIM = GRID_TYPE::DIM;

    virtual ~PatchAccepter() {};

    virtual void put(
        const GRID_TYPE& grid, 
        const Region<DIM>& validRegion, 
        const unsigned& nanoStep) = 0;

    virtual long nextRequiredNanoStep() = 0;
};

}
}

#endif
#endif
