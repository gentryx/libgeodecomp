#include <libgeodecomp/config.h>
#ifdef LIBGEODECOMP_FEATURE_MPI
#ifndef _libgeodecomp_parallelization_hiparsimulator_patchaccepter_h_
#define _libgeodecomp_parallelization_hiparsimulator_patchaccepter_h_

#include <libgeodecomp/misc/region.h>
#include <libgeodecomp/misc/superset.h>
#include <libgeodecomp/parallelization/hiparsimulator/gridvecconv.h>

namespace LibGeoDecomp {
namespace HiParSimulator {

template<class GRID_TYPE>
class PatchAccepter
{
public:
    friend class PatchBufferTest;

    const static int DIM = GRID_TYPE::DIM;

    virtual ~PatchAccepter() {};

    virtual void put(
        const GRID_TYPE& grid, 
        const Region<DIM>& validRegion, 
        const long& nanoStep) = 0;

    virtual long nextRequiredNanoStep() const
    {
        if (requestedNanoSteps.empty())
            return -1;
        return *requestedNanoSteps.begin();
    }

    void pushRequest(const long& nanoStep)
    {
        requestedNanoSteps << nanoStep;
    }

protected:
    SuperSet<long> requestedNanoSteps;

    bool checkNanoStepPut(const long& nanoStep) const
    {
        if (requestedNanoSteps.empty() || 
            nanoStep < requestedNanoSteps.min())
            return false;
        if (nanoStep > requestedNanoSteps.min()) {
            std::cout << "got: " << nanoStep << " expected " << requestedNanoSteps.min() << "\n";
            throw std::logic_error("expected nano step was left out");
        }
        return true;
    }
};

}
}

#endif
#endif
