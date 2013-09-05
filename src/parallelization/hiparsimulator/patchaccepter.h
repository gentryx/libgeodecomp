#ifndef LIBGEODECOMP_PARALLELIZATION_HIPARSIMULATOR_PATCHACCEPTER_H
#define LIBGEODECOMP_PARALLELIZATION_HIPARSIMULATOR_PATCHACCEPTER_H

#include <libgeodecomp/misc/region.h>
#include <libgeodecomp/misc/superset.h>
#include <libgeodecomp/parallelization/hiparsimulator/gridvecconv.h>

namespace LibGeoDecomp {
namespace HiParSimulator {

/**
 * The PatchAccepter collects grid snippets from steppers, either for
 * IO or for ghostzone synchronization.
 */
template<class GRID_TYPE>
class PatchAccepter
{
    friend class PatchBufferTest;
public:
    const static int DIM = GRID_TYPE::DIM;

    virtual ~PatchAccepter()
    {}

    virtual void put(
        const GRID_TYPE& grid,
        const Region<DIM>& validRegion,
        const std::size_t nanoStep) = 0;

    virtual void setRegion(const Region<DIM>& region)
    {
        // empty as most implementations won't need it anyway.
    }

    virtual std::size_t nextRequiredNanoStep() const
    {
        if (requestedNanoSteps.empty()) {
            return -1;
        }

        return *requestedNanoSteps.begin();
    }

    void pushRequest(const std::size_t nanoStep)
    {
        requestedNanoSteps << nanoStep;
    }

protected:
    SuperSet<std::size_t> requestedNanoSteps;

    bool checkNanoStepPut(const std::size_t nanoStep) const
    {
        if (requestedNanoSteps.empty() ||
            nanoStep < (requestedNanoSteps.min)())
            return false;
        if (nanoStep > (requestedNanoSteps.min)()) {
            std::cerr << "got: " << nanoStep << " expected " << (requestedNanoSteps.min)() << "\n";
            throw std::logic_error("expected nano step was left out");
        }

        return true;
    }
};

}
}

#endif
