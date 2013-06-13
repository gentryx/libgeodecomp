#ifndef LIBGEODECOMP_PARALLELIZATION_HIPARSIMULATOR_MOCKPATCHACCEPTER_H
#define LIBGEODECOMP_PARALLELIZATION_HIPARSIMULATOR_MOCKPATCHACCEPTER_H

#include <deque>
#include <libgeodecomp/parallelization/hiparsimulator/patchaccepter.h>

namespace LibGeoDecomp {
namespace HiParSimulator {

template<class GRID_TYPE>
class MockPatchAccepter : public PatchAccepter<GRID_TYPE>
{
    friend class VanillaStepperBasicTest;
public:
    const static int DIM = GRID_TYPE::DIM;

    virtual void put(
        const GRID_TYPE& /*grid*/,
        const Region<DIM>& /*validRegion*/,
        const long& nanoStep)
    {
        offeredNanoSteps.push_back(nanoStep);
        requestedNanoSteps.pop_front();
    }

    virtual long nextRequiredNanoStep() const
    {
        if (requestedNanoSteps.empty()) {
            return -1;
        }
        return requestedNanoSteps.front();
    }

    void pushRequest(const long& nanoStep)
    {
        requestedNanoSteps.push_back(nanoStep);
    }

    const std::deque<long>& getRequestedNanoSteps() const
    {
        return requestedNanoSteps;
    }

    const std::deque<long>& getOfferedNanoSteps() const
    {
        return offeredNanoSteps;
    }

private:
    std::deque<long> requestedNanoSteps;
    std::deque<long> offeredNanoSteps;
};

}
}

#endif
