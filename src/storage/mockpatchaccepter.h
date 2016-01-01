#ifndef LIBGEODECOMP_STORAGE_MOCKPATCHACCEPTER_H
#define LIBGEODECOMP_STORAGE_MOCKPATCHACCEPTER_H

#include <deque>
#include <libgeodecomp/storage/patchaccepter.h>

namespace LibGeoDecomp {

/**
 * This implementation of the PatchAccepter can record events, which
 * is useful for testing and debugging.
 */
template<class GRID_TYPE>
class MockPatchAccepter : public PatchAccepter<GRID_TYPE>
{
public:
    const static int DIM = GRID_TYPE::DIM;

    virtual void put(
        const GRID_TYPE& /*grid*/,
        const Region<DIM>& /*validRegion*/,
        const Coord<DIM>& globalGridDimensions,
        const std::size_t nanoStep,
        const std::size_t rank)
    {
        offeredNanoSteps.push_back(nanoStep);
        requestedNanoSteps.pop_front();
    }

    virtual std::size_t nextRequiredNanoStep() const
    {
        if (requestedNanoSteps.empty()) {
            return -1;
        }
        return requestedNanoSteps.front();
    }

    void pushRequest(const std::size_t nanoStep)
    {
        requestedNanoSteps.push_back(nanoStep);
    }

    const std::deque<std::size_t>& getRequestedNanoSteps() const
    {
        return requestedNanoSteps;
    }

    const std::deque<std::size_t>& getOfferedNanoSteps() const
    {
        return offeredNanoSteps;
    }

private:
    std::deque<std::size_t> requestedNanoSteps;
    std::deque<std::size_t> offeredNanoSteps;
};

}

#endif
