#ifndef LIBGEODECOMP_STORAGE_PATCHACCEPTER_H
#define LIBGEODECOMP_STORAGE_PATCHACCEPTER_H

#include <limits>

#include <libgeodecomp/geometry/region.h>
#include <libgeodecomp/misc/stdcontaineroverloads.h>
#include <libgeodecomp/storage/serializationbuffer.h>

#ifdef LIBGEODECOMP_WITH_HPX
#include <hpx/include/async.hpp>
#endif

namespace LibGeoDecomp {

/**
 * The PatchAccepter collects grid snippets from steppers, either for
 * IO or for ghostzone synchronization.
 */
template<class GRID_TYPE>
class PatchAccepter
{
public:
    friend class PatchBufferTest;
    const static int DIM = GRID_TYPE::DIM;

    static inline std::size_t infinity()
    {
        return std::numeric_limits<std::size_t>::max();
    }

    virtual ~PatchAccepter()
    {}

    virtual void put(
        const GRID_TYPE& grid,
        const Region<DIM>& validRegion,
        const Coord<DIM>& globalGridDimensions,
        const std::size_t nanoStep,
        const std::size_t rank) = 0;

#ifdef LIBGEODECOMP_WITH_HPX
    virtual hpx::future<void> putAsync(
        const GRID_TYPE& grid,
        const Region<DIM>& validRegion,
        const Coord<DIM>& globalGridDimensions,
        const std::size_t nanoStep,
        const std::size_t rank)
    {
        return hpx::async(&PatchAccepter::put, this,
            std::cref(grid), std::cref(validRegion),
            std::cref(globalGridDimensions), nanoStep, rank);
    }
#endif

    virtual void setRegion(const Region<DIM>& region)
    {
        // empty as most implementations won't need it anyway.
    }

    virtual std::size_t nextRequiredNanoStep() const
    {
        if (requestedNanoSteps.empty()) {
            return infinity();
        }

        return *requestedNanoSteps.begin();
    }

    void pushRequest(const std::size_t nanoStep)
    {
        requestedNanoSteps << nanoStep;
    }

protected:
    std::set<std::size_t> requestedNanoSteps;

    bool checkNanoStepPut(const std::size_t nanoStep) const
    {
        if (requestedNanoSteps.empty() ||
            nanoStep < (min)(requestedNanoSteps))
            return false;
        if (nanoStep > (min)(requestedNanoSteps)) {
            std::cerr << "got: " << nanoStep << " expected " << (min)(requestedNanoSteps) << "\n";
            throw std::logic_error("expected nano step was left out");
        }

        return true;
    }
};

}

#endif
