#ifndef LIBGEODECOMP_STORAGE_PATCHPROVIDER_H
#define LIBGEODECOMP_STORAGE_PATCHPROVIDER_H

#ifdef LIBGEODECOMP_WITH_HPX
#include <hpx/include/threads.hpp>
#include <hpx/synchronization/spinlock.hpp>
#include <mutex>
#endif

#include <libgeodecomp/geometry/region.h>
#include <libgeodecomp/misc/limits.h>
#include <libgeodecomp/misc/stdcontaineroverloads.h>
#include <libgeodecomp/misc/stringops.h>

namespace LibGeoDecomp {

/**
 * PatchProvider fills in grid patches into a Stepper, which is
 * important for ghostzone synchronization, but could be used for
 * steering, too.
 */
template<class GRID_TYPE>
class PatchProvider
{
public:
    typedef typename GRID_TYPE::CellType CellType;
    const static int DIM = GRID_TYPE::DIM;

    static inline std::size_t infinity()
    {
        return Limits<std::size_t>::getMax();
    }

    virtual ~PatchProvider() {};

    virtual void setRegion(const Region<DIM>& region)
    {
        // empty as most implementations won't need it anyway.
    }

    virtual void get(
        GRID_TYPE *destinationGrid,
        const Region<DIM>& patchableRegion,
        const Coord<DIM>& globalGridDimensions,
        const std::size_t nanoStep,
        const std::size_t rank,
        const bool remove=true) = 0;

#ifdef LIBGEODECOMP_WITH_HPX
    virtual hpx::future<void> getAsync(
        GRID_TYPE *destinationGrid,
        const Region<DIM>& patchableRegion,
        const Coord<DIM>& globalGridDimensions,
        const std::size_t nanoStep,
        const std::size_t rank,
        const bool remove=true)
    {
        return hpx::async(&PatchProvider::get, this,
            destinationGrid, std::cref(patchableRegion),
            std::cref(globalGridDimensions), nanoStep, rank, remove);
    }
#endif

    virtual std::size_t nextAvailableNanoStep() const
    {
        if (storedNanoSteps.empty()) {
            return infinity();
        }

        return *storedNanoSteps.begin();
    }

protected:
    std::set<std::size_t> storedNanoSteps;

    void checkNanoStepGet(const std::size_t nanoStep) const
    {
        if (storedNanoSteps.empty()) {
            throw std::logic_error("no nano step available");
        }

        if ((min)(storedNanoSteps) != nanoStep) {
            throw std::logic_error(
                std::string("requested time step doesn't match expected nano step.") +
                " expected: " + StringOps::itoa((min)(storedNanoSteps)) +
                " is: " + StringOps::itoa(nanoStep));
        }
    }
};

}

#endif
