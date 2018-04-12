#ifndef LIBGEODECOMP_STORAGE_PATCHBUFFER_H
#define LIBGEODECOMP_STORAGE_PATCHBUFFER_H

#include <libgeodecomp/storage/patchaccepter.h>
#include <libgeodecomp/storage/patchprovider.h>

// Kill some warnings in system headers
#ifdef _MSC_BUILD
#pragma warning( push )
#pragma warning( disable : 4514 4710 4711 )
#endif

#include <deque>

#ifdef _MSC_BUILD
#pragma warning( pop )
#endif

namespace LibGeoDecomp {

/**
 * The PatchBuffer stores a queue of grid fragments, specified by a
 * Region, for later retrieval. Useful for building Steppers which
 * implement overlapping communication and calculation (and hence need
 * to buffer certain parts of the grid which will be temporarily
 * overwritten).
 */
template<class GRID_TYPE1, class GRID_TYPE2>
class PatchBuffer :
        public PatchAccepter<GRID_TYPE1>,
        public PatchProvider<GRID_TYPE2>
{
public:
    friend class PatchBufferTest;
    typedef typename GRID_TYPE1::CellType CellType;
    typedef typename SerializationBuffer<CellType>::BufferType BufferType;
    const static int DIM = GRID_TYPE1::DIM;

    using PatchAccepter<GRID_TYPE1>::checkNanoStepPut;
    using PatchAccepter<GRID_TYPE1>::requestedNanoSteps;
    using PatchProvider<GRID_TYPE2>::checkNanoStepGet;
    using PatchProvider<GRID_TYPE2>::storedNanoSteps;

    explicit PatchBuffer(const Region<DIM>& region = Region<DIM>()) :
        region(region)
    {}

    virtual void put(
        const GRID_TYPE1& grid,
        const Region<DIM>& /*validRegion*/,
        const Coord<DIM>& globalGridDimensions,
        const std::size_t nanoStep,
        const std::size_t rank)
    {
        // It would be nice to check if validRegion was actually a
        // superset of the region we'll save, but that would be
        // prohibitively slow.
        if (!checkNanoStepPut(nanoStep)) {
            return;
        }

        storedRegions.push_back(BufferType());
        BufferType& buffer = storedRegions.back();
        buffer = SerializationBuffer<CellType>::create(region);

        grid.saveRegion(&buffer, region);
        storedNanoSteps << (min)(requestedNanoSteps);
        erase_min(requestedNanoSteps);
    }

    virtual void get(
        GRID_TYPE2 *destinationGrid,
        const Region<DIM>& patchableRegion,
        const Coord<DIM>& globalGridDimensions,
        const std::size_t nanoStep,
        const std::size_t rank,
        const bool remove=true)
    {
        checkNanoStepGet(nanoStep);
        if (storedRegions.empty()) {
            throw std::logic_error("no region available");
        }

        destinationGrid->loadRegion(storedRegions.front(), region);

        if (remove) {
            storedRegions.pop_front();
            erase_min(storedNanoSteps);
        }

    }

private:
    Region<DIM> region;
    std::deque<BufferType> storedRegions;
};

}

#endif
