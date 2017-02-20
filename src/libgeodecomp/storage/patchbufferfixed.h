#ifndef LIBGEODECOMP_STORAGE_PATCHBUFFERFIXED_H
#define LIBGEODECOMP_STORAGE_PATCHBUFFERFIXED_H

#include <libgeodecomp/misc/stdcontaineroverloads.h>
#include <libgeodecomp/storage/patchaccepter.h>
#include <libgeodecomp/storage/patchprovider.h>
#include <libgeodecomp/storage/serializationbuffer.h>

namespace LibGeoDecomp {

/**
 * The PatchBuffer's cousin can only store a fixed number of regions
 * at a time, but avoids the memory allocation hassle during
 * put().
 */
template<class GRID_TYPE1, class GRID_TYPE2, int SIZE>
class PatchBufferFixed :
        public PatchAccepter<GRID_TYPE1>,
        public PatchProvider<GRID_TYPE2>
{
public:
    friend class PatchBufferFixedTest;
    typedef typename GRID_TYPE1::CellType CellType;
    typedef typename SerializationBuffer<CellType>::BufferType BufferType;
    const static int DIM = GRID_TYPE1::DIM;

    using PatchAccepter<GRID_TYPE1>::checkNanoStepPut;
    using PatchAccepter<GRID_TYPE1>::requestedNanoSteps;
    using PatchProvider<GRID_TYPE2>::checkNanoStepGet;
    using PatchProvider<GRID_TYPE2>::storedNanoSteps;
    using PatchProvider<GRID_TYPE2>::get;

    explicit PatchBufferFixed(const Region<DIM>& region = Region<DIM>()) :
        region(region),
        indexRead(0),
        indexWrite(0),
        buffer(SIZE, SerializationBuffer<CellType>::create(region))
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
        if (storedNanoSteps.size() >= SIZE) {
            throw std::logic_error("PatchBufferFixed capacity exceeded.");
        }

        grid.saveRegion(&buffer[indexWrite], region);
        storedNanoSteps << (min)(requestedNanoSteps);
        erase_min(requestedNanoSteps);
        inc(&indexWrite);
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

        destinationGrid->loadRegion(buffer[indexRead], region);

        if (remove) {
            erase_min(storedNanoSteps);
            inc(&indexRead);
        }
    }

private:
    Region<DIM> region;
    int indexRead;
    int indexWrite;
    std::vector<BufferType> buffer;

    inline void inc(int *index)
    {
        *index = (*index + 1) % SIZE;
    }
};

}

#endif
