#ifndef LIBGEODECOMP_PARALLELIZATION_HIPARSIMULATOR_PATCHBUFFERFIXED_H
#define LIBGEODECOMP_PARALLELIZATION_HIPARSIMULATOR_PATCHBUFFERFIXED_H

#include <libgeodecomp/misc/supervector.h>
#include <libgeodecomp/parallelization/hiparsimulator/patchaccepter.h>
#include <libgeodecomp/parallelization/hiparsimulator/patchprovider.h>

namespace LibGeoDecomp {
namespace HiParSimulator {

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
    const static int DIM = GRID_TYPE1::DIM;

    using PatchAccepter<GRID_TYPE1>::checkNanoStepPut;
    using PatchAccepter<GRID_TYPE1>::requestedNanoSteps;
    using PatchProvider<GRID_TYPE2>::checkNanoStepGet;
    using PatchProvider<GRID_TYPE2>::storedNanoSteps;

    PatchBufferFixed(const Region<DIM>& region=Region<DIM>()) :
        region(region),
        indexRead(0),
        indexWrite(0),
        buffer(SIZE, std::vector<CellType>(region.size()))
    {}

    virtual void put(
        const GRID_TYPE1& grid,
        const Region<DIM>& /*validRegion*/,
        const std::size_t nanoStep)
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

        GridVecConv::gridToVector(grid, &buffer[indexWrite], region);
        storedNanoSteps << (requestedNanoSteps.min)();
        (requestedNanoSteps.erase_min)();
        inc(&indexWrite);
    }

    virtual void get(
        GRID_TYPE2 *destinationGrid,
        const Region<DIM>& patchableRegion,
        const std::size_t nanoStep,
        const bool remove=true)
    {
        checkNanoStepGet(nanoStep);

        GridVecConv::vectorToGrid(
            buffer[indexRead], destinationGrid, region);

        if (remove) {
            storedNanoSteps.erase_min();
            inc(&indexRead);
        }
    }

private:
    Region<DIM> region;
    int indexRead;
    int indexWrite;
    std::vector<std::vector<CellType> > buffer;

    inline void inc(int *index)
    {
        *index = (*index + 1) % SIZE;
    }
};

}
}

#endif
