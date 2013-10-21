#ifndef LIBGEODECOMP_PARALLELIZATION_HIPARSIMULATOR_PATCHBUFFER_H
#define LIBGEODECOMP_PARALLELIZATION_HIPARSIMULATOR_PATCHBUFFER_H

#include <deque>
#include <libgeodecomp/storage/patchaccepter.h>
#include <libgeodecomp/storage/patchprovider.h>

namespace LibGeoDecomp {

template<class GRID_TYPE1, class GRID_TYPE2>
class PatchBuffer :
        public PatchAccepter<GRID_TYPE1>,
        public PatchProvider<GRID_TYPE2>
{
public:
    friend class PatchBufferTest;
    typedef typename GRID_TYPE1::CellType CellType;
    const static int DIM = GRID_TYPE1::DIM;

    using PatchAccepter<GRID_TYPE1>::checkNanoStepPut;
    using PatchAccepter<GRID_TYPE1>::requestedNanoSteps;
    using PatchProvider<GRID_TYPE2>::checkNanoStepGet;
    using PatchProvider<GRID_TYPE2>::storedNanoSteps;

    PatchBuffer(const Region<DIM>& region=Region<DIM>()) :
        region(region)
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

        storedRegions.push_back(
            GridVecConv::gridToVector(grid, region));
        storedNanoSteps << (min)(requestedNanoSteps);
        erase_min(requestedNanoSteps);
    }

    virtual void get(
        GRID_TYPE2 *destinationGrid,
        const Region<DIM>& patchableRegion,
        const std::size_t nanoStep,
        const bool remove=true)
    {
        checkNanoStepGet(nanoStep);
        if (storedRegions.empty()) {
            throw std::logic_error("no region available");
        }

        GridVecConv::vectorToGrid(
            storedRegions.front(), destinationGrid, region);

        if (remove) {
            storedRegions.pop_front();
            erase_min(storedNanoSteps);
        }

    }

private:
    Region<DIM> region;
    std::deque<std::vector<CellType> > storedRegions;
};

}

#endif
