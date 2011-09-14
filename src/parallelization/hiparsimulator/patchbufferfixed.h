#include <libgeodecomp/config.h>
#ifdef LIBGEODECOMP_FEATURE_MPI
#ifndef _libgeodecomp_parallelization_hiparsimulator_patchbufferfixed_h_
#define _libgeodecomp_parallelization_hiparsimulator_patchbufferfixed_h_

#include <libgeodecomp/misc/supervector.h>
#include <libgeodecomp/parallelization/hiparsimulator/patchaccepter.h>
#include <libgeodecomp/parallelization/hiparsimulator/patchprovider.h>

namespace LibGeoDecomp {
namespace HiParSimulator {

/**
 * The PatchBuffer's cousin. Can only store a fixed number of regions
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

    PatchBufferFixed(const Region<DIM>& _region=Region<DIM>()) :
        region(_region),
        indexRead(0),
        indexWrite(0),
        buffer(SIZE, SuperVector<CellType>(_region.size()))
    {}

    virtual void put(
        const GRID_TYPE1& grid, 
        const Region<DIM>& /*validRegion*/, 
        const long& nanoStep) 
    {
        // It would be nice to check if validRegion was actually a
        // superset of the region we'll save, but that would be
        // prohibitively slow.
        if (!this->checkNanoStepPut(nanoStep))
            return;
        if (this->storedNanoSteps.size() >= SIZE)
            throw std::logic_error("PatchBufferFixed capacity exceeded.");

        GridVecConv::gridToVector(grid, &buffer[indexWrite], region);
        this->storedNanoSteps << this->requestedNanoSteps.min();
        this->requestedNanoSteps.erase_min();
        inc(&indexWrite);
    }

    virtual void get(
        GRID_TYPE2 *destinationGrid, 
        const Region<DIM>& patchableRegion, 
        const long& nanoStep,
        const bool& remove=true) 
    {
        this->checkNanoStepGet(nanoStep);

        GridVecConv::vectorToGrid(
            buffer[indexRead], destinationGrid, region);

        if (remove) {
            this->storedNanoSteps.erase_min();
            inc(&indexRead);
        }
    }

private:
    Region<DIM> region;
    int indexRead;
    int indexWrite;
    SuperVector<SuperVector<CellType> > buffer;

    inline void inc(int *index)
    {
        *index = (*index + 1) % SIZE;
    }
};

}
}

#endif
#endif
