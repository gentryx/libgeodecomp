#include <libgeodecomp/config.h>
#ifdef LIBGEODECOMP_FEATURE_MPI
#ifndef _libgeodecomp_parallelization_hiparsimulator_patchbuffer_h_
#define _libgeodecomp_parallelization_hiparsimulator_patchbuffer_h_

#include <libgeodecomp/parallelization/hiparsimulator/patchaccepter.h>
#include <libgeodecomp/parallelization/hiparsimulator/patchprovider.h>

namespace LibGeoDecomp {
namespace HiParSimulator {

template<class GRID_TYPE1, class GRID_TYPE2>
class PatchBuffer : 
        public PatchAccepter<GRID_TYPE1>, 
        public PatchProvider<GRID_TYPE2>
{
public:
    typedef typename GRID_TYPE1::CellType CellType;
    const static int DIM = GRID_TYPE1::DIM;

    virtual void put(
        const GRID_TYPE1& grid, 
        const Region<DIM>& /*validRegion*/, 
        const long& nanoStep) 
    {
        // fixme: check whether validRegion is actually a superset of
        // the requested region.
        if (!this->checkNanoStepPut(nanoStep))
            return;

        storedRegions.push_back(
            GridVecConv::gridToVector(grid, region));
        this->storedNanoSteps.push_back(this->requestedNanoSteps.front());
        this->requestedNanoSteps.pop_front();
    }

    // fixme: use pointer to grid here!
    virtual void get(
        GRID_TYPE2& destinationGrid, 
        const Region<DIM>& patchableRegion, 
        const long& nanoStep,
        const bool& remove=true) 
    {
        this->checkNanoStepGet(nanoStep);
        if (storedRegions.empty())
            throw std::logic_error("no region available");

        GridVecConv::vectorToGrid(
            storedRegions.front(), &destinationGrid, region);

        if (remove) {
            storedRegions.pop_front();
            this->storedNanoSteps.pop_front();
        }

    }

    void setRegion(const Region<DIM>& newRegion)
    {
        region = newRegion;
    }

private:
    Region<DIM> region;
    std::deque<SuperVector<CellType> > storedRegions;
};

}
}

#endif
#endif
