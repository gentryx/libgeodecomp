#include <libgeodecomp/config.h>
#ifdef LIBGEODECOMP_FEATURE_MPI
#ifndef _libgeodecomp_parallelization_hiparsimulator_patchprovider_h_
#define _libgeodecomp_parallelization_hiparsimulator_patchprovider_h_

#include <deque>

#include <libgeodecomp/misc/region.h>
#include <libgeodecomp/misc/stringops.h>
#include <libgeodecomp/parallelization/hiparsimulator/gridvecconv.h>

namespace LibGeoDecomp {
namespace HiParSimulator {

template<class GRID_TYPE>
class PatchProvider
{
public:
    typedef typename GRID_TYPE::CellType CellType;
    const static int DIM = GRID_TYPE::DIM;

    virtual ~PatchProvider() {};

    virtual void get(
        GRID_TYPE& destinationGrid, 
        const Region<DIM>& patchableRegion, 
        const long& nanoStep,
        const bool& remove=true) =0;

protected:
    std::deque<long> storedNanoSteps;
    std::deque<SuperVector<CellType> > storedRegions;

    void checkNanoStepGet(const long& nanoStep)
    {
        if (storedRegions.empty())
            throw std::logic_error("no region available");

        if (storedNanoSteps.front() != nanoStep) 
            throw std::logic_error(
                std::string(
                    "requested time step doesn't match expected nano step.") 
                + " expected: " + StringConv::itoa(storedNanoSteps.front()) 
                + " is: " + StringConv::itoa(nanoStep));
    }
};

}
}

#endif
#endif
