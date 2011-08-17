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

    // fixme: use pointer to destiantion grid here!
    virtual void get(
        GRID_TYPE& destinationGrid, 
        const Region<DIM>& patchableRegion, 
        const long& nanoStep,
        const bool& remove=true) =0;

protected:
    std::deque<long> storedNanoSteps;

    void checkNanoStepGet(const long& nanoStep) const
    {
        if (storedNanoSteps.empty())
            throw std::logic_error("no nano step available");
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
