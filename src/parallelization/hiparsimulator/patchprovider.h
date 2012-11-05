#ifndef _libgeodecomp_parallelization_hiparsimulator_patchprovider_h_
#define _libgeodecomp_parallelization_hiparsimulator_patchprovider_h_

#include <libgeodecomp/misc/region.h>
#include <libgeodecomp/misc/superset.h>
#include <libgeodecomp/misc/stringops.h>
#include <libgeodecomp/parallelization/hiparsimulator/gridvecconv.h>

namespace LibGeoDecomp {
namespace HiParSimulator {

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

    virtual ~PatchProvider() {};

    virtual void get(
        GRID_TYPE *destinationGrid, 
        const Region<DIM>& patchableRegion, 
        const long& nanoStep,
        const bool& remove=true) =0;

protected:
    SuperSet<long> storedNanoSteps;

    void checkNanoStepGet(const long& nanoStep) const
    {
        if (storedNanoSteps.empty()) {
            throw std::logic_error("no nano step available");
        }
        if ((storedNanoSteps.min)() != nanoStep) 
            throw std::logic_error(
                std::string(
                    "requested time step doesn't match expected nano step.") 
                + " expected: " + StringConv::itoa((storedNanoSteps.min)()) 
                + " is: " + StringConv::itoa(nanoStep));
    }
};

}
}

#endif
