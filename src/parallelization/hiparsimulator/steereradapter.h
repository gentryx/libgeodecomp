#ifndef LIBGEODECOMP_PARALLELIZATION_HIPARSIMULATOR_STEERERADAPTER_H
#define LIBGEODECOMP_PARALLELIZATION_HIPARSIMULATOR_STEERERADAPTER_H

#include <libgeodecomp/io/steerer.h>
#include <libgeodecomp/parallelization/hiparsimulator/patchprovider.h>

namespace LibGeoDecomp {
namespace HiParSimulator {

template<typename GRID_TYPE, typename CELL_TYPE>
class SteererAdapter : public PatchProvider<GRID_TYPE>
{
public:
    static const int DIM = CELL_TYPE::Topology::DIMENSIONS;

    SteererAdapter(
        boost::shared_ptr<Steerer<CELL_TYPE> > _steerer) :
        steerer(_steerer)
    {}

    virtual void get(
        GRID_TYPE *destinationGrid, 
        const Region<DIM>& patchableRegion, 
        const long& globalNanoStep,
        const bool& remove=true) 
    {
        int nanoStep = globalNanoStep % CELL_TYPE::nanoSteps();
        if (nanoStep != 0) {
            return;
        }

        long step = globalNanoStep / CELL_TYPE::nanoSteps();
        if (step % steerer->getPeriod() != 0) {
            return;
        }

        steerer->nextStep(destinationGrid, patchableRegion, step);
    }
    
private:
    boost::shared_ptr<Steerer<CELL_TYPE> > steerer;

};

}
}

#endif
