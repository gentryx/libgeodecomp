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
    static const int DIM = CELL_TYPE::Topology::DIM;

    SteererAdapter(
        boost::shared_ptr<Steerer<CELL_TYPE> > steerer) :
        steerer(steerer)
    {}

    virtual void setRegion(const Region<DIM>& region)
    {
        steerer->setRegion(region);
    }

    virtual void get(
        GRID_TYPE *destinationGrid,
        const Region<DIM>& patchableRegion,
        const std::size_t globalNanoStep,
        const bool remove = true)
    {
        std::size_t nanoStep = globalNanoStep % CELL_TYPE::nanoSteps();
        if (nanoStep != 0) {
            return;
        }

        std::size_t step = globalNanoStep / CELL_TYPE::nanoSteps();
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
