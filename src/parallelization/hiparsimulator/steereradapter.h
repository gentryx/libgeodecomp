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
    typedef typename APITraits::SelectTopology<CELL_TYPE>::Value Topology;

    static const unsigned NANO_STEPS = APITraits::SelectNanoSteps<CELL_TYPE>::VALUE;
    static const int DIM = Topology::DIM;

    SteererAdapter(
        boost::shared_ptr<Steerer<CELL_TYPE> > steerer,
        const std::size_t firstStep,
        const std::size_t lastStep,
        Coord<Topology::DIM> globalGridDimensions,
        std::size_t rank,
        bool lastCall) :
        steerer(steerer),
        firstNanoStep(firstStep * NANO_STEPS),
        lastNanoStep(lastStep   * NANO_STEPS),
        rank(rank),
        lastCall(lastCall),
        globalGridDimensions(globalGridDimensions)
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
        std::size_t nanoStep = globalNanoStep % NANO_STEPS;
        if (nanoStep != 0) {
            return;
        }

        std::size_t step = globalNanoStep / NANO_STEPS;
        if (step % steerer->getPeriod() != 0) {
            return;
        }

        SteererEvent event = STEERER_NEXT_STEP;
        if (nanoStep == firstNanoStep) {
            event = STEERER_INITIALIZED;
        }
        if (nanoStep == lastNanoStep) {
            event = STEERER_ALL_DONE;
        }

        steerer->nextStep(
            destinationGrid,
            patchableRegion,
            globalGridDimensions,
            step,
            event,
            rank,
            lastCall);
    }

private:
    boost::shared_ptr<Steerer<CELL_TYPE> > steerer;
    std::size_t firstNanoStep;
    std::size_t lastNanoStep;
    std::size_t rank;
    bool lastCall;
    Coord<Topology::DIM> globalGridDimensions;

};

}
}

#endif
