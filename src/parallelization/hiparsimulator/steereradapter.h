#ifndef LIBGEODECOMP_PARALLELIZATION_HIPARSIMULATOR_STEERERADAPTER_H
#define LIBGEODECOMP_PARALLELIZATION_HIPARSIMULATOR_STEERERADAPTER_H

#include <libgeodecomp/io/steerer.h>
#include <libgeodecomp/storage/patchprovider.h>

namespace LibGeoDecomp {
namespace HiParSimulator {

template<typename GRID_TYPE, typename CELL_TYPE>
class SteererAdapter : public PatchProvider<GRID_TYPE>
{
public:
    typedef typename APITraits::SelectTopology<CELL_TYPE>::Value Topology;

    static const unsigned NANO_STEPS = APITraits::SelectNanoSteps<CELL_TYPE>::VALUE;
    static const int DIM = Topology::DIM;

    using PatchProvider<GRID_TYPE>::storedNanoSteps;

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
    {
        std::size_t firstRegularEventStep = firstStep;
        std::size_t period = steerer->getPeriod();
        std::size_t offset = firstStep % period;
        firstRegularEventStep = firstStep + period - offset;

        storedNanoSteps << firstNanoStep;
        storedNanoSteps << firstRegularEventStep * NANO_STEPS;
        storedNanoSteps << lastNanoStep;
    }

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
            throw std::logic_error(
                "SteererAdapter expects to be called only at the beginning of a time step (nanoStep == 0)");
        }

        std::size_t step = globalNanoStep / NANO_STEPS;

        SteererEvent event = STEERER_NEXT_STEP;
        if (globalNanoStep == firstNanoStep) {
            event = STEERER_INITIALIZED;
        }
        if (globalNanoStep == lastNanoStep) {
            event = STEERER_ALL_DONE;
        }
        if (globalNanoStep > lastNanoStep) {
            return;
        }

        if ((event == STEERER_NEXT_STEP) && (step % steerer->getPeriod() != 0)) {
            throw std::logic_error("SteererAdapter called at wrong step (got " + StringOps::itoa(step) +
                                   " but expected multiple of " + StringOps::itoa(steerer->getPeriod()));
        }

        typename Steerer<CELL_TYPE>::SteererFeedback feedback;

        steerer->nextStep(
            destinationGrid,
            patchableRegion,
            globalGridDimensions,
            step,
            event,
            rank,
            lastCall,
            &feedback);

        if (remove) {
            storedNanoSteps.erase(globalNanoStep);
            storedNanoSteps << globalNanoStep + NANO_STEPS * steerer->getPeriod();
        }

        // fixme: apply SteererFeedback!
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
