#ifndef _libgeodecomp_parallelization_hiparsimulator_parallelwriteradapter_h_
#define _libgeodecomp_parallelization_hiparsimulator_parallelwriteradapter_h_

#include <libgeodecomp/parallelization/hiparsimulator.h>
#include <libgeodecomp/parallelization/hiparsimulator/patchaccepter.h>

namespace LibGeoDecomp {
namespace HiParSimulator {

template<class CELL_TYPE, class PARTITION>
class HiParSimulator;

/**
 * ParallelWriterAdapter translates the interface of a ParallelWriter
 * to a PatchAccepter, so that we can treat IO similarly to sending
 * ghost zones.
 */
template<typename GRID_TYPE, typename CELL_TYPE, typename PARTITION>
class ParallelWriterAdapter : public PatchAccepter<GRID_TYPE>
{
public:
    typedef HiParSimulator<CELL_TYPE, PARTITION> HiParSimulatorType;

    using PatchAccepter<GRID_TYPE>::checkNanoStepPut;
    using PatchAccepter<GRID_TYPE>::pushRequest;
    using PatchAccepter<GRID_TYPE>::requestedNanoSteps;

    ParallelWriterAdapter(
        HiParSimulatorType *sim,
        boost::shared_ptr<ParallelWriter<CELL_TYPE> > writer,
        const long& firstStep,
        const long& lastStep,
        Coord<CELL_TYPE::Topology::DIMENSIONS> globalGridDimensions,
        bool lastCall) :
        sim(sim),
        writer(writer),
        firstNanoStep(firstStep * CELL_TYPE::nanoSteps()),
        lastNanoStep(lastStep   * CELL_TYPE::nanoSteps()),
        globalGridDimensions(globalGridDimensions),
        lastCall(lastCall)
    {
        reload(firstNanoStep);
        reload(lastNanoStep);
    }

    virtual void put(
        const GRID_TYPE& grid, 
        const Region<GRID_TYPE::DIM>& validRegion, 
        const long& nanoStep) 
    {
        if (!checkNanoStepPut(nanoStep)) {
            return;
        }

        requestedNanoSteps.erase_min();

        WriterEvent event = WRITER_STEP_FINISHED;
        if (nanoStep == firstNanoStep) {
            event = WRITER_INITIALIZED;
        }
        if (nanoStep == lastNanoStep) {
            event = WRITER_ALL_DONE;
        }

        writer->stepFinished(
            grid, 
            validRegion, 
            globalGridDimensions, 
            nanoStep / CELL_TYPE::nanoSteps(), 
            event, 
            lastCall);
        reload();
    }

private:
    HiParSimulatorType *sim;
    boost::shared_ptr<ParallelWriter<CELL_TYPE> > writer;
    long firstNanoStep;
    long lastNanoStep;
    bool lastCall;
    Coord<CELL_TYPE::Topology::DIMENSIONS> globalGridDimensions;

    long nextOutputStep(const long& step)
    {
        long remainder = step % writer->getPeriod();
        long next = step + writer->getPeriod() - remainder;
        return next;
    }

    void reload()
    {
        long nextNanoStep = nextOutputStep(sim->getStep()) * CELL_TYPE::nanoSteps();
        reload(nextNanoStep);
    }

    void reload(const long& nextNanoStep)
    {
        pushRequest(nextNanoStep);
    }
 
};

}
}

#endif
