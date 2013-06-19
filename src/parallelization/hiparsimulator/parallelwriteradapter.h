#ifndef LIBGEODECOMP_PARALLELIZATION_HIPARSIMULATOR_PARALLELWRITERADAPTER_H
#define LIBGEODECOMP_PARALLELIZATION_HIPARSIMULATOR_PARALLELWRITERADAPTER_H

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
template<typename GRID_TYPE, typename CELL_TYPE, typename SIMULATOR>
class ParallelWriterAdapter : public PatchAccepter<GRID_TYPE>
{
public:

    using PatchAccepter<GRID_TYPE>::checkNanoStepPut;
    using PatchAccepter<GRID_TYPE>::pushRequest;
    using PatchAccepter<GRID_TYPE>::requestedNanoSteps;

    ParallelWriterAdapter(
        SIMULATOR * _sim,
        boost::shared_ptr<ParallelWriter<CELL_TYPE> > _writer,
        const long& firstStep,
        const long& lastStep,
        Coord<CELL_TYPE::Topology::DIM> globalGridDimensions,
        bool lastCall) :
        sim(_sim),
        writer(_writer),
        firstNanoStep(firstStep * CELL_TYPE::nanoSteps()),
        lastNanoStep(lastStep   * CELL_TYPE::nanoSteps()),
        lastCall(lastCall),
        globalGridDimensions(globalGridDimensions)
    {
        reload(firstNanoStep);
        reload(lastNanoStep);
    }

    virtual void setRegion(const Region<GRID_TYPE::DIM>& region)
    {
        writer->setRegion(region);
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
    SIMULATOR * sim;
    boost::shared_ptr<ParallelWriter<CELL_TYPE> > writer;
    long firstNanoStep;
    long lastNanoStep;
    bool lastCall;
    Coord<CELL_TYPE::Topology::DIM> globalGridDimensions;

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
