#ifndef _libgeodecomp_parallelization_hiparsimulator_parallelwriteradapter_h_
#define _libgeodecomp_parallelization_hiparsimulator_parallelwriteradapter_h_

//#include <libgeodecomp/parallelization/hiparsimulator.h>
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
template<typename GRID_TYPE, typename CELL_TYPE, typename Simulator>
class ParallelWriterAdapter : public PatchAccepter<GRID_TYPE>
{
public:

    using PatchAccepter<GRID_TYPE>::checkNanoStepPut;
    using PatchAccepter<GRID_TYPE>::pushRequest;
    using PatchAccepter<GRID_TYPE>::requestedNanoSteps;

    ParallelWriterAdapter(
        boost::shared_ptr<ParallelWriter<CELL_TYPE> > _writer,
        Simulator * _sim,
        const long& firstStep,
        const long& lastStep) :
        writer(_writer),
        sim(_sim),
        firstNanoStep(firstStep * CELL_TYPE::nanoSteps()),
        lastNanoStep(lastStep   * CELL_TYPE::nanoSteps())
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

        if (nanoStep == firstNanoStep) {
            writer->initialized(grid, validRegion);
        } else {
            if (nanoStep == lastNanoStep) {
                writer->allDone(grid, validRegion);
            } else {
                writer->stepFinished(grid, validRegion, nanoStep);
            }
        }

        // delete the pointers from the Simulator to prevent accesses
        // to stale pointers:
        reload();
    }

private:
    boost::shared_ptr<ParallelWriter<CELL_TYPE> > writer;
    Simulator * sim;
    long firstNanoStep;
    long lastNanoStep;

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
