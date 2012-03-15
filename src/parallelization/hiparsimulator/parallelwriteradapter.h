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

    ParallelWriterAdapter(
        HiParSimulatorType *_sim,
        boost::shared_ptr<ParallelWriter<CELL_TYPE> > _writer,
        const long& firstStep,
        const long& lastStep) :
        sim(_sim),
        writer(_writer),
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
        if (!this->checkNanoStepPut(nanoStep))
            return;
        this->requestedNanoSteps.erase_min();

        sim->setGridFragment(&grid, &validRegion);

        if (nanoStep == firstNanoStep) {
            writer->initialized();
        } else {
            if (nanoStep == lastNanoStep) {
                writer->allDone();
            } else {
                writer->stepFinished();
            }
        }

        // delete the pointers from the Simulator to prevent accesses
        // to stale pointers:
        sim->setGridFragment(0, 0);

        reload();
    }

private:
    HiParSimulatorType *sim;
    boost::shared_ptr<ParallelWriter<CELL_TYPE> > writer;
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
        this->pushRequest(nextNanoStep);
    }
 
};

}
}

#endif
