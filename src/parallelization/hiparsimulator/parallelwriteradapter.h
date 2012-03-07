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
        boost::shared_ptr<ParallelWriter<CELL_TYPE> > _writer) :
        sim(_sim),
        writer(_writer)
    {
        reload();
    }

    virtual void put(
        const GRID_TYPE& grid, 
        const Region<GRID_TYPE::DIM>& validRegion, 
        const long& nanoStep) 
    {
        std::cout << "bingobongoA " << nanoStep << "\n";
        if (!this->checkNanoStepPut(nanoStep))
            return;
        this->requestedNanoSteps.erase_min();

        std::cout << "bingobongoB " << nanoStep << "\n";
        // fixme: load next event
        // fixme: set simulator up to link to correct grid/validRegion
        // writer->stepFinished();
        reload();
    }

private:
    HiParSimulatorType *sim;
    boost::shared_ptr<ParallelWriter<CELL_TYPE> > writer;

    long nextOutputStep()
    {
        long step = sim->getStep();
        long remainder = step % writer->getPeriod();
        step += writer->getPeriod() - remainder;
        return step;
    }

    void reload()
    {
        long nextNanoStep = nextOutputStep() * CELL_TYPE::nanoSteps();
        std::cout << "nextNanoStep: " << nextNanoStep << "\n";
        this->pushRequest(nextNanoStep);
    }
 
};

}
}

#endif
