#ifndef _libgeodecomp_io_parallelwriter_h_
#define _libgeodecomp_io_parallelwriter_h_

#include <string>
#include <stdexcept>
#include <libgeodecomp/parallelization/distributedsimulator.h>

namespace LibGeoDecomp {

template<typename CELL_TYPE>
class DistributedSimulator;

/**
 * ParallelWriter is the parent class for all parallel IO. Its being
 * used with ParallelSimulator, which contrasts it from Writer. Just
 * like writer, it defines a number of callbacks which are invoked by
 * the simulator. Also, ParallelWriter registers at the 
 * DistributedSimulator, which will delete it upon its destruction.
 * Never allocate a ParallelWriter on the stack! 
 *
 * A conceptual difference from Writer should be noted: multiple
 * ParallelWriter objects of the same type will exists, typically one
 * per MPI process. Thus one either needs to use MPI IO or individual
 * files per instance. For other differences see below.
 */
template<typename CELL_TYPE>
class ParallelWriter
{
public:

    /**
     * is the equivalent to Writer().
     */
    ParallelWriter(
        const std::string& _prefix, 
        DistributedSimulator<CELL_TYPE> *_distSim, 
        const unsigned& _period = 1) : 
        prefix(_prefix), distSim(_distSim), period(_period)
    {
        if (prefix == "") {
            throw std::invalid_argument("empty prefixes are forbidden");
        }
        if (period == 0) {
            throw std::invalid_argument("period must be positive");
        }
        if (distSim) {
            distSim->registerWriter(this);
        }
    }

    virtual ~ParallelWriter() {};    

    /**
     * is equivalent to Writer::initialized()
     */
    virtual void initialized() = 0;

    /**
     * is similar to Writer::stepFinished() BUT this function may be
     * called multiple times per timestep. This is because simulators
     * will typically update the ghost zones before they update the
     * inner sets of their domain. Calling stepFinished() for each
     * updated region simplifies their design and avoids buffering.
     * The simulator will provide the ParallelWriter with which region
     * of the grid needs to be output.
     */
    virtual void stepFinished() = 0;

    /**
     * is equivalent to Writer::addDone()
     */
    virtual void allDone() = 0;

    const unsigned& getPeriod() const
    {
        return period;
    }

    const std::string& getPrefix() const
    {
        return prefix;
    }

protected:
    std::string prefix;
    DistributedSimulator<CELL_TYPE> *distSim;
    unsigned period;
};

}

#endif
