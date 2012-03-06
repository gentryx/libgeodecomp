#ifndef _libgeodecomp_io_parallelwriter_h_
#define _libgeodecomp_io_parallelwriter_h_

#include <string>
#include <stdexcept>
#include <libgeodecomp/parallelization/distributedsimulator.h>

namespace LibGeoDecomp {

template<typename CELL_TYPE>
class DistributedSimulator;

/**
 * Superclass for all output file formats.
 * 
 * The Writer defines three callbacks which are invoked by the Simulator on
 * appropriate times. Subclasses are required to implement the callbacks and
 * emit chunks of the output format. The file name prefix needs to be extended
 * by a specific suffix, which includes an optional step number and must end 
 * with a proper extension.
 */
template<typename CELL_TYPE>
class ParallelWriter
{
public:

    /**
     * initialize a writer using @a prefix which is an incomplete pathname to
     * the output file(s) lacking an extension. The extension needs to be
     * appended by the concrete Writer. If a Writer generates several output
     * files, it should put a running number between the prefix and the
     * extension. @a prefix must not be empty.
     * The Writer should write a frame for @a frequency steps.
     */
    ParallelWriter(
        const std::string& _prefix, 
        DistributedSimulator<CELL_TYPE> *_distSim, 
        const unsigned& _period = 1): 
        prefix(_prefix), distSim(_distSim), period(_period)
    {
        if (prefix == "") 
            throw std::invalid_argument("empty prefixes are forbidden");
        if (period == 0) 
            throw std::invalid_argument("period must be positive");
        if (distSim)
            distSim->registerWriter(this);
    }

    virtual ~ParallelWriter() {};    

    /**
     * is called from @a distSim after the initialization phase is
     * finished. This allow the Writer to query static parameters from
     * the Simulator.
     */
    virtual void initialized() = 0;

    /**
     * is called from @a distSim after each simulation step. One
     * important difference to a normal writer is, that a 
     * DistributedSimulator MAY call this function multiple times per
     * step on a single node. The reason for this is that a some
     * parallelizations may choose to update parts of the grid
     * seperately from other parts.
     */
    virtual void stepFinished() = 0;

    /**
     * is called from @a distSim at the end of the whole
     * simulation. The Writer may close open files or do any other
     * finalization routine.
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

};

#endif
