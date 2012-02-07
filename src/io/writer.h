#ifndef _libgeodecomp_io_writer_h_
#define _libgeodecomp_io_writer_h_

#include <string>
#include <stdexcept>
#include <libgeodecomp/parallelization/monolithicsimulator.h>

namespace LibGeoDecomp {

template<typename CELL_TYPE>
class MonolithicSimulator;

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
class Writer
{
    friend class WriterTest;

public:

    /**
     * initialize a writer using @a prefix which is an incomplete pathname to
     * the output file(s) lacking an extension. The extension needs to be
     * appended by the concrete Writer. If a Writer generates several output
     * files, it should put a running number between the prefix and the
     * extension. @a prefix must not be empty.
     * The Writer should write a frame for @a frequency steps.
     */
    Writer(
        const std::string& _prefix, 
        MonolithicSimulator<CELL_TYPE> *_sim, 
        const unsigned& _period = 1) : 
        prefix(_prefix), 
        sim(_sim), 
        period(_period)
    {
        if (prefix == "") 
            throw std::invalid_argument("empty prefixes are forbidden");
        if (period == 0) 
            throw std::invalid_argument("period must be positive");
        if (sim)
            sim->registerWriter(this);
    }

    virtual ~Writer() {};    

    /**
     * is called back from @a sim after the initialization phase is finished.
     * This allow the Writer to query static parameters from the Simulator.
     */
    virtual void initialized() = 0;

    /**
     * is called back from @a sim after each simulation step. The Writer has to
     * decide whether it likes to write this step's results.
     */
    virtual void stepFinished() = 0;

    /**
     * is called back from @a sim at the end of the whole simulation. The Writer
     * may close open files or do any other finalization routine.
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
    MonolithicSimulator<CELL_TYPE> *sim;
    unsigned period;
};

}

#endif
