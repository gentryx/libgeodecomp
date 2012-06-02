#ifndef _libgeodecomp_io_writer_h_
#define _libgeodecomp_io_writer_h_

#include <string>
#include <stdexcept>
#include <libgeodecomp/parallelization/monolithicsimulator.h>

namespace LibGeoDecomp {

template<typename CELL_TYPE>
class MonolithicSimulator;

/**
 * Writer and ParallelWriter are the superclasses for all output
 * formats. Writer is for use with MonolithicSimulator and its heirs.
 * It defines three callbacks which are invoked by the simulator. The
 * prefix may be used by file-based IO objects to generate file names.
 * Usually one will add a time step number and a suitable extension.
 */
template<typename CELL_TYPE>
class Writer
{
    friend class WriterTest;

public:

    /**
     * initializes a writer using \param _prefix which subclasses may
     * use to generate filenames. \param _period should be used by
     * them to control how many time steps lie between outputs. The
     * Writer will register at \param _sim which in turn will delete
     * the Writer. Thus a writer should always be constructed via
     * new(), unless _sim is 0.
     */
    Writer(
        const std::string& _prefix, 
        MonolithicSimulator<CELL_TYPE> *_sim, 
        const unsigned& _period = 1) : 
        prefix(_prefix), 
        sim(_sim), 
        period(_period)
    {
        if (prefix == "") {
            throw std::invalid_argument("empty prefixes are forbidden");
        }
        if (period == 0) {
            throw std::invalid_argument("period must be positive");
        }
        if (sim) {
            sim->registerWriter(this);
        }
    }

    virtual ~Writer() {};    

    /**
     * is called back from \a sim after the initialization phase is
     * finished. This may be useful to e.g. open files.
     */
    virtual void initialized() = 0;

    /**
     * is called back from \a sim after each simulation step.
     */
    virtual void stepFinished() = 0;

    /**
     * is called back from \a sim at the end of the whole simulation.
     * The Writer may close open files or do any other finalization
     * routine.
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
