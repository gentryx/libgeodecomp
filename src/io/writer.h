#ifndef _libgeodecomp_io_writer_h_
#define _libgeodecomp_io_writer_h_

#include <string>
#include <stdexcept>
#include <libgeodecomp/parallelization/monolithicsimulator.h>

namespace LibGeoDecomp {

template<typename CELL_TYPE>
class MonolithicSimulator;

enum WriterEvent {
    WRITER_INITIALIZED,
    WRITER_STEP_FINISHED,
    WRITER_ALL_DONE
};

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
    typedef typename MonolithicSimulator<CELL_TYPE>::GridType GridType;

    /**
     * initializes a writer using \param _prefix which subclasses may
     * use to generate filenames. \param _period should be used by
     * them to control how many time steps lie between outputs. The
     * Writer will register at \param _sim which in turn will delete
     * the Writer. Thus a writer should always be constructed via
     * new(), unless _sim is 0.
     */
    Writer(
        const std::string& prefix, 
        const unsigned period = 1) : 
        prefix(prefix), 
        period(period)
    {
        if (prefix == "") {
            throw std::invalid_argument("empty prefixes are forbidden");
        }

        if (period == 0) {
            throw std::invalid_argument("period must be positive");
        }
    }

    virtual ~Writer() {};    

    /**
     * is called back from \a sim after each simulation step. event
     * specifies the phase in which the simulation is currently in.
     * This may be used for instance to open/close files at the
     * beginning/end of the simulation.
     */
    virtual void stepFinished(const GridType& grid, unsigned step, WriterEvent event) = 0;

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
    unsigned period;
};

}

#endif
