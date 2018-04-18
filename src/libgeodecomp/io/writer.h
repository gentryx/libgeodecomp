#ifndef LIBGEODECOMP_IO_WRITER_H
#define LIBGEODECOMP_IO_WRITER_H

#include <libgeodecomp/config.h>
#include <libgeodecomp/parallelization/monolithicsimulator.h>
#include <libflatarray/macros.hpp>

LIBFLATARRAY_DISABLE_SYSTEM_HEADER_WARNINGS_PRE
#ifdef LIBGEODECOMP_WITH_MPI
#include <mpi.h>
#endif
#include <string>
#include <stdexcept>
LIBFLATARRAY_DISABLE_SYSTEM_HEADER_WARNINGS_POST

namespace LibGeoDecomp {

enum WriterEvent {
    WRITER_INITIALIZED,
    WRITER_STEP_FINISHED,
    WRITER_ALL_DONE
};

template <class CELL_TYPE> class MonolithicSimulator;

#ifdef _MSC_BUILD
#pragma warning( push )
#pragma warning( disable : 4710 4711 )
#endif

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
public:
    friend class WriterTest;

    typedef typename MonolithicSimulator<CELL_TYPE>::GridType GridType;
    typedef typename APITraits::SelectTopology<CELL_TYPE>::Value Topology;
    const static int DIM = Topology::DIM;
    static const unsigned NANO_STEPS = APITraits::SelectNanoSteps<CELL_TYPE>::VALUE;

    /**
     * initializes a writer using prefix which subclasses may
     * use to generate filenames. period should be used by
     * them to control how many time steps lie between outputs. The
     * Writer will register at sim which in turn will delete
     * the Writer. Thus a writer should always be constructed via
     * new(), unless sim is 0.
     */
    Writer(
        const std::string& prefix,
        const unsigned period) :
        prefix(prefix),
        period(period)
    {
        if (period == 0) {
            throw std::invalid_argument("period must be positive");
        }
    }

    virtual ~Writer()
    {}

    /**
     * "virtual copy constructor". This function may be called
     * whenever a deep copy of a Writer is needed instead of a plain
     * pointer copy.
     *
     * Advice to implementers: use CRTP (
     * http://en.wikipedia.org/wiki/Curiously_recurring_template_pattern
     * ) to implement this automagically -- see other Writer
     * implementations for advice on this subject.
     */
    virtual Writer *clone() const = 0;

    /**
     * is called back from sim after each simulation step. event
     * specifies the phase in which the simulation is currently in.
     * This may be used for instance to open/close files at the
     * beginning/end of the simulation.
     */
    virtual void stepFinished(const GridType& grid, unsigned step, WriterEvent event) = 0;

    unsigned getPeriod() const
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

#ifdef _MSC_BUILD
#pragma warning( pop )
#endif

}

#endif
