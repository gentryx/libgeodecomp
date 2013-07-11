#ifndef LIBGEODECOMP_IO_WRITER_H
#define LIBGEODECOMP_IO_WRITER_H

#include <string>
#include <stdexcept>
#include <libgeodecomp/config.h>
#include <libgeodecomp/parallelization/monolithicsimulator.h>
#ifdef LIBGEODECOMP_FEATURE_BOOST_SERIALIZATION
#include <boost/serialization/access.hpp>
#endif

namespace LibGeoDecomp {

enum WriterEvent {
    WRITER_INITIALIZED,
    WRITER_STEP_FINISHED,
    WRITER_ALL_DONE
};

template <class CELL_TYPE> class MonolithicSimulator;

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
#ifdef LIBGEODECOMP_FEATURE_BOOST_SERIALIZATION
    friend class boost::serialization::access;
#endif
public:
    typedef typename MonolithicSimulator<CELL_TYPE>::GridType GridType;
    const static int DIM = CELL_TYPE::Topology::DIM;

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
        const unsigned period) :
        prefix(prefix),
        period(period)
    {
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

#ifdef LIBGEODECOMP_FEATURE_BOOST_SERIALIZATION
    Writer()
    {}

    template <typename ARCHIVE>
    void serialize(ARCHIVE & ar, unsigned)
    {
        ar & prefix;
        ar & period;
    }
#endif
};

}

#endif
