#ifndef LIBGEODECOMP_IO_PARALLELWRITER_H
#define LIBGEODECOMP_IO_PARALLELWRITER_H

#include <string>
#include <stdexcept>
#include <libgeodecomp/io/writer.h>
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
    typedef typename CELL_TYPE::Topology Topology;
    typedef typename DistributedSimulator<CELL_TYPE>::GridType GridType;

    /**
     * is the equivalent to Writer().
     */
    ParallelWriter(
        const std::string& prefix, 
        const unsigned& period = 1) : 
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

    virtual ~ParallelWriter() 
    {};    

    /**
     * is called back from \a sim after each simulation step. event
     * specifies the phase in which the simulation is currently in.
     * This may be used for instance to open/close files at the
     * beginning/end of the simulation. lastCall is set to true if
     * this is the final invocation for this step -- handy if the
     * simulator needs to call the writer multiple times for different
     * parts of the grid (e.g. for the ghost zones and then again for
     * the interior of the domain).
     */
    virtual void stepFinished(
        const GridType& grid, 
        const Region<Topology::DIMENSIONS>& validRegion, 
        const Coord<Topology::DIMENSIONS>& globalDimensions,
        unsigned step, 
        WriterEvent event, 
        bool lastCall) = 0;

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
