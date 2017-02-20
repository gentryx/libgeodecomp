#ifndef LIBGEODECOMP_IO_PARALLELWRITER_H
#define LIBGEODECOMP_IO_PARALLELWRITER_H

#include <libgeodecomp/config.h>
#include <libgeodecomp/io/writer.h>
#include <libgeodecomp/geometry/coord.h>
#include <libgeodecomp/geometry/region.h>
#include <libgeodecomp/parallelization/distributedsimulator.h>

#include <string>
#include <stdexcept>

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
    typedef typename APITraits::SelectTopology<CELL_TYPE>::Value Topology;
    typedef typename DistributedSimulator<CELL_TYPE>::GridType GridType;
    typedef Region<Topology::DIM> RegionType;
    typedef Coord<Topology::DIM> CoordType;

    /**
     * is the equivalent to Writer().
     */
    ParallelWriter(
        const std::string& prefix,
        const unsigned period) :
        prefix(prefix),
        period(period)
    {
        if (period == 0) {
            throw std::invalid_argument("period must be positive");
        }
    }

    virtual ~ParallelWriter()
    {}

    /**
     * "virtual copy constructor". This function may be called
     * whenever a deep copy of a ParallelWriter is needed instead of a plain
     * pointer copy.
     *
     * Advice to implementers: use CRTP (
     * http://en.wikipedia.org/wiki/Curiously_recurring_template_pattern
     * ) to implement this automagically -- see other ParallelWriter
     * implementations for advice on this subject.
     */
    virtual ParallelWriter *clone() const = 0;

    /**
     * notifies the ParallelWriter that the supplied region is the
     * domain of the current process. This fuction will be called once
     * the domain decomposition has been done. Writers can use this
     * information to decide on the size of buffers to allocate or
     * determine file offsets. validRegion in stepFinished() will
     * always be a subset of newRegion.
     */
    virtual void setRegion(const Region<Topology::DIM>& newRegion)
    {
        region = newRegion;
    }

    /**
     * is called back from sim after each simulation step. event
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
        const RegionType& validRegion,
        const CoordType& globalDimensions,
        unsigned step,
        WriterEvent event,
        std::size_t rank,
        bool lastCall) = 0;

    unsigned getPeriod() const
    {
        return period;
    }

    const std::string& getPrefix() const
    {
        return prefix;
    }

protected:
    Region<Topology::DIM> region;
    std::string prefix;
    unsigned period;
};

}

#endif
