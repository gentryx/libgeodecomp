#ifndef _libgeodecomp_io_parallelwriter_h_
#define _libgeodecomp_io_parallelwriter_h_

#include <string>
#include <stdexcept>
#include <libgeodecomp/parallelization/distributedsimulator.h>

#include <boost/serialization/base_object.hpp>

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
    typedef GridBase<CELL_TYPE, Topology::DIMENSIONS> GridType;
    typedef Region<Topology::DIMENSIONS> RegionType;

    /**
     * is the equivalent to Writer().
     */
    ParallelWriter(
        const std::string& _prefix, 
        const unsigned& _period = 1): 
        prefix(_prefix), period(_period)
    {
        if (prefix == "") 
            throw std::invalid_argument("empty prefixes are forbidden");
        if (period == 0) 
            throw std::invalid_argument("period must be positive");
    }

    virtual ~ParallelWriter() {};    

    /**
     * is equivalent to Writer::initialized()
     */
    virtual void initialized(GridType const & grid, RegionType const & region) = 0;

    /**
     * is similar to Writer::stepFinished() BUT this function may be
     * called multiple times per timestep. This is because simulators
     * will typically update the ghost zones before they update the
     * inner sets of their domain. Calling stepFinished() for each
     * updated region simplifies their design and avoids buffering.
     * The simulator will provide the ParallelWriter with which region
     * of the grid needs to be output.
     */
    virtual void stepFinished(GridType const & grid, RegionType const & region, std::size_t) = 0;

    /**
     * is equivalent to Writer::addDone()
     */
    virtual void allDone(GridType const & grid, RegionType const & region) = 0;

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
private:
    friend class boost::serialization::access;
    template <typename Archive>
    void serialize(Archive & ar, unsigned)
    {
        ar & prefix;
        ar & period;
    }
};

};

#endif
