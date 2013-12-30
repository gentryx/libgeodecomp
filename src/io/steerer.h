#ifndef LIBGEODECOMP_IO_STEERER_H
#define LIBGEODECOMP_IO_STEERER_H

#include <libgeodecomp/config.h>
#include <libgeodecomp/geometry/region.h>
#include <libgeodecomp/misc/apitraits.h>
#include <libgeodecomp/storage/gridbase.h>

namespace LibGeoDecomp {

enum SteererEvent {
    STEERER_INITIALIZED,
    STEERER_NEXT_STEP,
    STEERER_ALL_DONE
};

/**
 * A steerer is an object which is allowed to modify a Simulator's
 * (region of the) grid. It is the counterpart to a ParallelWriter
 * (there is no counterpart to the SerialWriter though. Steerers
 * are all expected to run in parallel). Possible uses include
 * dynamically introducing new obstacles in a LBM solver or
 * modifying the ambient temperature in a dendrite simulation.
 */
template<typename CELL_TYPE>
class Steerer
{
public:
    friend class Serialization;
    typedef typename APITraits::SelectStaticData<CELL_TYPE>::Value StaticData;
    typedef typename APITraits::SelectTopology<CELL_TYPE>::Value Topology;
    typedef GridBase<CELL_TYPE, Topology::DIM> GridType;
    typedef Coord<Topology::DIM> CoordType;

    /**
     * This class may be used by a Steerer to convey feedback to the
     * Simulator.
     */
    class SteererFeedback
    {
    public:
        SteererFeedback() :
            simulationEnd(false)
        {}

        void endSimulation()
        {
            simulationEnd = true;
        }

        /**
         * Instructs the Simulator to update the Cell's static data
         * block (e.g. global constants like dt or an ambient
         * temperature). The update will be carried out at the next
         * possible time step and will be synchronized among all parts
         * of the grid (i.e. all nodes and all threads/GPUs therein).
         */
        void setStaticData(const StaticData& data)
        {
            // immediate assignent is good enough for now. we might
            // want to hook into here for more complex NUMA scenarios.
            CELL_TYPE::staticData = data;
        }

        bool simulationEnded()
        {
            return simulationEnd;
        }

    private:
        bool simulationEnd;
    };

    Steerer(const unsigned period) :
        period(period)
    {}

    virtual ~Steerer()
    {}

    /**
     * "Virtual Copy constructor"
     * This function may be called whenever a copy of a steerer is needed
     * instead of a plain pointer copy. Must be implemented by t
     **/
    virtual Steerer *clone()
    {
        throw std::logic_error("clone not implemented");
    }

    /**
     * notifies the Steerer that the supplied region is the domain of
     * the current process. This fuction will be called once the
     * domain decomposition has been done. Steerers can use this
     * information to determine for instance where to forward certain
     * steering data fragments. validRegion in nextStep() will
     * generally be a subset of newRegion.
     */
    virtual void setRegion(const Region<Topology::DIM>& newRegion)
    {
        region = newRegion;
    }

    /**
     * is a callback which gives the Steerer access to a Simulator's
     * grid. Returns false if the Steerer wishes to end the simulation.
     *
     * The part of the simulation space which is accessible via \p
     * grid is specified in \p validRegion. The current time step is
     * given in \p step. This function may be called multiple times
     * per step (e.g. seperately for inner ghost zones and inner set
     * (which is equivalent to the interface of ParallelWriter).
     */
    virtual void nextStep(
        GridType *grid,
        const Region<Topology::DIM>& validRegion,
        const CoordType& globalDimensions,
        unsigned step,
        SteererEvent event,
        std::size_t rank,
        bool lastCall,
        SteererFeedback *feedback) = 0;

    const unsigned& getPeriod() const
    {
        return period;
    }

protected:
    Region<Topology::DIM> region;
    unsigned period;
};

}

#endif
