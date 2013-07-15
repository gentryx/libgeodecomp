#ifndef LIBGEODECOMP_IO_STEERER_H
#define LIBGEODECOMP_IO_STEERER_H

#include <libgeodecomp/config.h>
#include <libgeodecomp/misc/gridbase.h>
#include <libgeodecomp/misc/region.h>

#ifdef LIBGEODECOMP_FEATURE_BOOST_SERIALIZATION
#include <boost/serialization/base_object.hpp>
#endif

namespace LibGeoDecomp {

enum SteererEvent {
    STEERER_INITIALIZED,
    STEERER_NEXT_STEP,
    STEERER_ALL_DONE
};

template<typename CELL_TYPE>
class Steerer
{
public:
    typedef typename CELL_TYPE::Topology Topology;
    typedef GridBase<CELL_TYPE, Topology::DIM> GridType;
    typedef Coord<Topology::DIM> CoordType;

    /**
     * A steerer is an object which is allowed to modify a Simulator's
     * (region of the) grid. It is the counterpart to a ParallelWriter
     * (there is no counterpart to the SerialWriter though. Steerers
     * are all expected to run in parallel). Possible uses include
     * dynamically introducing new obstacles in a LBM solver or
     * modifying the ambient temperature in a dendrite simulation.
     */
    Steerer(const unsigned period) :
        period(period)
    {}

    virtual ~Steerer()
    {}

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
     * grid. The part which is accessible via \p grid is specified in
     * \p validRegion. The current time step is given in \p step. This
     * function may be called multiple times per step (e.g. seperately
     * for inner ghost zones and inner set (which is equivalent to the
     * interface of ParallelWriter).
     */
    virtual void nextStep(
        GridType *grid,
        const Region<Topology::DIM>& validRegion,
        const CoordType& globalDimensions,
        unsigned step,
        SteererEvent event,
        std::size_t rank,
        bool lastCall) = 0;

    const unsigned& getPeriod() const
    {
        return period;
    }

#ifdef LIBGEODECOMP_FEATURE_BOOST_SERIALIZATION
    template <typename Archive>
    void serialize(Archive& ar, unsigned)
    {
        ar & region;
        ar & period;
    }
#endif

protected:
    Region<Topology::DIM> region;
    unsigned period;
};

}

#endif
