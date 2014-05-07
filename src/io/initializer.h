#ifndef LIBGEODECOMP_IO_INITIALIZER_H
#define LIBGEODECOMP_IO_INITIALIZER_H

#include <libgeodecomp/config.h>
#include <libgeodecomp/misc/apitraits.h>
#include <libgeodecomp/storage/gridbase.h>

namespace LibGeoDecomp {

/**
 * The initializer sets up the initial state of the grid. For this a
 * Simulator will invoke Initializer::grid(). Keep in mind that grid()
 * might be called multiple times and that for parallel runs each
 * Initializer will be responsible just for a sub-cuboid of the whole
 * grid.
 */
template<typename CELL>
class Initializer
{
public:
    friend class Serialization;
    typedef typename APITraits::SelectTopology<CELL>::Value Topology;

    static const unsigned NANO_STEPS = APITraits::SelectNanoSteps<CELL>::VALUE;
    static const int DIM = Topology::DIM;

    virtual ~Initializer()
    {}

    /**
     * initializes all cells of the grid at target
     */
    virtual void grid(GridBase<CELL, DIM> *target) = 0;

    /**
     * Allows a Simulator to discover the extent of the whole
     * simulation. Usually Simulations will use 0 as the origin, but
     * there is no oblication to do so.
     */
    virtual CoordBox<DIM> gridBox()
    {
        return CoordBox<DIM>(Coord<DIM>(), gridDimensions());
    }

    /**
     * returns the size of the gridBox().
     */
    virtual Coord<DIM> gridDimensions() const = 0;

    /**
     * yields the logical time step at which the simulation should start
     */
    virtual unsigned startStep() const = 0;

    /**
     * gives the time step at which the simulation should terminate.
     *
     * Example: if startStep is 0 and maxSteps is 10, then the
     * Simulator should start at t=0, update to t=1, update to t=2,
     * ... until it has updated to t=10.
     *
     * If startStep is 5 and maxSteps is 8, then the Simulator is
     * expected to init at t=5, update to t=6, update to t=7, and
     * finally update to t=8.
     */
    virtual unsigned maxSteps() const = 0;

};

}

#endif
