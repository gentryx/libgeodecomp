#ifndef LIBGEODECOMP_PARALLELIZATION_SIMULATOR_H
#define LIBGEODECOMP_PARALLELIZATION_SIMULATOR_H

#include <vector>
#include <boost/shared_ptr.hpp>
#include <libgeodecomp/io/initializer.h>
#include <libgeodecomp/io/steerer.h>
#include <libgeodecomp/misc/grid.h>

namespace LibGeoDecomp {

/**
 * This is the abstract main application class. Its descendants
 * perform the iteration of simulation steps.
 */
template<typename CELL_TYPE>
class Simulator
{
public:
    typedef typename CellAPITraitsFixme::SelectTopology<CELL_TYPE>::Value Topology;
    static const int DIM = Topology::DIM;
    typedef GridBase<CELL_TYPE, DIM> GridType;
    typedef SuperVector<boost::shared_ptr<Steerer<CELL_TYPE> > > SteererVector;

    /**
     * Creates the abstract Simulator object. The Initializer is
     * assumed to belong to the Simulator, which means that it'll
     * delete the  initializer at the end of its lifetime.
     */
    inline Simulator(Initializer<CELL_TYPE> *initializer) :
        stepNum(0),
        initializer(initializer),
        gridDim(initializer->gridDimensions())
    {}

    inline virtual ~Simulator()
    {}

    /**
     * performs a single simulation step.
     */
    virtual void step() = 0;

    /**
     * performs step() until the maximum number of steps is reached.
     */
    virtual void run() = 0;

    /**
     * returns the number of the current logical simulation step.
     */
    virtual unsigned getStep() const
    {
        return stepNum;
    }

    virtual boost::shared_ptr<Initializer<CELL_TYPE> > getInitializer() const
    {
        return initializer;
    }

    /**
     * adds a Steerer which will be called back before simulation
     * steps as specified by the Steerer's period. The Steerer is
     * assumed to be owned by the Simulator. It will destroy the
     * Steerer upon its death.
     */
    virtual void addSteerer(Steerer<CELL_TYPE> *steerer)
    {
        steerers << boost::shared_ptr<Steerer<CELL_TYPE> >(steerer);
    }

protected:
    unsigned stepNum;
    boost::shared_ptr<Initializer<CELL_TYPE> > initializer;
    SteererVector steerers;
    Coord<DIM> gridDim;
};

}

#endif
