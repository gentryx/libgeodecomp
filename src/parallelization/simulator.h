#ifndef _libgeodecomp_parallelization_simulator_h_
#define _libgeodecomp_parallelization_simulator_h_

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
    typedef typename CELL_TYPE::Topology Topology;
    typedef Grid<CELL_TYPE, Topology> GridType;
    typedef SuperVector<boost::shared_ptr<Steerer<CELL_TYPE> > > SteererVector;

    /**
     * Creates the abstract Simulator object. The Initializer is
     * assumed to belong to the Simulator, which means that it'll
     * delete the @a _initializer at the end of its lifetime.
     */
    inline Simulator(Initializer<CELL_TYPE> *_initializer) : 
        stepNum(0), 
        initializer(_initializer)
    {}

    inline virtual ~Simulator() 
    { 
        delete initializer;
    }

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

    virtual Initializer<CELL_TYPE> *getInitializer() const
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
    Initializer<CELL_TYPE> *initializer;
    SteererVector steerers;
};

}

#endif
