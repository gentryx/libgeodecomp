#ifndef _libgeodecomp_parallelization_simulator_h_
#define _libgeodecomp_parallelization_simulator_h_

#include <vector>
#include <boost/shared_ptr.hpp>
#include <libgeodecomp/io/initializer.h>
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
    
    inline virtual ~Simulator() 
    { 
        delete initializer;
    }

    inline Simulator(Initializer<CELL_TYPE> *_initializer) : 
        stepNum(0), initializer(_initializer)
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
     * @return the number of the current logical step.
     */
    virtual unsigned getStep() const 
    { 
        return stepNum; 
    }

    virtual Initializer<CELL_TYPE> *getInitializer() const
    {
        return initializer;
    }
    
protected:
    unsigned stepNum;
    Initializer<CELL_TYPE> *initializer;
};

};

#endif
