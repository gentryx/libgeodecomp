#ifndef _libgeodecomp_parallelization_serialsimulator_h_
#define _libgeodecomp_parallelization_serialsimulator_h_

#include <libgeodecomp/misc/grid.h>
#include <libgeodecomp/parallelization/monolithicsimulator.h>

namespace LibGeoDecomp {

/**
 * Implements the Simulator functionality by running all calculations
 * sequencially in a single process.
 */
template<typename CELL_TYPE>
class SerialSimulator : public MonolithicSimulator<CELL_TYPE>
{
    friend class SerialSimulatorTest;
    friend class PPMWriterTest;
    
public:
    typedef typename CELL_TYPE::Topology Topology;
    typedef Grid<CELL_TYPE, Topology> GridType;
    static const int DIMENSIONS = Topology::DIMENSIONS;

    /**
     * creates a SerialSimulator with the given @a initializer.
     */
    SerialSimulator(Initializer<CELL_TYPE> *_initializer) : 
        MonolithicSimulator<CELL_TYPE>(_initializer)
    {
        Coord<DIMENSIONS> dim = this->initializer->gridBox().dimensions;
        curGrid = new GridType(dim);
        newGrid = new GridType(dim);
        this->initializer->grid(curGrid);
        this->initializer->grid(newGrid);
    }

    ~SerialSimulator()
    {
        delete newGrid;
        delete curGrid;
    }

    /**
     * performs a single simulation step.
     */
    virtual void step()
    {
        for (unsigned i = 0; i < CELL_TYPE::nanoSteps(); i++)
            nanoStep(i);

        this->stepNum++;    
        // call back all registered Writers
        for(unsigned i = 0; i < this->writers.size(); i++) 
            this->writers[i]->stepFinished();
    }

    /**
     * performs step() until the maximum number of steps is reached.
     */
    virtual void run()
    {
        this->initializer->grid(curGrid);
        this->stepNum = 0;
        for(unsigned i = 0; i < this->writers.size(); i++) 
            this->writers[i]->initialized();

        for (this->stepNum = this->initializer->startStep(); 
             this->stepNum < this->initializer->maxSteps();) 
            step();

        for(unsigned i = 0; i < this->writers.size(); i++) 
            this->writers[i]->allDone();        
    }

    /**
     * Returns the current grid.
     */
    virtual const GridType *getGrid()
    {
        return curGrid;
    }

protected:
    GridType *curGrid;
    GridType *newGrid;

    void nanoStep(const unsigned& nanoStep)
    {
        CoordBox<DIMENSIONS> box = curGrid->boundingBox();

        for(typename CoordBox<DIMENSIONS>::Iterator i = box.begin(); i != box.end(); ++i) {
            CoordMap<CELL_TYPE, GridType> neighborhood = 
                curGrid->getNeighborhood(*i);
            (*newGrid)[*i].update(neighborhood, nanoStep);
        }

        std::swap(curGrid, newGrid);
    }
};

}

#endif
