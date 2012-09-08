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
        Coord<DIMENSIONS> dim = initializer->gridBox().dimensions;
        curGrid = new GridType(dim);
        newGrid = new GridType(dim);
        initializer->grid(curGrid);
        initializer->grid(newGrid);

        CoordBox<DIMENSIONS> box = curGrid->boundingBox();
        unsigned endX = box.dimensions.x();
        box.dimensions.x() = 1;
        for(typename CoordBox<DIMENSIONS>::Iterator i = box.begin(); i != box.end(); ++i) {
            simArea << Streak<DIMENSIONS>(*i, endX);
        }
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
        // notify all registered Steerers
        for(unsigned i = 0; i < steerers.size(); ++i) {
            if (stepNum % steerers[i]->getPeriod() == 0) {
                steerers[i]->nextStep(curGrid, simArea, stepNum);
            }
        }

        for (unsigned i = 0; i < CELL_TYPE::nanoSteps(); ++i) {
            nanoStep(i);
        }

        ++stepNum; 

        // call back all registered Writers
        for(unsigned i = 0; i < writers.size(); ++i) {
            if (stepNum % writers[i]->getPeriod() == 0) {
                writers[i]->stepFinished();
            }
        }
    }

    /**
     * continue simulating until the maximum number of steps is reached.
     */
    virtual void run()
    {
        initializer->grid(curGrid);
        stepNum = 0;
        for(unsigned i = 0; i < writers.size(); ++i) {
            writers[i]->initialized();
        }

        for (stepNum = initializer->startStep(); 
             stepNum < initializer->maxSteps();) {
            step();
        }

        for(unsigned i = 0; i < writers.size(); ++i) {
            writers[i]->allDone();        
        }
    }

    /**
     * returns the current grid.
     */
    virtual const GridType *getGrid()
    {
        return curGrid;
    }

protected:
    using MonolithicSimulator<CELL_TYPE>::initializer;
    using MonolithicSimulator<CELL_TYPE>::steerers;
    using MonolithicSimulator<CELL_TYPE>::stepNum;
    using MonolithicSimulator<CELL_TYPE>::writers;

    GridType *curGrid;
    GridType *newGrid;
    Region<DIMENSIONS> simArea;

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
