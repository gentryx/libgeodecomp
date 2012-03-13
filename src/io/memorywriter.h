#ifndef _libgeodecomp_io_memorywriter_h_
#define _libgeodecomp_io_memorywriter_h_

#include <libgeodecomp/misc/grid.h>
#include <libgeodecomp/misc/supervector.h>
#include <libgeodecomp/parallelization/simulator.h>
#include <libgeodecomp/io/writer.h>

namespace LibGeoDecomp {

/**
 * The MemoryWriter is good for debugging a Simulator. As it name
 * says, it will simply store all grids in memory for later inspection.
 */
template<typename CELL_TYPE>
class MemoryWriter : public Writer<CELL_TYPE>
{

public:
    typedef typename Simulator<CELL_TYPE>::GridType GridType;

    MemoryWriter(MonolithicSimulator<CELL_TYPE>* sim, int period = 1) : 
        Writer<CELL_TYPE>("foobar", sim, period) 
    {}
    
    void initialized()
    {
        saveGrid(); 
    }

    void stepFinished()
    {
        if ((this->sim->getStep() % this->period) == 0) 
            saveGrid();
    }
    
    GridType& getGrid(int i)
    {
        return grids[i];
    }

    void allDone() 
    { 
        saveGrid(); 
    }
    
    SuperVector<GridType> getGrids()
    {
        return grids;
    }

private:

    SuperVector<GridType> grids;

    void saveGrid()
    {
        grids.push_back(*(this->sim->getGrid()));
    }
};

};

#endif
