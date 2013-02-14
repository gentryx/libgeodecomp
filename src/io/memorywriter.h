#ifndef _libgeodecomp_io_memorywriter_h_
#define _libgeodecomp_io_memorywriter_h_

#include <libgeodecomp/misc/grid.h>
#include <libgeodecomp/misc/supervector.h>
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
    typedef typename Writer<CELL_TYPE>::GridType GridType;
    using Writer<CELL_TYPE>::period;

    MemoryWriter(unsigned period = 1) : 
        Writer<CELL_TYPE>("foobar", period) 
    {}
    
    virtual void stepFinished(const GridType& grid, unsigned step, WriterEvent event) 
    {
        if ((event == WRITER_STEP_FINISHED) && (step % period != 0)) {
            return;
        }

        grids.push_back(grid);
    }
        
    GridType& getGrid(int i)
    {
        return grids[i];
    }

    SuperVector<GridType>& getGrids()
    {
        return grids;
    }

private:
    SuperVector<GridType> grids;
};

}

#endif
