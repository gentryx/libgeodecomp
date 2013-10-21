#ifndef LIBGEODECOMP_IO_MEMORYWRITER_H
#define LIBGEODECOMP_IO_MEMORYWRITER_H

#include <libgeodecomp/io/writer.h>
#include <libgeodecomp/misc/stdcontaineroverloads.h>
#include <libgeodecomp/storage/grid.h>

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
    typedef typename APITraits::SelectTopology<CELL_TYPE>::Value Topology;
    typedef Grid<CELL_TYPE, Topology> StorageGrid;
    using Writer<CELL_TYPE>::period;

    MemoryWriter(unsigned period = 1) :
        Writer<CELL_TYPE>("", period)
    {}

    virtual void stepFinished(const GridType& grid, unsigned step, WriterEvent event)
    {
        if ((event == WRITER_STEP_FINISHED) && (step % period != 0)) {
            return;
        }

        grids.push_back(StorageGrid(grid));
    }

    GridType& getGrid(int i)
    {
        return grids[i];
    }

    std::vector<StorageGrid>& getGrids()
    {
        return grids;
    }

private:
    std::vector<StorageGrid> grids;
};

}

#endif
