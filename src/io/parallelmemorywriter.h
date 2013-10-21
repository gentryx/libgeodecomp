#include <libgeodecomp/config.h>
#ifdef LIBGEODECOMP_FEATURE_MPI
#ifndef LIBGEODECOMP_IO_PARALLELMEMORYWRITER_H
#define LIBGEODECOMP_IO_PARALLELMEMORYWRITER_H

#include <libgeodecomp/communication/mpilayer.h>
#include <libgeodecomp/io/parallelwriter.h>

namespace LibGeoDecomp {

/**
 * The ParallelMemoryWriter is the MemoryWriters's cousin which is
 * compatible with a DistributedSimulator. Useful for debugging,
 * nothing else.
 */
template<typename CELL_TYPE>
class ParallelMemoryWriter : public ParallelWriter<CELL_TYPE>
{

public:
    typedef typename APITraits::SelectTopology<CELL_TYPE>::Value Topology;
    typedef DisplacedGrid<CELL_TYPE, Topology> GridType;
    typedef typename ParallelWriter<CELL_TYPE>::GridType WriterGridType;
    typedef std::map<unsigned, GridType> GridMap;
    using ParallelWriter<CELL_TYPE>::period;
    static const int DIM = Topology::DIM;

    ParallelMemoryWriter(
        int period = 1,
        MPI_Comm communicator = MPI_COMM_WORLD) :
        ParallelWriter<CELL_TYPE>("", period),
        mpiLayer(communicator, MPILayer::PARALLEL_MEMORY_WRITER)
    {}

    virtual void stepFinished(
        const WriterGridType& grid,
        const Region<DIM>& validRegion,
        const Coord<DIM>& globalDimensions,
        unsigned step,
        WriterEvent event,
        std::size_t rank,
        bool lastCall)
    {
        if ((event == WRITER_STEP_FINISHED) && (step % period != 0)) {
            return;
        }

        if (grids[step].getDimensions() != globalDimensions) {
            grids[step].resize(CoordBox<DIM>(Coord<DIM>(), globalDimensions));
        }

        CoordBox<DIM> box = grid.boundingBox();
        GridType localGrid(box);
        for (typename CoordBox<DIM>::Iterator i = box.begin(); i != box.end(); ++i) {
            localGrid[*i] = grid.get(*i);
        }

        grids[step].paste(grid, validRegion);
        grids[step].setEdge(grid.getEdge());

        for (std::size_t sender = 0; sender < mpiLayer.size(); ++sender) {
            for (std::size_t receiver = 0; receiver < mpiLayer.size(); ++receiver) {
                if (sender != receiver) {
                    sendRecvGrid(sender, receiver, localGrid, validRegion, step);
                }
            }
        }

        mpiLayer.waitAll();
    }

    void sendRecvGrid(size_t sender, size_t receiver, const GridType& grid, const Region<DIM>& validRegion, int step)
    {
        if (sender == mpiLayer.rank()) {
            mpiLayer.sendRegion(validRegion, receiver);
            mpiLayer.sendUnregisteredRegion(
                &grid,
                validRegion,
                receiver,
                MPILayer::PARALLEL_MEMORY_WRITER,
                Typemaps::lookup<CELL_TYPE>());
        }
        if (receiver == mpiLayer.rank()) {
            Region<DIM> recvRegion;
            mpiLayer.recvRegion(&recvRegion, sender);
            mpiLayer.recvUnregisteredRegion(
                &grids[step],
                recvRegion,
                sender,
                MPILayer::PARALLEL_MEMORY_WRITER,
                Typemaps::lookup<CELL_TYPE>());
        }
    }

    GridType& getGrid(int i)
    {
        return grids[i];
    }

    std::map<unsigned, GridType> getGrids()
    {
        return grids;
    }

private:
    std::map<unsigned, GridType> grids;
    MPILayer mpiLayer;

};

}

#endif
#endif
