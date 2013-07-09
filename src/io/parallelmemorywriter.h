#include <libgeodecomp/config.h>
#ifdef LIBGEODECOMP_FEATURE_MPI
#ifndef LIBGEODECOMP_IO_PARALLELMEMORYWRITER_H
#define LIBGEODECOMP_IO_PARALLELMEMORYWRITER_H

#include <libgeodecomp/io/parallelwriter.h>
#include <libgeodecomp/mpilayer/mpilayer.h>

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
    static const int DIM = CELL_TYPE::Topology::DIM;
    typedef DisplacedGrid<CELL_TYPE, typename CELL_TYPE::Topology> GridType;
    typedef typename ParallelWriter<CELL_TYPE>::GridType WriterGridType;
    typedef SuperMap<unsigned, GridType> GridMap;
    using ParallelWriter<CELL_TYPE>::period;

    ParallelMemoryWriter(
        int period = 1,
        MPI::Comm *communicator = &MPI::COMM_WORLD) :
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

        grids[step].paste(grid, validRegion);
        grids[step].atEdge() = grid.atEdge();

        for (std::size_t sender = 0; sender < mpiLayer.size(); ++sender) {
            for (std::size_t receiver = 0; receiver < mpiLayer.size(); ++receiver) {
                if (sender != receiver) {
                    sendRecvGrid(sender, receiver, grid, validRegion, step);
                }
            }
        }

        mpiLayer.waitAll();
    }

    void sendRecvGrid(int sender, int receiver, const WriterGridType& grid, const Region<DIM>& validRegion, int step)
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

    SuperMap<unsigned, GridType> getGrids()
    {
        return grids;
    }

private:
    SuperMap<unsigned, GridType> grids;
    MPILayer mpiLayer;

};

}

#endif
#endif
