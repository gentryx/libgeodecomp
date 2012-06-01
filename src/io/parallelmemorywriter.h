#include <libgeodecomp/config.h>
#ifdef LIBGEODECOMP_FEATURE_MPI
#ifndef _libgeodecomp_io_parallelmemorywriter_h_
#define _libgeodecomp_io_parallelmemorywriter_h_

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
    static const int DIM = CELL_TYPE::Topology::DIMENSIONS;
    typedef DisplacedGrid<CELL_TYPE, typename CELL_TYPE::Topology> GridType;
    typedef typename DistributedSimulator<CELL_TYPE>::GridType SimulatorGridType;
    typedef SuperMap<unsigned, GridType> GridMap;

    using ParallelWriter<CELL_TYPE>::distSim;
    using ParallelWriter<CELL_TYPE>::period;

    ParallelMemoryWriter(
        DistributedSimulator<CELL_TYPE>* sim, 
        int period = 1,
        MPI::Comm *communicator = &MPI::COMM_WORLD) : 
        ParallelWriter<CELL_TYPE>("foobar", sim, period),
        boundingBox(distSim->getInitializer()->gridBox()),
        mpiLayer(communicator, MPILayer::PARALLEL_MEMORY_WRITER)
    {}
    
    void initialized()
    {
        saveGrid(); 
    }

    void stepFinished()
    {
        if ((distSim->getStep() % period) == 0) 
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
    
    SuperMap<unsigned, GridType> getGrids()
    {
        return grids;
    }

private:
    CoordBox<DIM> boundingBox;
    SuperMap<unsigned, GridType> grids;
    MPILayer mpiLayer;

    void saveGrid()
    {
        unsigned step = distSim->getStep();

        if (grids[step].boundingBox() != boundingBox) {
            grids[step].resize(boundingBox);
        }

        const SimulatorGridType *grid;
        const Region<DIM> *region;
        distSim->getGridFragment(&grid, &region);

        grids[step].pasteGridBase(*grid, *region);
        grids[step].atEdge() = grid->atEdge();

        for (int sender = 0; sender < mpiLayer.size(); ++sender) {
            for (int receiver = 0; receiver < mpiLayer.size(); ++receiver) {
                mpiLayer.barrier();

                if (sender != receiver) {
                    if (sender == mpiLayer.rank()) {
                        mpiLayer.sendRegion(*region, receiver);
                        mpiLayer.sendUnregisteredRegion(
                            grid, 
                            *region, 
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
            }
        }
    }
};

};

#endif
#endif
