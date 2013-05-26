#ifndef LIBGEODECOMP_IO_COLLECTINGWRITER_H
#define LIBGEODECOMP_IO_COLLECTINGWRITER_H

#include <libgeodecomp/io/parallelwriter.h>
#include <libgeodecomp/mpilayer/mpilayer.h>

namespace LibGeoDecomp {

/**
 * Adapter class whose purpose is to use legacy Writer objects
 * together with a DistributedSimulator. Good for testing, but doesn't
 * scale, as all memory is concentrated on one node and IO is
 * serialized to that node. Use with care!
 */
template<typename CELL_TYPE>
class CollectingWriter : public ParallelWriter<CELL_TYPE>
{
public:
    typedef typename CELL_TYPE::Topology Topology;
    typedef DisplacedGrid<CELL_TYPE, Topology> StorageGridType;
    typedef typename DistributedSimulator<CELL_TYPE>::GridType SimulatorGridType;

    static const int DIM = Topology::DIM;

    using ParallelWriter<CELL_TYPE>::distSim;

    CollectingWriter(
        Writer<CELL_TYPE> *writer,
        const unsigned period = 1,
        const int root = 0,
        MPI::Comm *communicator = &MPI::COMM_WORLD,
        MPI::Datatype mpiDatatype = Typemaps::lookup<CELL_TYPE>()) :
        ParallelWriter<CELL_TYPE>("",  period),
        writer(writer),
        mpiLayer(communicator),
        root(root),
        datatype(mpiDatatype)
    {
        if ((mpiLayer.rank() != root) && (writer != 0)) {
            throw std::invalid_argument("can't call back a writer on a node other than the root");
        }

        if (writer && (period != writer->getPeriod())) {
            throw std::invalid_argument("period must match delegate's period");
        }
    }

    virtual void stepFinished(
        const SimulatorGridType& grid,
        const Region<DIM>& validRegion,
        const Coord<DIM>& globalDimensions,
        unsigned step,
        WriterEvent event,
        bool lastCall)
    {
        if (mpiLayer.rank() == root) {
            if (globalGrid.boundingBox().dimensions != globalDimensions) {
                globalGrid.resize(CoordBox<DIM>(Coord<DIM>(), globalDimensions));
            }

            globalGrid.pasteGridBase(grid, validRegion);
            globalGrid.atEdge() = grid.atEdge();
        }


        for (int sender = 0; sender < mpiLayer.size(); ++sender) {
            if (sender != root) {
                if (mpiLayer.rank() == root) {
                    Region<DIM> recvRegion;
                    mpiLayer.recvRegion(&recvRegion, sender);
                    mpiLayer.recvUnregisteredRegion(
                        &globalGrid, 
                        recvRegion, 
                        sender, 
                        MPILayer::PARALLEL_MEMORY_WRITER, 
                        datatype);                    
                }
                if (mpiLayer.rank() == sender) {
                    if (sender == mpiLayer.rank()) {
                        mpiLayer.sendRegion(validRegion, root);
                        mpiLayer.sendUnregisteredRegion(
                            &grid, 
                            validRegion, 
                            root, 
                            MPILayer::PARALLEL_MEMORY_WRITER, 
                            datatype);
                    }
                }
            }
        }

        mpiLayer.waitAll();

        if (lastCall && (mpiLayer.rank() == root)) {
            writer->stepFinished(*globalGrid.vanillaGrid(), step, event);
        }
    }

private:
    boost::shared_ptr<Writer<CELL_TYPE> > writer;
    MPILayer mpiLayer;
    int root;
    StorageGridType globalGrid;    
    MPI::Datatype datatype;
};

}

#endif
