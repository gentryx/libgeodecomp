#ifndef LIBGEODECOMP_IO_COLLECTINGWRITER_H
#define LIBGEODECOMP_IO_COLLECTINGWRITER_H

#include <libgeodecomp/config.h>
#ifdef LIBGEODECOMP_WITH_MPI

#include <libgeodecomp/communication/mpilayer.h>
#include <libgeodecomp/io/parallelwriter.h>
#include <libgeodecomp/misc/clonable.h>
#include <libgeodecomp/storage/gridtypeselector.h>

namespace LibGeoDecomp {

/**
 * Adapter class whose purpose is to use legacy Writer objects
 * together with a DistributedSimulator. Good for testing, but doesn't
 * scale, as all memory is concentrated on one node and IO is
 * serialized to that node. Use with care!
 */
template<typename CELL_TYPE>
class CollectingWriter : public Clonable<ParallelWriter<CELL_TYPE>, CollectingWriter<CELL_TYPE> >
{
public:
    typedef typename ParallelWriter<CELL_TYPE>::Topology Topology;
    typedef typename APITraits::SelectSoA<CELL_TYPE>::Value SupportsSoA;
    typedef typename GridTypeSelector<CELL_TYPE, Topology, false, SupportsSoA>::Value StorageGridType;
    typedef typename SerializationBuffer<CELL_TYPE>::BufferType BufferType;
    typedef typename DistributedSimulator<CELL_TYPE>::GridType SimulatorGridType;

    using ParallelWriter<CELL_TYPE>::period;

    static const int DIM = Topology::DIM;

    explicit CollectingWriter(
        Writer<CELL_TYPE> *writer,
        int root = 0,
        MPI_Comm communicator = MPI_COMM_WORLD,
        MPI_Datatype mpiDatatype = SerializationBuffer<CELL_TYPE>::cellMPIDataType()) :
        Clonable<ParallelWriter<CELL_TYPE>, CollectingWriter<CELL_TYPE> >("",  1),
        writer(writer),
        mpiLayer(communicator),
        root(root),
        datatype(mpiDatatype)
    {
        if ((mpiLayer.rank() != root) && (writer != 0)) {
            throw std::invalid_argument("can't call back a writer on a node other than the root");
        }

        if (mpiLayer.rank() == root) {
            if (writer == 0) {
                throw std::invalid_argument("delegate writer on root must not be null");
            }

            period = writer->getPeriod();
        }

        period = mpiLayer.broadcast(period, root);
    }

    virtual void stepFinished(
        const SimulatorGridType& grid,
        const Region<DIM>& validRegion,
        const Coord<DIM>& globalDimensions,
        unsigned step,
        WriterEvent event,
        std::size_t rank,
        bool lastCall)
    {
        SerializationBuffer<CELL_TYPE>::resize(&buffer, validRegion);
        grid.saveRegion(&buffer, validRegion);

        if (mpiLayer.rank() == root) {
            if (globalGrid.boundingBox().dimensions != globalDimensions) {
                Region<DIM> region;
                region << CoordBox<DIM>(Coord<DIM>(), globalDimensions);
                globalGrid = StorageGridType(region);
            }

            globalGrid.loadRegion(buffer, validRegion);
            globalGrid.setEdge(grid.getEdge());
        }

        for (int sender = 0; sender < mpiLayer.size(); ++sender) {
            if (sender != root) {
                if (mpiLayer.rank() == root) {
                    Region<DIM> recvRegion;
                    mpiLayer.recvRegion(&recvRegion, sender);
                    SerializationBuffer<CELL_TYPE>::resize(&buffer, recvRegion);

                    mpiLayer.recv(
                        buffer.data(),
                        sender,
                        buffer.size(),
                        MPILayer::COLLECTING_WRITER,
                        SerializationBuffer<CELL_TYPE>::cellMPIDataType());
                    mpiLayer.waitAll();
                    globalGrid.loadRegion(buffer, recvRegion);
                }
                if (mpiLayer.rank() == sender) {
                    mpiLayer.sendRegion(validRegion, root);
                    mpiLayer.send(
                        buffer.data(),
                        root,
                        buffer.size(),
                        MPILayer::COLLECTING_WRITER,
                        SerializationBuffer<CELL_TYPE>::cellMPIDataType());
                }
            }
        }

        mpiLayer.waitAll();

        if (lastCall && (mpiLayer.rank() == root)) {
            writer->stepFinished(globalGrid, step, event);
        }
    }

private:
    boost::shared_ptr<Writer<CELL_TYPE> > writer;
    MPILayer mpiLayer;
    int root;
    StorageGridType globalGrid;
    BufferType buffer;
    MPI_Datatype datatype;
};

}

#endif

#endif
