#ifndef LIBGEODECOMP_IO_BOVWRITER_H
#define LIBGEODECOMP_IO_BOVWRITER_H

#include <libgeodecomp/config.h>
#ifdef LIBGEODECOMP_WITH_MPI

#include <libgeodecomp/communication/typemaps.h>
#include <libgeodecomp/io/mpiio.h>
#include <libgeodecomp/io/parallelwriter.h>
#include <libgeodecomp/io/selector.h>
#include <libgeodecomp/misc/clonable.h>

#include <iomanip>

namespace LibGeoDecomp {

/**
 * writes simulation snapshots compatible with VisIt's Brick of Values
 * (BOV) format using MPI-IO. Uses a selector which maps a cell to a
 * primitive data type so that it can be fed into VisIt or ParaView.
 */
template<typename CELL_TYPE>
class BOVWriter : public Clonable<ParallelWriter<CELL_TYPE>, BOVWriter<CELL_TYPE> >
{
public:
    friend class BOVWriterTest;

    using ParallelWriter<CELL_TYPE>::period;
    using ParallelWriter<CELL_TYPE>::prefix;

    typedef typename ParallelWriter<CELL_TYPE>::Topology Topology;

    static const int DIM = Topology::DIM;

    BOVWriter(
        const Selector<CELL_TYPE>& selector,
        const std::string& prefix,
        const unsigned period,
        const Coord<3>& brickletDim = Coord<3>(),
        const MPI_Comm& communicator = MPI_COMM_WORLD) :
        Clonable<ParallelWriter<CELL_TYPE>, BOVWriter<CELL_TYPE> >(prefix, period),
        selector(selector),
        brickletDim(brickletDim),
        comm(communicator),
        datatype(selector.mpiDatatype())
    {}

    virtual void stepFinished(
        const typename ParallelWriter<CELL_TYPE>::GridType& grid,
        const Region<Topology::DIM>& validRegion,
        const Coord<Topology::DIM>& globalDimensions,
        unsigned step,
        WriterEvent event,
        std::size_t rank,
        bool lastCall)
    {
        if ((event == WRITER_STEP_FINISHED) && (step % period != 0)) {
            return;
        }

        writeHeader(step, globalDimensions);
        writeRegion(step, globalDimensions, grid, validRegion);
    }


private:
    Selector<CELL_TYPE> selector;
    Coord<3> brickletDim;
    MPI_Comm comm;
    MPI_Datatype datatype;

    std::string filename(const unsigned& step, const std::string& suffix) const
    {
        std::ostringstream buf;
        buf << prefix << "." << std::setfill('0') << std::setw(5) << step << "." << suffix;
        return buf.str();
    }

    void writeHeader(const unsigned& step, const Coord<DIM>& dimensions)
    {
        MPI_File file = MPIIO<CELL_TYPE, Topology>::openFileForWrite(
            filename(step, "bov"), comm);
        int rank;
        MPI_Comm_rank(comm, &rank);

        if (rank == 0) {
            // BOV only accepts 3D data, so we'll have to inflate 1D
            // and 2D dimensions.
            Coord<DIM> c = dimensions;
            Coord<3> bovDim = Coord<3>::diagonal(1);
            for (int i = 0; i < DIM; ++i) {
                bovDim[i] = c[i];
            }

            Coord<3> bricDim = (brickletDim == Coord<3>()) ? bovDim : brickletDim;

            std::ostringstream buf;
            buf << "TIME: " << step << "\n"
                << "DATA_FILE: " << filename(step, "data") << "\n"
                << "DATA_SIZE: "
                << bovDim.x() << " " << bovDim.y() << " " << bovDim.z() << "\n"
                << "DATA_FORMAT: " << selector.typeName() << "\n"
                << "VARIABLE: " << selector.name() << "\n"
                << "DATA_ENDIAN: LITTLE\n"
                << "BRICK_ORIGIN: 0 0 0\n"
                << "BRICK_SIZE: "
                << bovDim.x() << " " << bovDim.y() << " " << bovDim.z() << "\n"
                << "DIVIDE_BRICK: true\n"
                << "DATA_BRICKLETS: "
                << bricDim.x() << " " << bricDim.y() << " " << bricDim.z() << "\n"
                << "DATA_COMPONENTS: " << selector.arity() << "\n";
            std::string s = buf.str();
            MPI_File_write(file, const_cast<char*>(s.c_str()), s.length(), MPI_CHAR, MPI_STATUS_IGNORE);
        }

        MPI_File_close(&file);
    }

    template<typename GRID_TYPE>
    void writeRegion(
        const unsigned& step,
        const Coord<DIM>& dimensions,
        const GRID_TYPE& grid,
        const Region<DIM>& region)
    {
        MPI_File file = MPIIO<CELL_TYPE, Topology>::openFileForWrite(
            filename(step, "data"), comm);
        MPI_Aint varLength = MPIIO<CELL_TYPE, Topology>::getLength(datatype);
        std::vector<char> buffer;

        for (typename Region<DIM>::StreakIterator i = region.beginStreak();
             i != region.endStreak();
             ++i) {
            // the coords need to be normalized because on torus
            // topologies the coordnates may exceed the bounding box
            // (especially negative coordnates may occurr).
            Coord<DIM> coord = Topology::normalize(i->origin, dimensions);
            int dataComponents = selector.arity();
            MPI_Offset index = coord.toIndex(dimensions) * varLength * dataComponents;
            MPI_File_seek(file, index, MPI_SEEK_SET);
            int length = i->endX - i->origin.x();
            std::size_t byteSize = length * selector.sizeOfExternal();

            if (buffer.size() != byteSize) {
                buffer.resize(byteSize);
            }

            Region<DIM> tempRegion;
            tempRegion << *i;
            grid.saveMemberUnchecked(&buffer[0], selector, tempRegion);

            MPI_File_write(file, &buffer[0], length, datatype, MPI_STATUS_IGNORE);
        }

        MPI_File_close(&file);
    }
};

}

#endif
#endif
