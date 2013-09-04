#include <libgeodecomp/config.h>
#ifdef LIBGEODECOMP_FEATURE_MPI
#ifndef LIBGEODECOMP_IO_MPIIO_H
#define LIBGEODECOMP_IO_MPIIO_H

#include <mpi.h>

#include <libgeodecomp/misc/region.h>
#include <libgeodecomp/loadbalancer/randombalancer.h>
#include <libgeodecomp/mpilayer/typemaps.h>
#include <libgeodecomp/mpilayer/mpilayer.h>

namespace LibGeoDecomp {

template<
    typename CELL_TYPE,
    typename TOPOLOGY = typename APITraits::SelectTopology<CELL_TYPE>::Value
>
class MPIIO
{
public:
    template<typename GRID_TYPE, int DIM>
    static void readRegion(
        GRID_TYPE *grid,
        const std::string& filename,
        const Region<DIM>& region,
        const MPI::Intracomm& comm = MPI::COMM_WORLD,
        const MPI::Datatype& mpiDatatype = Typemaps::lookup<CELL_TYPE>())
    {
        MPI::File file = openFileForRead(filename, comm);
        Coord<DIM> dimensions = getDimensions<DIM>(&file);
        MPI::Aint headerLength;
        MPI::Aint cellLength;
        getLengths<DIM>(&headerLength, &cellLength, mpiDatatype);

        // edge cell is the last element of the header:
        file.Seek(headerLength - cellLength, MPI_SEEK_SET);
        CELL_TYPE cell;
        file.Read(&cell, 1, mpiDatatype);
        grid->setEdge(cell);

        for (typename Region<DIM>::StreakIterator i = region.beginStreak();
             i != region.endStreak();
             ++i) {
            // the coords need to be normalized because on torus
            // topologies the coordnates may exceed the bounding box
            // (especially negative coordnates may occurr).
            Coord<DIM> coord = TOPOLOGY::normalize(i->origin, dimensions);
            file.Seek(
                offset(headerLength, coord, dimensions, cellLength),
                MPI_SEEK_SET);
            int length = i->endX - i->origin.x();
            SuperVector<CELL_TYPE> vec(length);

            file.Read(&vec[0], length, mpiDatatype);

            Coord<DIM> c = i->origin;
            for (int index = 0; index < length; ++index) {
                grid->set(c, vec[index]);
                ++c.x();
            }
        }

        file.Close();
    }

    template<int DIM>
    static void readMetadata(
        Coord<DIM> *dimensions,
        unsigned *step,
        unsigned *maxSteps,
        const std::string& filename,
        const MPI::Intracomm& comm = MPI::COMM_WORLD)
    {
        MPI::File file = openFileForRead(filename, comm);
        *dimensions = getDimensions<DIM>(&file);
        file.Read(step,     1, MPI::UNSIGNED);
        file.Read(maxSteps, 1, MPI::UNSIGNED);
        file.Close();
    }

    template<typename GRID_TYPE, int DIM>
    static void writeRegion(
        const GRID_TYPE& grid,
        const Coord<DIM>& dimensions,
        const unsigned& step,
        const unsigned& maxSteps,
        const std::string& filename,
        const Region<DIM>& region,
        const MPI::Datatype& mpiDatatype = Typemaps::lookup<CELL_TYPE>(),
        const MPI::Intracomm& comm = MPI::COMM_WORLD)
    {
        MPI::File file = openFileForWrite(filename, comm);
        MPI::Aint headerLength;
        MPI::Aint cellLength;
        getLengths<DIM>(&headerLength, &cellLength, mpiDatatype);

        if (comm.Get_rank() == 0) {
            CELL_TYPE cell = grid.getEdge();
            file.Write(&dimensions, 1, Typemaps::lookup<Coord<DIM> >());
            file.Write(&step,       1, MPI::UNSIGNED);
            file.Write(&maxSteps,   1, MPI::UNSIGNED);
            file.Write(&cell,       1, mpiDatatype);
        }

        for (typename Region<DIM>::StreakIterator i = region.beginStreak();
             i != region.endStreak();
             ++i) {
            // the coords need to be normalized because on torus
            // topologies the coordnates may exceed the bounding box
            // (especially negative coordnates may occurr).
            Coord<DIM> coord = TOPOLOGY::normalize(i->origin, dimensions);
            file.Seek(offset(headerLength, coord, dimensions, cellLength),
                      MPI_SEEK_SET);
            int length = i->endX - i->origin.x();
            SuperVector<CELL_TYPE> vec(length);

            Coord<DIM> c = i->origin;
            for (int index = 0; index < length; ++index) {
                vec[index] = grid.get(c);
                ++c.x();
            }
            file.Write(&vec[0], length, mpiDatatype);
        }

        file.Close();
    }

    static MPI::File openFileForRead(
        const std::string& filename,
        const MPI::Intracomm& comm)
    {
        MPI::File file = MPI::File::Open(
            comm, filename.c_str(),
            MPI_MODE_RDONLY, MPI::INFO_NULL);
        file.Set_errhandler(MPI::ERRORS_ARE_FATAL);
        return file;
    }


    static MPI::File openFileForWrite(
        const std::string& filename,
        const MPI::Intracomm& comm)
    {
        MPI::File file = MPI::File::Open(
            comm, filename.c_str(),
            MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI::INFO_NULL);
        file.Set_errhandler(MPI::ERRORS_ARE_FATAL);
        return file;
    }

    static MPI::Aint getLength(const MPI::Datatype& datatype)
    {
        MPI::Aint length;
        MPI::Aint lowerBound;
        datatype.Get_extent(lowerBound, length);
        return length;
    }

private:

    template<int DIM>
    static MPI::Offset offset(
        const MPI::Offset& headerLength,
        const Coord<DIM>& c,
        const Coord<DIM>& dimensions,
        const MPI::Aint& cellLength)
    {
        return headerLength + c.toIndex(dimensions) * cellLength;
    }

    template<int DIM>
    static void getLengths(
        MPI::Aint *headerLength,
        MPI::Aint *cellLength,
        const MPI::Datatype& mpiDatatype)
    {
        MPI::Aint coordLength = getLength(Typemaps::lookup<Coord<DIM> >());
        MPI::Aint unsignedLength = getLength(MPI::UNSIGNED);
        *cellLength =  getLength(mpiDatatype);
        *headerLength = coordLength + 2 * unsignedLength + *cellLength;
    }

    template<int DIM>
    static Coord<DIM> getDimensions(MPI::File *file)
    {
        Coord<DIM> ret;
        file->Read(&ret, 1, Typemaps::lookup<Coord<DIM> >());
        return ret;
    }

};

}

#endif
#endif
