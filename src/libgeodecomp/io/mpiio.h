#include <libgeodecomp/config.h>
#ifdef LIBGEODECOMP_WITH_MPI
#ifndef LIBGEODECOMP_IO_MPIIO_H
#define LIBGEODECOMP_IO_MPIIO_H

#include <mpi.h>

#include <libgeodecomp/communication/mpilayer.h>
#include <libgeodecomp/communication/typemaps.h>
#include <libgeodecomp/geometry/region.h>
#include <libgeodecomp/loadbalancer/randombalancer.h>

namespace LibGeoDecomp {

/**
 * Utility class which bundles common MPI-based input/output code.
 */
template<
    typename CELL_TYPE,
    typename TOPOLOGY = typename APITraits::SelectTopology<CELL_TYPE>::Value
>
class MPIIO
{
public:
    template<typename GRID_TYPE, int DIM>
    void readRegion(
        GRID_TYPE *grid,
        const std::string& filename,
        const Region<DIM>& region,
        const MPI_Comm& comm = MPI_COMM_WORLD,
        const MPI_Datatype& mpiDatatype = Typemaps::lookup<CELL_TYPE>())
    {
        MPI_File file = openFileForRead(filename, comm);
        Coord<DIM> dimensions = getDimensions<DIM>(file);
        MPI_Aint headerLength;
        MPI_Aint cellLength;
        getLengths<DIM>(&headerLength, &cellLength, mpiDatatype);

        // edge cell is the last element of the header:
        MPI_File_seek(file, headerLength - cellLength, MPI_SEEK_SET);
        CELL_TYPE cell;
        MPI_File_read(file, &cell, 1, mpiDatatype, MPI_STATUS_IGNORE);
        grid->setEdge(cell);

        for (typename Region<DIM>::StreakIterator i = region.beginStreak();
             i != region.endStreak();
             ++i) {
            // the coords need to be normalized because on torus
            // topologies the coordnates may exceed the bounding box
            // (especially negative coordnates may occurr).
            Coord<DIM> coord = TOPOLOGY::normalize(i->origin, dimensions);
            MPI_File_seek(
                file,
                offset(headerLength, coord, dimensions, cellLength),
                MPI_SEEK_SET);
            int length = i->endX - i->origin.x();
            std::vector<CELL_TYPE> vec(length);

            MPI_File_read(file, &vec[0], length, mpiDatatype, MPI_STATUS_IGNORE);
            grid->set(*i, &vec[0]);
        }

        MPI_File_close(&file);
    }

    template<int DIM>
    void readMetadata(
        Coord<DIM> *dimensions,
        unsigned *step,
        unsigned *maxSteps,
        const std::string& filename,
        const MPI_Comm& comm = MPI_COMM_WORLD)
    {
        MPI_File file = openFileForRead(filename, comm);
        *dimensions = getDimensions<DIM>(file);
        MPI_File_read(file, step,     1, MPI_UNSIGNED, MPI_STATUS_IGNORE);
        MPI_File_read(file, maxSteps, 1, MPI_UNSIGNED, MPI_STATUS_IGNORE);
        MPI_File_close(&file);
    }

    template<typename GRID_TYPE, int DIM>
    void writeRegion(
        const GRID_TYPE& grid,
        const Coord<DIM>& dimensions,
        unsigned step,
        unsigned maxSteps,
        const std::string& filename,
        const Region<DIM>& region,
        const MPI_Datatype& mpiDatatype = Typemaps::lookup<CELL_TYPE>(),
        const MPI_Comm& comm = MPI_COMM_WORLD)
    {
        MPI_File file = openFileForWrite(filename, comm);
        MPI_Aint headerLength = 0;
        MPI_Aint cellLength = 0;
        getLengths<DIM>(&headerLength, &cellLength, mpiDatatype);
        int rank;
        MPI_Comm_rank(comm, &rank);

        if (rank == 0) {
            CELL_TYPE cell = grid.getEdge();
            MPI_File_write(file, const_cast<Coord<DIM>*>(&dimensions),
                           1, Typemaps::lookup<Coord<DIM> >(), MPI_STATUS_IGNORE);

            MPI_File_write(file, const_cast<unsigned*>(&step),
                           1, MPI_UNSIGNED, MPI_STATUS_IGNORE);

            MPI_File_write(file, const_cast<unsigned*>(&maxSteps),
                           1, MPI_UNSIGNED, MPI_STATUS_IGNORE);

            MPI_File_write(file, &cell,
                           1, mpiDatatype,  MPI_STATUS_IGNORE);
        }

        for (typename Region<DIM>::StreakIterator i = region.beginStreak();
             i != region.endStreak();
             ++i) {
            // the coords need to be normalized because on torus
            // topologies the coordnates may exceed the bounding box
            // (especially negative coordnates may occurr).
            Coord<DIM> coord = TOPOLOGY::normalize(i->origin, dimensions);
            MPI_File_seek(
                file,
                offset(headerLength, coord, dimensions, cellLength),
                MPI_SEEK_SET);

            int length = i->endX - i->origin.x();
            std::vector<CELL_TYPE> vec(length);
            grid.get(*i, &vec[0]);

            MPI_File_write(file, &vec[0], length, mpiDatatype, MPI_STATUS_IGNORE);
        }

        MPI_File_close(&file);
    }

    MPI_File openFileForRead(
        const std::string& filename,
        MPI_Comm comm)
    {
        MPI_File file;
        MPI_File_open(
            comm, const_cast<char*>(filename.c_str()),
            MPI_MODE_RDONLY, MPI_INFO_NULL,
            &file);
        MPI_File_set_errhandler(file, MPI_ERRORS_ARE_FATAL);
        return file;
    }


    MPI_File openFileForWrite(
        const std::string& filename,
        MPI_Comm comm)
    {
        MPI_File file;
        int res = MPI_File_open(
            comm, const_cast<char*>(filename.c_str()),
            MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL,
            &file);
	if (res != 0) {
	    char buf[MPI_MAX_ERROR_STRING];
	    int length;
	    MPI_Error_string(res, buf, &length);
	    std::ostringstream temp;
	    temp << "MPI_File_open() failed with error \"" << buf << "\"";
	    throw std::runtime_error(temp.str());
	}
        MPI_File_set_errhandler(file, MPI_ERRORS_ARE_FATAL);
        return file;
    }

    MPI_Aint getLength(const MPI_Datatype& datatype)
    {
        MPI_Aint length;
        MPI_Aint lowerBound;
        MPI_Type_get_extent(datatype, &lowerBound, &length);
        return length;
    }

private:
    // fixme: use MPILayer for MPI-IO
    MPILayer mpiLayer;

    template<int DIM>
    MPI_Offset offset(
        const MPI_Offset& headerLength,
        const Coord<DIM>& c,
        const Coord<DIM>& dimensions,
        const MPI_Aint& cellLength)
    {
        return headerLength + c.toIndex(dimensions) * cellLength;
    }

    template<int DIM>
    void getLengths(
        MPI_Aint *headerLength,
        MPI_Aint *cellLength,
        const MPI_Datatype& mpiDatatype)
    {
        MPI_Aint coordLength = getLength(Typemaps::lookup<Coord<DIM> >());
        MPI_Aint unsignedLength = getLength(MPI_UNSIGNED);
        *cellLength =  getLength(mpiDatatype);
        *headerLength = coordLength + 2 * unsignedLength + *cellLength;
    }

    template<int DIM>
    Coord<DIM> getDimensions(MPI_File file)
    {
        Coord<DIM> ret;
        MPI_File_read(file, &ret, 1, Typemaps::lookup<Coord<DIM> >(), MPI_STATUS_IGNORE);
        return ret;
    }

};

}

#endif
#endif
