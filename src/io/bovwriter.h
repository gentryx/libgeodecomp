#include <libgeodecomp/config.h>
#ifdef LIBGEODECOMP_FEATURE_MPI
#ifndef _libgeodecomp_io_bovwriter_h_
#define _libgeodecomp_io_bovwriter_h_

#include <iomanip>

#include <libgeodecomp/io/mpiio.h>
#include <libgeodecomp/io/parallelwriter.h>
#include <libgeodecomp/mpilayer/typemaps.h>

namespace LibGeoDecomp {

/**
 * writes simulation snapshots compatible with VisIt's Brick of Values
 * (BOV) format using MPI-IO. 
 */
template<typename CELL_TYPE, typename SELECTOR_TYPE>
class BOVWriter : public ParallelWriter<CELL_TYPE>
{    
public:
    friend class BOVWriterTest;

    typedef typename CELL_TYPE::Topology Topology;
    typedef typename SELECTOR_TYPE::VariableType VariableType;

    static const int DIM = CELL_TYPE::Topology::DIMENSIONS;

    BOVWriter(
        const std::string& prefix, 
        DistributedSimulator<CELL_TYPE> *sim, 
        const unsigned& period, 
        const Coord<3>& brickletDim = Coord<3>(),
        const MPI::Intracomm& communicator = MPI::COMM_WORLD,
        MPI::Datatype mpiDatatype = Typemaps::lookup<VariableType>()) :
        ParallelWriter<CELL_TYPE>(prefix, sim, period),
        comm(communicator),
        datatype(mpiDatatype)
    {
        // BOV only accepts 3D data, so we'll have to inflate 1D and
        // 2D dimensions.
        Coord<DIM> c = this->distSim->getInitializer()->gridDimensions();
        Coord<3> initDim = Coord<3>::diagonal(1);
        for (int i = 0; i < DIM; ++i)
            initDim.c[i] = c.c[i]; 
        bricDim = (brickletDim == Coord<3>()) ? initDim : brickletDim;
    }

    virtual void initialized() 
    {
        writeGrid();
    }

    virtual void stepFinished()
    {
        if ((this->distSim->getStep() % this->period) == 0) 
            writeGrid();
    }

    virtual void allDone()
    {
        writeGrid();
    }

private:
    MPI::Intracomm comm;
    MPI::Datatype datatype;
    Coord<3> bricDim;

    std::string filename(const unsigned& step, const std::string& suffix) const 
    {
        std::ostringstream buf;
        buf << this->prefix << "." << std::setfill('0') << std::setw(5) << step << "." << suffix;
        return buf.str();
    }

    void writeGrid()
    {
        const Region<DIM> *region;
        const typename DistributedSimulator<CELL_TYPE>::GridType *grid;
        this->distSim->getGridFragment(&grid, &region);
        unsigned step = this->distSim->getStep();
        Coord<DIM> dimensions = 
            this->distSim->getInitializer()->gridDimensions();

        writeHeader(step, dimensions);
        writeRegion(step, dimensions, grid, *region);
    }

    void writeHeader(const unsigned& step, const Coord<DIM>& dimensions)
    {
        MPI::File file = MPIIO<CELL_TYPE, Topology>::openFileForWrite(
            this->filename(step, "bov"), comm);

        if (comm.Get_rank() == 0) {
            // BOV only accepts 3D data, so we'll have to inflate 1D
            // and 2D dimensions.
            Coord<DIM> c = this->distSim->getInitializer()->gridDimensions();
            Coord<3> bovDim = Coord<3>::diagonal(1);
            for (int i = 0; i < DIM; ++i)
                bovDim.c[i] = c.c[i]; 

            std::ostringstream buf;
            buf << "TIME: " << step << "\n"
                << "DATA_FILE: " << this->filename(step, "data") << "\n"
                << "DATA_SIZE: " 
                << bovDim.x() << " " << bovDim.y() << " " << bovDim.z() << "\n"
                << "DATA_FORMAT: " << SELECTOR_TYPE::dataFormat() << "\n"
                << "VARIABLE: " << SELECTOR_TYPE::varName() << "\n"
                << "DATA_ENDIAN: LITTLE\n"
                << "BRICK_ORIGIN: 0 0 0\n"
                << "BRICK_SIZE: " 
                << bovDim.x() << " " << bovDim.y() << " " << bovDim.z() << "\n"
                << "DIVIDE_BRICK: true\n"
                << "DATA_BRICKLETS: " 
                << bricDim.x() << " " << bricDim.y() << " " << bricDim.z() << "\n"
                << "DATA_COMPONENTS: " << SELECTOR_TYPE::dataComponents() << "\n";
            std::string s = buf.str();
            file.Write(s.c_str(), s.length(), MPI::CHAR);
        }

        file.Close();
    }

    template<typename GRID_TYPE>
    void writeRegion(
        const unsigned& step, 
        const Coord<DIM>& dimensions, 
        GRID_TYPE *grid, 
        const Region<DIM>& region)
    {
        MPI::File file = MPIIO<CELL_TYPE, Topology>::openFileForWrite(
            this->filename(step, "data"), comm);
        MPI::Aint varLength = MPIIO<CELL_TYPE, Topology>::getLength(datatype);
        SuperVector<VariableType> buffer;

        for (StreakIterator<DIM> i = region.beginStreak();
             i != region.endStreak();
             ++i) {
            // the coords need to be normalized because on torus
            // topologies the coordnates may exceed the bounding box
            // (especially negative coordnates may occurr).
            Coord<DIM> coord = Topology::normalize(i->origin, dimensions);
            int dataComponents = SELECTOR_TYPE::dataComponents();
            MPI::Offset index = 
                CoordToIndex<DIM>()(coord, dimensions) * varLength * dataComponents;
            file.Seek(index, MPI_SEEK_SET);
            int length = i->endX - i->origin.x();
            int effectiveLength = length * dataComponents;
            Coord<DIM> walker = i->origin;
            
            if (buffer.size() != effectiveLength)
                buffer = SuperVector<VariableType>(effectiveLength);
            for (int i = 0; i < effectiveLength; i += dataComponents) {
                SELECTOR_TYPE()(grid->at(walker), &buffer[i]);
                walker.x()++;
            }
            file.Write(&buffer[0], effectiveLength, datatype);
        }

        file.Close();
    }
};

}

#endif
#endif
