#include <libgeodecomp/config.h>
#if defined(LIBGEODECOMP_FEATURE_MPI) || defined(LIBGEODECOMP_FEATURE_HPX)
#ifndef LIBGEODECOMP_IO_BOVWRITERALT_H
#define LIBGEODECOMP_IO_BOVWRITERALT_H

#include <iomanip>

#if defined(LIBGEODECOMP_FEATURE_MPI)
#include <libgeodecomp/io/mpiio.h>
#endif
#include <libgeodecomp/io/parallelwriter.h>
#if defined(LIBGEODECOMP_FEATURE_MPI)
#include <libgeodecomp/mpilayer/typemaps.h>
#endif

namespace LibGeoDecomp {

/**
 * writes simulation snapshots compatible with VisIt's Brick of Values
 * (BOV) format using one file per partition. Uses a selector which maps a cell to a
 * primitive data type so that it can be fed into VisIt.
 */

template<typename CELL_TYPE, typename SELECTOR_TYPE>
class BOVWriterAlt : public ParallelWriter<CELL_TYPE>
{
public:
    friend class BOVWriterAltTest;

    typedef typename CELL_TYPE::Topology Topology;
    typedef typename SELECTOR_TYPE::VariableType VariableType;

    static const int DIM = CELL_TYPE::Topology::DIM;

    using ParallelWriter<CELL_TYPE>::period;
    using ParallelWriter<CELL_TYPE>::prefix;

    BOVWriterAlt() {}

    BOVWriterAlt(
        const std::string& prefix,
        const unsigned period,
        const Coord<3>& brickletDim = Coord<3>()) :
        ParallelWriter<CELL_TYPE>(prefix, period),
        brickletDim(brickletDim)
    {}

    ParallelWriter<CELL_TYPE> * clone()
    {
        return new BOVWriterAlt(this->prefix, this->period, brickletDim);
    }


    virtual void stepFinished(
        const typename ParallelWriter<CELL_TYPE>::GridType& grid,
        const Region<Topology::DIM>& validRegion,
        const Coord<Topology::DIM>& globalDimensions,
        unsigned step,
        WriterEvent event,
        bool lastCall)
    {
        if ((event == WRITER_STEP_FINISHED) && (step % period != 0)) {
            return;
        }

        //if(this->region.boundingBox().origin == Coord<2>(0,0))
        /*
        {
            std::cout
                << "BOVWriter ... "
                << step << " "
                << this->region.boundingBox() << " "
                << validRegion.boundingBox() << " "
                << (lastCall ? std::string("true") : "false")
                << "\n";
        }
        */


        //if(validRegion.boundinBox()

        writeHeader(step, globalDimensions);
        writeRegion(step, globalDimensions, grid, validRegion);
    }


private:
    Coord<3> brickletDim;
    
    friend class boost::serialization::access;
    template <typename ARCHIVE>
    void serialize(ARCHIVE & ar, unsigned)
    {
        ar & boost::serialization::base_object<ParallelWriter<CELL_TYPE> >(*this);
        ar & brickletDim;
    }

    std::string filename(const unsigned& step, const std::string& suffix) const
    {
        std::ostringstream buf;
        buf << prefix << "." << std::setfill('0') << std::setw(5) << step << "." << suffix;
        return buf.str();
    }

    void writeHeader(const unsigned& step, const Coord<DIM>& dimensions)
    {
        /*
        MPI::File file = MPIIO<CELL_TYPE, Topology>::openFileForWrite(
            filename(step, "bov"), comm);

        if (comm.Get_rank() == 0) {
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
        */
    }

    template<typename GRID_TYPE>
    void writeRegion(
        const unsigned& step,
        const Coord<DIM>& dimensions,
        const GRID_TYPE& grid,
        const Region<DIM>& region)
    {
        /*
        MPI::File file = MPIIO<CELL_TYPE, Topology>::openFileForWrite(
            filename(step, "data"), comm);
        MPI::Aint varLength = MPIIO<CELL_TYPE, Topology>::getLength(datatype);
        SuperVector<VariableType> buffer;

        for (typename Region<DIM>::StreakIterator i = region.beginStreak();
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

            if (buffer.size() != effectiveLength) {
                buffer = SuperVector<VariableType>(effectiveLength);
            }

            for (int i = 0; i < effectiveLength; i += dataComponents) {
                SELECTOR_TYPE()(grid.at(walker), &buffer[i]);
                walker.x()++;
            }
            file.Write(&buffer[0], effectiveLength, datatype);
        }

        file.Close();
        */
    }
};

}

#endif
#endif
