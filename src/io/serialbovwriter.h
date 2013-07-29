#ifndef LIBGEODECOMP_IO_SERIALBOVWRITER_H
#define LIBGEODECOMP_IO_SERIALBOVWRITER_H

#include <iomanip>

#include <libgeodecomp/io/writer.h>

#include <iostream>
#include <fstream>

namespace LibGeoDecomp {

/**
 * writes simulation snapshots compatible with VisIt's Brick of Values
 * (BOV) format using one file per partition. Uses a selector which maps a cell to a
 * primitive data type so that it can be fed into VisIt.
 */

template<typename CELL_TYPE, typename SELECTOR_TYPE>
class SerialBOVWriter : public Writer<CELL_TYPE>
{
#ifdef LIBGEODECOMP_FEATURE_BOOST_SERIALIZATION
    friend class boost::serialization::access;
#endif
public:
    typedef typename CELL_TYPE::Topology Topology;
    typedef typename SELECTOR_TYPE::VariableType VariableType;
    typedef Grid<CELL_TYPE, Topology> GridType;

    static const int DIM = CELL_TYPE::Topology::DIM;

    using Writer<CELL_TYPE>::period;
    using Writer<CELL_TYPE>::prefix;

    SerialBOVWriter(
        const std::string& prefix,
        const unsigned period,
        const Coord<3>& brickletDim = Coord<3>()) :
        Writer<CELL_TYPE>(prefix, period),
        brickletDim(brickletDim)
    {}

    Writer<CELL_TYPE> *clone()
    {
        return new SerialBOVWriter(this->prefix, this->period, brickletDim);
    }

    virtual void stepFinished(const GridType& grid, unsigned step, WriterEvent event)
    {
        if ((event == WRITER_STEP_FINISHED) && (step % period != 0)) {
            return;
        }

        writeHeader(step, grid.getDimensions());

        writeRegion(step, grid);
    }

private:
    Coord<3> brickletDim;

#ifdef LIBGEODECOMP_FEATURE_BOOST_SERIALIZATION
    SerialBOVWriter()
    {}

    template <typename ARCHIVE>
    void serialize(ARCHIVE& ar, unsigned)
    {
        ar & boost::serialization::base_object<Writer<CELL_TYPE> >(*this);
        ar & brickletDim;
    }
#endif

    std::string filename(unsigned step, const std::string& suffix) const
    {
        std::ostringstream buf;
        buf
            << prefix << "."
            << std::setfill('0') << std::setw(5) << step
            << "." << suffix;

        return buf.str();
    }

    void writeHeader(unsigned step, const Coord<DIM>& dimensions)
    {
        std::ofstream file;

        file.open(filename(step, "bov").c_str());

        // BOV only accepts 3D data, so we'll have to inflate 1D
        // and 2D dimensions.
        Coord<DIM> c = dimensions;
        Coord<3> bovDim = Coord<3>::diagonal(1);
        for (int i = 0; i < DIM; ++i) {
            bovDim[i] = c[i];
        }

        Coord<3> bricDim = (brickletDim == Coord<3>()) ? bovDim : brickletDim;

        file << "TIME: " << step << "\n"
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

        file.close();
    }

    template<typename GRID_TYPE>
    void writeRegion(
        unsigned step,
        const GRID_TYPE& grid)
    {
        std::ofstream file;

        file.open(
            filename(step, "data").c_str(), std::ios::binary);

        SuperVector<VariableType> buffer;

        Coord<DIM> dimensions = grid.getDimensions();

        std::size_t dataComponents = SELECTOR_TYPE::dataComponents();
        std::size_t length = dimensions.prod();
        std::size_t effectiveLength = dataComponents * length;
        buffer.resize(effectiveLength);

        CoordBox<DIM> boundingBox = grid.boundingBox();
        std::size_t j = 0;
        for(
            typename CoordBox<DIM>::Iterator i = boundingBox.begin();
            i != boundingBox.end();
            ++i)
        {
            SELECTOR_TYPE()(grid.at(*i), &buffer[j]);
            j += dataComponents;
        }

        file.write(
            reinterpret_cast<char *>(&buffer[0]),
            effectiveLength * sizeof(VariableType));

        file.close();
    }
};

}

#endif
