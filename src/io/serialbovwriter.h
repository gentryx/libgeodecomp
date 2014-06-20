#ifndef LIBGEODECOMP_IO_SERIALBOVWRITER_H
#define LIBGEODECOMP_IO_SERIALBOVWRITER_H

#include <libgeodecomp/io/writer.h>
#include <libgeodecomp/misc/clonable.h>

#include <iomanip>
#include <iostream>
#include <fstream>
#include <stdexcept>

namespace LibGeoDecomp {

/**
 * writes simulation snapshots compatible with VisIt's Brick of Values
 * (BOV) format using one file per partition. Uses a selector which maps a cell to a
 * primitive data type so that it can be fed into VisIt.
 */
template<typename CELL_TYPE>
class SerialBOVWriter : public Clonable<Writer<CELL_TYPE>, SerialBOVWriter<CELL_TYPE> >
{
public:
    friend class Serialization;

    typedef typename APITraits::SelectTopology<CELL_TYPE>::Value Topology;
    typedef typename Writer<CELL_TYPE>::GridType GridType;

    static const int DIM = Topology::DIM;

    using Writer<CELL_TYPE>::period;
    using Writer<CELL_TYPE>::prefix;

    SerialBOVWriter(
        const Selector<CELL_TYPE>& selector,
        const std::string& prefix,
        const unsigned period,
        const Coord<3>& brickletDim = Coord<3>()) :
        Clonable<Writer<CELL_TYPE>, SerialBOVWriter<CELL_TYPE> >(prefix, period),
        selector(selector),
        brickletDim(brickletDim)
    {}

    void stepFinished(const GridType& grid, unsigned step, WriterEvent event)
    {
        if ((event == WRITER_STEP_FINISHED) && (step % period != 0)) {
            return;
        }

        writeHeader(step, grid.dimensions());
        writeRegion(step, grid);
    }

private:
    Selector<CELL_TYPE> selector;
    Coord<3> brickletDim;

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

        file.close();
    }

    template<typename GRID_TYPE>
    void writeRegion(
        unsigned step,
        const GRID_TYPE& grid)
    {
        std::ofstream file;
        file.open(filename(step, "data").c_str(), std::ios::binary);
        if (!file.good()) {
            throw std::runtime_error("could not open output file");
        }

        std::vector<char> buffer;
        Coord<DIM> dimensions = grid.dimensions();
        std::size_t length = dimensions.x();
        std::size_t byteSize = length * selector.sizeOfExternal();
        buffer.resize(byteSize);

        CoordBox<DIM> boundingBox = grid.boundingBox();
        boundingBox.dimensions.x() = 1;
        for (typename CoordBox<DIM>::Iterator i = boundingBox.begin();
             i != boundingBox.end();
             ++i) {
            Streak<DIM> s(*i, i->x() + dimensions.x());

            Region<DIM> tempRegion;
            tempRegion << s;
            grid.saveMemberUnchecked(&buffer[0], selector, tempRegion);

            file.write(
                &buffer[0],
                byteSize);
        }

        file.close();
    }
};

class Serialization;

}


namespace boost {
namespace serialization {

template<typename ARCHIVE, typename CELL_TYPE>
inline
static void serialize(ARCHIVE& archive, LibGeoDecomp::SerialBOVWriter<CELL_TYPE>& object, const unsigned /*version*/);

template<class Archive, typename CELL_TYPE>
inline void load_construct_data(
    Archive& archive, LibGeoDecomp::SerialBOVWriter<CELL_TYPE> *object, const unsigned version)
{
    ::new(object)LibGeoDecomp::SerialBOVWriter<CELL_TYPE>("", 1);
    serialize(archive, *object, version);
}

}
}

#endif
