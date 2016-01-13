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
    HPX_SERIALIZATION_POLYMORPHIC_TEMPLATE_SEMIINTRUSIVE(SerialBOVWriter)

    friend class PolymorphicSerialization;
    friend class BoostSerialization;
    friend class HPXSerialization;
    friend class SerialBOVWriterTest;

    typedef typename APITraits::SelectTopology<CELL_TYPE>::Value Topology;
    typedef typename Writer<CELL_TYPE>::GridType GridType;

    static const int DIM = Topology::DIM;

    using Writer<CELL_TYPE>::period;
    using Writer<CELL_TYPE>::prefix;

    explicit
    SerialBOVWriter(
        const Selector<CELL_TYPE>& selector = Selector<CELL_TYPE>(),
        const std::string& prefix = "serial_bov_writer_output",
        const unsigned period = 1,
        const Coord<3>& brickletDim = Coord<3>()) :
        Clonable<Writer<CELL_TYPE>, SerialBOVWriter<CELL_TYPE> >(prefix, period),
        selector(selector),
        brickletDim(brickletDim)
    {}

    template<typename MEMBER>
    SerialBOVWriter(
        MEMBER CELL_TYPE:: *member,
        const std::string& prefix,
        const unsigned period,
        const Coord<3>& brickletDim = Coord<3>()) :
        Clonable<Writer<CELL_TYPE>, SerialBOVWriter<CELL_TYPE> >(prefix, period),
        selector(member, prefix),
        brickletDim(brickletDim)
    {}

    void stepFinished(const GridType& grid, unsigned step, WriterEvent event)
    {
        if ((event == WRITER_STEP_FINISHED) && (step % period != 0)) {
            return;
        }

        std::string filename1 = filename(step, "bov");
        std::string filename2 = filename(step, "data");
        writeHeader(filename1, filename2, step, grid.dimensions(), brickletDim, selector);
        writeRegion(filename2, grid, brickletDim, selector);
    }

private:
    Selector<CELL_TYPE> selector;
    Coord<3> brickletDim;

    std::string filename(unsigned step, const std::string& suffix)
    {
        std::ostringstream buf;
        buf << prefix << "."
            << std::setfill('0') << std::setw(5) << step
            << "." << suffix;

        return buf.str();
    }

    static void writeHeader(
        std::string filenameBoV,
        std::string filenameData,
        int step,
        const Coord<DIM>& dimensions,
        const Coord<3>& brickletDim,
        const Selector<CELL_TYPE>& selector)
    {
        std::ofstream file;
        file.open(filenameBoV.c_str());

        // BOV only accepts 3D data, so we'll have to inflate 1D
        // and 2D dimensions.
        Coord<DIM> c = dimensions;
        Coord<3> bovDim = Coord<3>::diagonal(1);
        for (int i = 0; i < DIM; ++i) {
            bovDim[i] = c[i];
        }

        Coord<3> bricDim = (brickletDim == Coord<3>()) ? bovDim : brickletDim;

        file << "TIME: " << step << "\n"
             << "DATA_FILE: " << filenameData << "\n"
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
    static void writeRegion(
        std::string filename,
        const GRID_TYPE& grid,
        const Coord<3>& brickletDim,
        const Selector<CELL_TYPE>& selector)
    {
        std::ofstream file;
        file.open(filename.c_str(), std::ios::binary);
        if (!file.good()) {
            throw std::runtime_error("could not open output file");
        }

        std::vector<char> buffer;
        Coord<DIM> dimensions = grid.dimensions();
        std::size_t length = dimensions.x();
        std::size_t byteSize = length * selector.sizeOfExternal();
        buffer.resize(byteSize);

        CoordBox<DIM> boundingBox = grid.boundingBox();
        for (typename CoordBox<DIM>::StreakIterator i = boundingBox.beginStreak();
             i != boundingBox.endStreak();
             ++i) {
            Streak<DIM> s(*i);

            Region<DIM> tempRegion;
            tempRegion << s;
            grid.saveMemberUnchecked(&buffer[0], MemoryLocation::HOST, selector, tempRegion);

            file.write(
                &buffer[0],
                byteSize);
        }

        file.close();
    }
};

class Serialization;

}

#endif
