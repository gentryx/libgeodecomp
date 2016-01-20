#ifndef LIBGEODECOMP_IO_SERIALBOVWRITER_H
#define LIBGEODECOMP_IO_SERIALBOVWRITER_H

#include <libgeodecomp/io/bovoutput.h>
#include <libgeodecomp/io/writer.h>
#include <libgeodecomp/misc/clonable.h>

#include <iomanip>
#include <iostream>
#include <stdexcept>

namespace LibGeoDecomp {

/**
 * This class writes simulation snapshots compatible with VisIt's
 * Brick of Values (BOV) format using one file per partition. Uses a
 * selector which maps a cell to a primitive data type so that it can
 * be fed into VisIt.
 */
template<typename CELL_TYPE, typename TOPOLOGY = typename APITraits::SelectTopology<CELL_TYPE>::Value>
class SerialBOVWriter : public Clonable<Writer<CELL_TYPE>, SerialBOVWriter<CELL_TYPE> >
{
public:
    friend class SerialBOVWriterTest;

    typedef TOPOLOGY Topology;
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
        BOVOutput<CELL_TYPE, DIM>::writeHeader(filename1, filename2, step, grid.boundingBox(), brickletDim, selector);
        BOVOutput<CELL_TYPE, DIM>::writeGrid(filename2, grid, selector);
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
};

}

#endif
