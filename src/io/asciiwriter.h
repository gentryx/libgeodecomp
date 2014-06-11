#ifndef LIBGEODECOMP_IO_ASCIIWRITER_H
#define LIBGEODECOMP_IO_ASCIIWRITER_H

#include <libgeodecomp/io/ioexception.h>
#include <libgeodecomp/io/plotter.h>
#include <libgeodecomp/misc/clonable.h>
#include <libgeodecomp/storage/image.h>

#include <string>
#include <cerrno>
#include <fstream>
#include <iomanip>

namespace LibGeoDecomp {

/**
 * An output plugin for writing text files. Uses the same selector
 * infrastucture as the BOVWriter.
 *
 * fixme: actually use selector here as advertised
 */
template<typename CELL_TYPE, typename ATTRIBUTE_SELECTOR>
class ASCIIWriter : public Clonable<Writer<CELL_TYPE>, ASCIIWriter<CELL_TYPE, ATTRIBUTE_SELECTOR> >
{
public:
    friend class ASCIIWriterTest;
    typedef typename Writer<CELL_TYPE>::GridType GridType;
    typedef typename APITraits::SelectTopology<CELL_TYPE>::Value Topology;
    static const int DIM = Topology::DIM;
    using Writer<CELL_TYPE>::period;
    using Writer<CELL_TYPE>::prefix;

    explicit ASCIIWriter(
        const std::string& prefix,
        const unsigned period = 1) :
        Clonable<Writer<CELL_TYPE>, ASCIIWriter<CELL_TYPE, ATTRIBUTE_SELECTOR> >(prefix, period)
    {}

    virtual void stepFinished(const GridType& grid, unsigned step, WriterEvent event)
    {
        if ((event == WRITER_STEP_FINISHED) && (step % period != 0)) {
            return;
        }

        std::ostringstream filename;
        filename << prefix << "." << std::setfill('0') << std::setw(4)
                 << step << ".ascii";
        std::ofstream outfile(filename.str().c_str());
        if (!outfile) {
            throw FileOpenException(filename.str());
        }

        CoordBox<DIM> box = grid.boundingBox();
        for (typename CoordBox<DIM>::Iterator i = box.begin(); i != box.end(); ++i) {
            if ((*i)[0] == 0) {
                for (int d = 0; d < DIM; ++d) {
                    if ((*i)[d] == 0) {
                        outfile << "\n";
                    }
                }
            }
            outfile << ATTRIBUTE_SELECTOR()(grid.get(*i)) << " ";
        }

        if (!outfile.good()) {
            throw FileWriteException(filename.str());
        }
        outfile.close();
    }
};

}

#endif
