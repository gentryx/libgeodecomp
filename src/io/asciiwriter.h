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
 */
template<typename CELL_TYPE>
class ASCIIWriter : public Clonable<Writer<CELL_TYPE>, ASCIIWriter<CELL_TYPE> >
{
public:
    friend class ASCIIWriterTest;
    typedef typename Writer<CELL_TYPE>::GridType GridType;
    typedef typename APITraits::SelectTopology<CELL_TYPE>::Value Topology;
    static const int DIM = Topology::DIM;
    using Writer<CELL_TYPE>::period;
    using Writer<CELL_TYPE>::prefix;

    template<typename MEMBER>
    explicit ASCIIWriter(
        const std::string& prefix,
        MEMBER CELL_TYPE:: *memberPointer,
        const unsigned period = 1) :
        Clonable<Writer<CELL_TYPE>, ASCIIWriter<CELL_TYPE> >(prefix, period)
    {
        FileDumpingFilter<CELL_TYPE, MEMBER, char> *dumpingFilter =
            new FileDumpingFilter<CELL_TYPE, MEMBER, char>();
        filter = dumpingFilter;
        selector = Selector<CELL_TYPE>(
            memberPointer,
            "unused member name",
            boost::shared_ptr<Filter<CELL_TYPE, MEMBER, char> >(dumpingFilter));
    }

    virtual void stepFinished(const GridType& grid, unsigned step, WriterEvent event)
    {
        if ((event == WRITER_STEP_FINISHED) && (step % period != 0)) {
            return;
        }

        std::ostringstream filename;
        filename << prefix << "." << std::setfill('0') << std::setw(4) << step << ".ascii";
        std::ofstream outfile(filename.str().c_str());
        if (!outfile) {
            throw FileOpenException(filename.str());
        }
        filter->setFile(&outfile);

        CoordBox<DIM> box = grid.boundingBox();
        for (typename CoordBox<DIM>::StreakIterator i = box.beginStreak(); i != box.endStreak(); ++i) {
            if ((*i).origin[0] == 0) {
                for (int d = 0; d < DIM; ++d) {
                    if ((*i).origin[d] == 0) {
                        outfile << "\n";
                    }
                }
            }

            Region<DIM> region;
            region << *i;
            grid.saveMemberUnchecked(0, selector, region);
        }

        if (!outfile.good()) {
            throw FileWriteException(filename.str());
        }
        outfile.close();
    }

private:
    class OutputDelegate
    {
    public:
        OutputDelegate() :
            outfile(0)
        {}

        void setFile(std::ofstream *newOutfile)
        {
            outfile = newOutfile;
        }

    protected:
        std::ofstream *outfile;
    };

    template<typename CELL, typename MEMBER, typename EXTERNAL>
    class FileDumpingFilter : public Filter<CELL, MEMBER, EXTERNAL>, public OutputDelegate
    {
    public:
        using OutputDelegate::outfile;

        void copyStreakInImpl(const EXTERNAL *source, MEMBER *target, const std::size_t num, const std::size_t stride)
        {
            throw std::logic_error("this filter is meant for output only");
        }

        void copyStreakOutImpl(const MEMBER *source, EXTERNAL */*target*/, const std::size_t num, const std::size_t stride)
        {
            for (std::size_t i = 0; i < num; ++i) {
                *outfile << source[i] << " ";
            }
        }

        void copyMemberInImpl(
            const EXTERNAL *source, CELL *target, std::size_t num, MEMBER CELL:: *memberPointer)
        {
            throw std::logic_error("this filter is meant for output only");
        }

        void copyMemberOutImpl(
            const CELL *source, EXTERNAL */*target*/, std::size_t num, MEMBER CELL:: *memberPointer)
        {
            for (std::size_t i = 0; i < num; ++i) {
                *outfile << source[i].*memberPointer << " ";
            }
        }
    };

    OutputDelegate *filter;
    Selector<CELL_TYPE> selector;

};

}

#endif
