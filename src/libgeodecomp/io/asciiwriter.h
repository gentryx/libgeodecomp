#ifndef LIBGEODECOMP_IO_ASCIIWRITER_H
#define LIBGEODECOMP_IO_ASCIIWRITER_H

#include <libgeodecomp/io/ioexception.h>
#include <libgeodecomp/io/plotter.h>
#include <libgeodecomp/misc/clonable.h>
#include <libgeodecomp/storage/image.h>

// Kill warning 4514 in system headers
#ifdef _MSC_BUILD
#pragma warning( push )
#pragma warning( disable : 4514 )
#endif

#include <string>
#include <cerrno>
#include <fstream>
#include <iomanip>

#ifdef _MSC_BUILD
#pragma warning( pop )
#endif

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
            typename SharedPtr<Filter<CELL_TYPE, MEMBER, char> >::Type(dumpingFilter));
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
            grid.saveMemberUnchecked(0, MemoryLocation::HOST, selector, region);
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

        void copyStreakInImpl(
            const EXTERNAL * /* source */,
            MemoryLocation::Location /* sourceLocation */,
            MEMBER * /* target */,
            MemoryLocation::Location /* targetLocation */,
            const std::size_t /* num */,
            const std::size_t /* stride */)
        {
            throw std::logic_error("this filter is meant for output only");
        }

        void copyStreakOutImpl(
            const MEMBER *source,
            MemoryLocation::Location sourceLocation,
            EXTERNAL * /*target*/,
            MemoryLocation::Location targetLocation,
            const std::size_t num,
            const std::size_t stride)
        {
            checkMemoryLocations(sourceLocation, targetLocation);

            for (std::size_t i = 0; i < num; ++i) {
                *outfile << source[i] << " ";
            }
        }

        void copyMemberInImpl(
            const EXTERNAL * /* source */,
            MemoryLocation::Location /* sourceLocation */,
            CELL */* target */,
            MemoryLocation::Location /* targetLocation */,
            std::size_t /* num */,
            MEMBER CELL:: * /* memberPointer */)
        {
            throw std::logic_error("this filter is meant for output only");
        }

        void copyMemberOutImpl(
            const CELL *source,
            MemoryLocation::Location sourceLocation,
            EXTERNAL * /*target*/,
            MemoryLocation::Location targetLocation,
            std::size_t num,
            MEMBER CELL:: *memberPointer)
        {
            checkMemoryLocations(sourceLocation, targetLocation);

            for (std::size_t i = 0; i < num; ++i) {
                *outfile << source[i].*memberPointer << " ";
            }
        }

    private:
        void checkMemoryLocations(
            MemoryLocation::Location sourceLocation,
            MemoryLocation::Location targetLocation)
        {
            if ((sourceLocation == MemoryLocation::CUDA_DEVICE) ||
                (targetLocation == MemoryLocation::CUDA_DEVICE)) {
                throw std::logic_error("DefaultFilter can't access CUDA device memory");
            }

            if ((sourceLocation != MemoryLocation::HOST) ||
                (targetLocation != MemoryLocation::HOST)) {
                throw std::invalid_argument("unknown combination of source and target memory locations");
            }

        }
    };

    OutputDelegate *filter;
    Selector<CELL_TYPE> selector;
};

}

#endif
