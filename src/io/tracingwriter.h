#ifndef LIBGEODECOMP_IO_TRACINGWRITER_H
#define LIBGEODECOMP_IO_TRACINGWRITER_H

#include <boost/date_time/posix_time/posix_time.hpp>
#include <iostream>
#include <stdexcept>

#include <libgeodecomp/io/parallelwriter.h>
#include <libgeodecomp/io/writer.h>

namespace LibGeoDecomp {

template<typename CELL_TYPE>
class TracingWriter : public Writer<CELL_TYPE>, public ParallelWriter<CELL_TYPE>
{
public:
#ifdef LIBGEODECOMP_FEATURE_BOOST_SERIALIZATION
    friend class boost::serialization::access;
#endif
    typedef boost::posix_time::ptime Time;
    typedef boost::posix_time::time_duration Duration;
    typedef typename Writer<CELL_TYPE>::GridType WriterGridType;
    typedef typename ParallelWriter<CELL_TYPE>::GridType ParallelWriterGridType;
    static const int DIM = CELL_TYPE::Topology::DIM;

    TracingWriter(
        const unsigned period,
        const unsigned maxSteps,
        std::ostream& stream = std::cout) :
        Writer<CELL_TYPE>("", period),
        ParallelWriter<CELL_TYPE>("", period),
        stream(stream),
        lastStep(0),
        maxSteps(maxSteps)
    {}


    ParallelWriter<CELL_TYPE> * clone()
    {
        return new TracingWriter(Writer<CELL_TYPE>::period, maxSteps);
    }

    virtual void stepFinished(const WriterGridType& grid, unsigned step, WriterEvent event)
    {
        stepFinished(step, grid.dimensions(), event);
    }

    virtual void stepFinished(
        const ParallelWriterGridType& grid,
        const Region<DIM>& validRegion,
        const Coord<DIM>& globalDimensions,
        unsigned step,
        WriterEvent event,
        std::size_t rank,
        bool lastCall)
    {
        if (lastCall) {
            stepFinished(step, globalDimensions, event);
        }
    }

private:
    std::ostream& stream;
    Time startTime;
    unsigned lastStep;
    unsigned maxSteps;

#ifdef LIBGEODECOMP_FEATURE_BOOST_SERIALIZATION
    template <typename ARCHIVE>
    void serialize(ARCHIVE& ar, unsigned)
    {
        ar & boost::serialization::base_object<Writer<CELL_TYPE> >(*this);
        ar & boost::serialization::base_object<ParallelWriter<CELL_TYPE> >(*this);
        ar & lastStep;
        ar & maxSteps;
    }

    TracingWriter() :
        stream(std::cout)
    {}
#endif

    void stepFinished(unsigned step, const Coord<DIM>& globalDimensions, WriterEvent event)
    {
        Duration delta;

        switch (event) {
        case WRITER_INITIALIZED:
            startTime = currentTime();
            stream << "TracingWriter::initialized()\n";
            printTime();
            lastStep = step;
            break;
        case WRITER_STEP_FINISHED:
            normalStepFinished(step, globalDimensions);
            break;
        case WRITER_ALL_DONE:
            delta = currentTime() - startTime;
            stream << "TracingWriter::allDone()\n"
                   << "  total time: " << boost::posix_time::to_simple_string(delta) << "\n";
            printTime();
            break;
        default:
            throw std::invalid_argument("unknown event");
            break;
        }
    }

    void normalStepFinished(unsigned step, const Coord<DIM>& globalDimensions)
    {
        if (step % Writer<CELL_TYPE>::period != 0) {
            return;
        }

        Time now = currentTime();
        Duration delta = now - startTime;
        Duration remaining = delta * (maxSteps - step) / step;
        Time eta = now + remaining;

        double updates = 1.0 * step * CELL_TYPE::nanoSteps() * globalDimensions.prod();
        double seconds = delta.total_microseconds() / 1000.0 / 1000.0;
        double glups = updates / seconds / 1000.0 / 1000.0 / 1000.0;
        double bandwidth = glups * 2 * sizeof(CELL_TYPE);

        stream << "TracingWriter::stepFinished()\n"
               << "  step: " << step << " of " << maxSteps << "\n"
               << "  elapsed: " << delta << "\n"
               << "  remaining: "
               << boost::posix_time::to_simple_string(remaining) << "\n"
               << "  ETA:  "
               << boost::posix_time::to_simple_string(eta) << "\n"
               << "  speed: " << glups << " GLUPS\n"
               << "  effective memory bandwidth " << bandwidth << " GB/s\n";
        printTime();
    }

    void printTime() const
    {
        stream << "  time: " << boost::posix_time::to_simple_string(currentTime()) << "\n";
        stream.flush();
    }

    Time currentTime() const
    {
        return boost::posix_time::microsec_clock::local_time();
    }
};

}

#endif
