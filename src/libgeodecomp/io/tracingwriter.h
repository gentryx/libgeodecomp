#ifndef LIBGEODECOMP_IO_TRACINGWRITER_H
#define LIBGEODECOMP_IO_TRACINGWRITER_H

#include <libflatarray/macros.hpp>

LIBFLATARRAY_DISABLE_SYSTEM_HEADER_WARNINGS_PRE
#include <iostream>
#include <stdexcept>
LIBFLATARRAY_DISABLE_SYSTEM_HEADER_WARNINGS_POST

#include <libgeodecomp/io/parallelwriter.h>
#include <libgeodecomp/io/timestringconversion.h>
#include <libgeodecomp/io/writer.h>
#include <libgeodecomp/misc/clonable.h>
#include <libgeodecomp/misc/scopedtimer.h>

namespace LibGeoDecomp {

#ifdef _MSC_BUILD
#pragma warning( push )
#pragma warning( disable : 4626 4820 5027 )
#endif

/**
 * The purpose of the TracingWriter is out output performance data
 * which allows the user to gauge execution time (current, remaining,
 * estimated time of arrival (ETA)) and performance (GLUPS, memory
 * bandwidth).
 */
template<typename CELL_TYPE>
class TracingWriter :
        public Clonable<Writer<CELL_TYPE>, TracingWriter<CELL_TYPE> >,
        public Clonable<ParallelWriter<CELL_TYPE>, TracingWriter<CELL_TYPE> >
{
public:
    using Writer<CELL_TYPE>::NANO_STEPS;

    typedef double TimeType;
    typedef double DurationType;
    typedef typename Writer<CELL_TYPE>::GridType WriterGridType;
    typedef typename ParallelWriter<CELL_TYPE>::GridType ParallelWriterGridType;
    typedef typename ParallelWriter<CELL_TYPE>::Topology Topology;

    static const int DIM = Topology::DIM;
    static const int OUTPUT_ON_ALL_RANKS = -1;

    explicit TracingWriter(
        const unsigned period = 1,
        const unsigned maxSteps = 1,
        int outputRank = OUTPUT_ON_ALL_RANKS,
        std::ostream& stream = std::cerr) :
        Clonable<Writer<CELL_TYPE>, TracingWriter<CELL_TYPE> >("", period),
        Clonable<ParallelWriter<CELL_TYPE>, TracingWriter<CELL_TYPE> >("", period),
        outputRank(outputRank),
        stream(stream),
        lastStep(0),
        maxSteps(maxSteps)
    {}

    virtual void stepFinished(const WriterGridType& grid, unsigned step, WriterEvent event)
    {
        stepFinished(step, grid.dimensions(), event);
    }

    virtual void stepFinished(
        const ParallelWriterGridType& /* grid */,
        const Region<DIM>& /* validRegion */,
        const Coord<DIM>& globalDimensions,
        unsigned step,
        WriterEvent event,
        std::size_t rank,
        bool lastCall)
    {
        if (lastCall && ((outputRank == OUTPUT_ON_ALL_RANKS) || (outputRank == (int)rank))) {
            stepFinished(step, globalDimensions, event);
        }
    }

private:
    int outputRank;
    std::ostream& stream;
    TimeType startTime;
    unsigned lastStep;
    unsigned maxSteps;

    void stepFinished(unsigned step, const Coord<DIM>& globalDimensions, WriterEvent event)
    {
        DurationType delta;

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
                   << "  total time: " << TimeStringConversion::renderDuration(delta) << "\n";
            printTime();
            break;
        default:
            throw std::invalid_argument("unknown event");
        }
    }

    void normalStepFinished(unsigned step, const Coord<DIM>& globalDimensions)
    {
        if (step % Writer<CELL_TYPE>::period != 0) {
            return;
        }

        TimeType now = currentTime();
        DurationType delta = now - startTime;
        DurationType remaining = delta * (maxSteps - step) / step;
        TimeType eta = now + remaining;

        double updates = 1.0 * step * NANO_STEPS * globalDimensions.prod();
        double glups = updates / delta / 1000.0 / 1000.0 / 1000.0;
        double bandwidth = glups * 2 * sizeof(CELL_TYPE);

        stream << "TracingWriter::stepFinished()\n"
               << "  step: " << step << " of " << maxSteps << "\n"
               << "  elapsed: " << delta << "\n"
               << "  remaining: "
               << TimeStringConversion::renderDuration(remaining) << "\n"
               << "  ETA:  "
               << TimeStringConversion::renderDuration(eta) << "\n"
               << "  speed: " << glups << " GLUPS\n"
               << "  effective memory bandwidth " << bandwidth << " GB/s\n";
        printTime();
    }

    void printTime() const
    {
        stream << "  time: " << TimeStringConversion::renderDuration(currentTime()) << "\n";
        stream.flush();
    }

    TimeType currentTime() const
    {
        return ScopedTimer::time();
    }
};

#ifdef _MSC_BUILD
#pragma warning( pop )
#endif

}

#endif
