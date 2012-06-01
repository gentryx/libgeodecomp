#ifndef _libgeodecomp_io_tracingwriter_h_
#define _libgeodecomp_io_tracingwriter_h_

#include <boost/date_time/posix_time/posix_time.hpp>
#include <iostream>

#include <libgeodecomp/io/parallelwriter.h>
#include <libgeodecomp/io/writer.h>

namespace LibGeoDecomp {

template<typename CELL_TYPE>
class TracingWriter : public Writer<CELL_TYPE>, public ParallelWriter<CELL_TYPE>
{
public:
    typedef boost::posix_time::ptime Time;
    typedef boost::posix_time::time_duration Duration;

    using Writer<CELL_TYPE>::sim;
    using ParallelWriter<CELL_TYPE>::distSim;

    TracingWriter(MonolithicSimulator<CELL_TYPE> *sim, 
                  const unsigned& period = 1, 
                  std::ostream& _stream = std::cout) :
        Writer<CELL_TYPE>("foo", sim, period), 
        ParallelWriter<CELL_TYPE>("foo", 0, period), 
        stream(_stream) 
    {}

    TracingWriter(DistributedSimulator<CELL_TYPE> *sim, 
                  const unsigned& period = 1, 
                  std::ostream& _stream = std::cout) :
        Writer<CELL_TYPE>("foo", 0, period), 
        ParallelWriter<CELL_TYPE>("foo", sim, period), 
        stream(_stream) 
    {}

    virtual void initialized()
    {
        startTime = currentTime();
        stream << "TracingWriter::initialized()\n";
        printTime();
    }

    virtual void stepFinished()
    {
        int step;
        unsigned maxSteps;
        if (sim) {
            step     = sim->getStep();
            maxSteps = sim->getInitializer()->maxSteps();
        } else {
            step     = distSim->getStep();
            maxSteps = distSim->getInitializer()->maxSteps();
        }

        if (step % Writer<CELL_TYPE>::period != 0) return;

        Time now = currentTime();
        Duration delta = now - startTime;
        Duration remaining = delta * (maxSteps - step) / step;
        Time eta = now + remaining;
        Coord<CELL_TYPE::Topology::DIMENSIONS> coordBox;
        
        if (sim) {
            step     = sim->getStep();
            maxSteps = sim->getInitializer()->maxSteps();
            coordBox = sim->getInitializer()->gridDimensions();
        } else {
            step     = distSim->getStep();
            maxSteps = distSim->getInitializer()->maxSteps();
            coordBox = distSim->getInitializer()->gridDimensions();
        }

        double updates = 1.0 * step * CELL_TYPE::nanoSteps() * coordBox.prod();
        double glups = updates / delta.total_microseconds() * 1000.0 * 1000.0 
            / 1000.0 / 1000.0 / 1000.0;
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

    virtual void allDone()
    {
        Duration delta = currentTime() - startTime;
        stream << "TracingWriter::allDone()\n"
                << "  total time: " << boost::posix_time::to_simple_string(delta) << "\n";
        printTime();    
    }


private:
    std::ostream& stream;
    Time startTime;

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

};

#endif
