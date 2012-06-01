#include <libgeodecomp/config.h>
#ifdef LIBGEODECOMP_FEATURE_MPI
#ifndef _libgeodecomp_io_parallelmpiiowriter_h_
#define _libgeodecomp_io_parallelmpiiowriter_h_

#include <libgeodecomp/io/mpiio.h>
#include <libgeodecomp/io/parallelwriter.h>
#include <libgeodecomp/mpilayer/typemaps.h>

namespace LibGeoDecomp {

template<typename CELL_TYPE>
class ParallelMPIIOWriter : public ParallelWriter<CELL_TYPE>
{    
public:
    friend class ParallelMPIIOWriterTest;

    static const int DIM = CELL_TYPE::Topology::DIMENSIONS;

    using ParallelWriter<CELL_TYPE>::distSim;
    using ParallelWriter<CELL_TYPE>::period;
    using ParallelWriter<CELL_TYPE>::prefix;

    ParallelMPIIOWriter(
        const std::string& prefix, 
        DistributedSimulator<CELL_TYPE> *sim, 
        const unsigned& period, 
        const MPI::Intracomm& communicator = MPI::COMM_WORLD,
        MPI::Datatype mpiDatatype = Typemaps::lookup<CELL_TYPE>()) :
        ParallelWriter<CELL_TYPE>(prefix, sim, period),
        comm(communicator),
        datatype(mpiDatatype)
    {}

    virtual void initialized() 
    {
        writeGrid();
    }

    virtual void stepFinished()
    {
        if ((distSim->getStep() % period) == 0) 
            writeGrid();
    }

    virtual void allDone()
    {
        writeGrid();
    }

private:
    MPI::Intracomm comm;
    MPI::Datatype datatype;

    std::string filename(const unsigned& step) const 
    {
        std::ostringstream buf;
        buf << prefix << std::setfill('0') << std::setw(5) << step << ".mpiio";
        return buf.str();
    }

    void writeGrid()
    {
        const Region<DIM> *region;
        const typename DistributedSimulator<CELL_TYPE>::GridType *grid;
        distSim->getGridFragment(&grid, &region);
        unsigned step = distSim->getStep();

        MPIIO<CELL_TYPE>::writeRegion(
            *grid, 
            distSim->getInitializer()->gridDimensions(),
            step,
            distSim->getInitializer()->maxSteps(),
            filename(step),
            *region,
            datatype,
            comm);
    }
};

}

#endif
#endif
